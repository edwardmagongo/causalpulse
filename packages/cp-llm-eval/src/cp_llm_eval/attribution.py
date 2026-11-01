"""Matching the numbers in a narrative back to the facts behind them.

The grounded arms need no such thing: a placeholder names its fact, so a
grader knows the value behind it without reading prose. The ungrounded arm
writes values itself, and every grader in `deterministic` reads placeholders,
so without this module they all see an empty statement and score it perfect.

So each numeric span is matched against the table and the statement is
rewritten with the placeholder it would have carried, which lets D1-D5 run on
this arm unchanged. A span that matches nothing is the arm's own finding.

Five outcomes, because they are five different things:

- **exact** -- the span is one fact's rendered form. Nothing to report, and
  this is the only outcome rewritten as a citation.
- **rounded** -- the span is a correct rounding of exactly one fact, at the
  model's own precision rather than the renderer's. The quantity is right and
  the digits shown are the model's choice. Reported as `D0_ROUNDED`, and
  deliberately **not** as `D0`: it is not a value the table fails to support,
  and folding it into the primary endpoint would inflate that endpoint with a
  milder defect.
- **ambiguous** -- two or more facts carry the span's value, so which one it
  cites cannot be recovered. Not a misreading; counted so a reader knows how
  far D1-D5 can be read on this arm.
- **quoted** -- no fact's own value is the span, but the span appears inside
  a fact's prose, as `0.01` does in "a propensity score within 0.01 of 0 or
  1". The table holds the number, so it is not invented and not `D0`; but it
  is not a citable value either, and a placeholder for the fact that quotes
  it would render the whole sentence. So it carries no flag and no citation.
- **unsupported** -- no fact supports the value at any rounding, and no
  fact's prose holds it. This is `D0`, the primary endpoint of the control
  arm.

Where two facts carry the same value -- `est.ipw.point` and the comparison
table's `point_low`, which restates it -- the span is resolved to the one
whose subject the statement names, as a reader would resolve it, and only
stays ambiguous when that leaves more than one. Without that nearly every
value in an estimates statement is ambiguous, and an ambiguous span is not
reconstructed, so D1-D5 would be left with almost nothing to read.

Comparison is against the **rendered** form, because that is what a reader of
a grounded narrative sees and so what the system promises. Thousands
separators and the spacing of `%` and `<` are normalised away first: those
are formatting, not a different value, and counting them as fabrication would
overstate the case for grounding. A quantity spelled in words ("a third",
"twenty units") is not a numeric span and is not matched, the same gap phase
1's verifier has.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from cp_llm.facts import Fact, FactTable
from cp_llm.narrative import Narrative
from cp_llm.render import PLACEHOLDER, format_fact

from cp_llm_eval.deterministic import Flag, StatementGrade, grade_statement

# A value no fact supports: the control arm's primary endpoint.
D0 = "D0"
# Not misreadings. Counted so the report can say what the arm's D1-D5 rest on.
D0_AMBIGUOUS = "D0-ambiguous"
D0_ROUNDED = "D0-rounded"

Outcome = Literal["exact", "rounded", "ambiguous", "quoted", "unsupported"]
# The outcomes that say nothing against the model: the span carries a value
# the table holds.
_SUPPORTED = frozenset({"exact", "quoted"})

# The kinds whose value is a float, and so can be matched at a coarser rounding.
_FLOAT_KINDS = frozenset({"number", "level", "p_value"})
# How many significant figures a rounding match will try. The renderer uses at
# most 4, so a model writing more digits than the table holds is not "rounded".
_MAX_DIGITS = 4

# A number as prose carries it: an optional `<`, an optional sign, digits with
# optional thousands separators, an optional decimal part and an optional `%`.
# The leading guard keeps the match off a number already inside a longer token;
# the trailing one keeps it from stopping mid-number. A following `.` is left
# alone, since the decimal part is taken greedily and a sentence ends in one.
NUMERIC = re.compile(
    r"(?<![\w.])(?:(?P<lt><\s*))?(?P<sign>-)?"
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?"
    r"(?P<pct>\s*%)?(?!\d)"
)


def _normalised(text: str) -> str:
    """The span as the renderer would have written it, formatting aside."""
    text = text.replace(",", "")
    text = re.sub(r"<\s*", "< ", text)
    return re.sub(r"\s*%", "%", text)


def _masked(statement: str) -> str:
    """The statement with each placeholder blanked, so its id is not scanned.

    A rendered narrative holds no placeholders, but a raw one does, and
    `{est.aipw.ci_level}` must not have `5` read out of it.
    """
    return PLACEHOLDER.sub(lambda m: " " * len(m.group(0)), statement)


@dataclass(frozen=True)
class Span:
    """One numeric span of a statement, and what the table made of it."""

    start: int
    end: int
    text: str
    outcome: Outcome
    # The facts the span matched: one when exact or rounded, several when
    # ambiguous, the facts whose prose holds it when quoted, none when
    # unsupported.
    fact_ids: tuple[str, ...]

    @property
    def value(self) -> str:
        return _normalised(self.text)


@dataclass(frozen=True)
class Attribution:
    """A statement's spans, and the statement rewritten as citations."""

    statement: str
    reconstructed: str
    spans: tuple[Span, ...]

    @property
    def flags(self) -> tuple[Flag, ...]:
        return tuple(_flag(span) for span in self.spans if span.outcome not in _SUPPORTED)


def _flag(span: Span) -> Flag:
    if span.outcome == "unsupported":
        return Flag(D0, f"no fact supports the value {span.value!r}")
    if span.outcome == "ambiguous":
        cited = ", ".join(span.fact_ids)
        return Flag(D0_AMBIGUOUS, f"{span.value!r} is the value of several facts: {cited}")
    assert span.outcome == "rounded", f"{span.outcome} is not a flag"
    return Flag(D0_ROUNDED, f"{span.value!r} is {span.fact_ids[0]} at a coarser rounding")


def _numeric_spans_in(text: str) -> Iterator[re.Match[str]]:
    return NUMERIC.finditer(text)


Index = Mapping[str, tuple[str, ...]]


def _index(pairs: Iterator[tuple[str, str]]) -> Index:
    found: dict[str, list[str]] = {}
    for key, fact_id in pairs:
        ids = found.setdefault(key, [])
        if fact_id not in ids:
            ids.append(fact_id)
    return {key: tuple(ids) for key, ids in found.items()}


def _values(facts: FactTable) -> tuple[Index, Index]:
    """Two indexes from a normalised value to the facts behind it.

    The first holds each fact's own rendered value: a span matching there
    cites that fact, and can be rewritten as its placeholder.

    The second holds the numbers that appear *inside* a `label`, `name` or
    `text` fact's prose ("a propensity score within 0.01 of 0 or 1"). Those
    are numbers the table holds, so a model repeating such a sentence has not
    invented the number in it -- but the fact they belong to is the sentence,
    not the number, so they are kept apart. Mixing the two would make one
    prose fact a match for every number it mentions and leave almost every
    value in the narrative ambiguous.
    """
    citable = _index((_normalised(format_fact(fact)), fact.id) for fact in facts)
    quoted = _index(
        (_normalised(match.group(0)), fact.id)
        for fact in facts
        if fact.kind not in _FLOAT_KINDS and fact.kind != "count"
        for match in _numeric_spans_in(str(fact.value))
    )
    return citable, quoted


def _parsed(match: re.Match[str]) -> tuple[float, bool, bool, int] | None:
    """`(value, is_percent, is_bounded, significant_digits)` for a span, or None.

    `is_bounded` marks a `< 0.001` form, which the renderer writes for a
    p-value under its floor and which can only ever match exactly.
    """
    digits = (match.group("int").replace(",", "") + (match.group("frac") or "")).lstrip("0")
    try:
        value = float(_normalised(match.group(0)).lstrip("< ").rstrip("%"))
    except ValueError:
        return None
    return value, match.group("pct") is not None, match.group("lt") is not None, len(digits)


def _rounds_to(fact: Fact, value: float, is_percent: bool, digits: int) -> bool:
    """Whether `fact` is this span at a rounding no finer than the renderer's.

    A `level` renders as a percentage, so its magnitude is compared at 100x.
    A level restated as a bare proportion ("0.95" for `95%`) is accepted here
    too: the quantity is right and only the unit differs, which is the same
    kind of defect as showing fewer digits.
    """
    if fact.kind not in _FLOAT_KINDS:
        return False
    assert isinstance(fact.value, float)
    if fact.kind == "level":
        # A level renders as a percentage, but the model may restate it either
        # way -- "95%", "95 percent" (a bare span) or the proportion "0.95" --
        # so both scales count. The quantity is right and only the unit or the
        # precision differs.
        magnitudes = [fact.value * 100.0, fact.value]
    elif is_percent:
        # Only a level renders with a `%`, so a percentage cannot be one of these.
        return False
    else:
        magnitudes = [fact.value]
    # Below `digits` would be a coarser rounding than the span shows, and above
    # `_MAX_DIGITS` is finer than the renderer ever writes: neither is a match.
    for magnitude in magnitudes:
        for precision in range(max(1, digits), _MAX_DIGITS + 1):
            if float(format(magnitude, f".{precision}g")) == value:
                return True
    return False


def _names_subject(statement: str, fact_id: str, facts: FactTable) -> bool:
    """Whether `statement` names the method the fact is about."""
    fact = facts.get(fact_id)
    if fact is None or fact.subject is None:
        return False
    subject = facts.get(fact.subject)
    if subject is None:
        return False
    return re.search(rf"\b{re.escape(str(subject.value))}\b", statement) is not None


def _narrowed(candidates: tuple[str, ...], statement: str, facts: FactTable) -> tuple[str, ...]:
    """`candidates` cut to the one whose subject the statement names, if there is one.

    Two facts can carry the same value: `est.ipw.point` and the comparison
    table's `point_low`, which is that same estimate restated as the range's
    lower end. A reader tells them apart by what the sentence is about, and
    so does this. The subject must pick out exactly one; otherwise the span
    stays as ambiguous as it was, because a guess here would put a citation
    the model never made in front of D1-D5.
    """
    named = tuple(i for i in candidates if _names_subject(statement, i, facts))
    return named if len(named) == 1 else candidates


def _classify(
    match: re.Match[str], statement: str, facts: FactTable, citable: Index, quoted: Index
) -> Span:
    def span(outcome: Outcome, fact_ids: tuple[str, ...]) -> Span:
        return Span(match.start(), match.end(), match.group(0), outcome, fact_ids)

    key = _normalised(match.group(0))
    exact = _narrowed(citable.get(key, ()), statement, facts)
    if exact:
        return span("exact" if len(exact) == 1 else "ambiguous", exact)
    # Before any rounding: a number the table writes out in its own prose was
    # read, not rounded. Flagging `D0-rounded` for a value shown verbatim
    # somewhere in the table would be an accusation the table itself answers.
    in_prose = quoted.get(key, ())
    if in_prose:
        return span("quoted", in_prose)
    parsed = _parsed(match)
    if parsed is not None:
        value, is_percent, is_bounded, digits = parsed
        # `< 0.001` is the renderer's floor form for a p-value under its
        # floor: it matches that form exactly or nothing at all, so there is
        # no coarser rounding of it to look for.
        if not is_bounded:
            rounded = _narrowed(
                tuple(fact.id for fact in facts if _rounds_to(fact, value, is_percent, digits)),
                statement,
                facts,
            )
            if rounded:
                return span("rounded" if len(rounded) == 1 else "ambiguous", rounded)
    return span("unsupported", ())


def _reconstructed(statement: str, spans: Sequence[Span]) -> str:
    """`statement` with every exact span rewritten as the placeholder it cites.

    Only an exact, unambiguous span is rewritten. An ambiguous one cannot be
    attributed to one fact; a rounded or unsupported one does not carry the
    table's value; a quoted one belongs to a sentence, not to a number. Each
    is left as the model wrote it -- which is what D1-D5 then see: a number
    with no citation.
    """
    out = []
    cursor = 0
    for span in spans:
        if span.outcome != "exact":
            continue
        out.append(statement[cursor : span.start])
        out.append(f"{{{span.fact_ids[0]}}}")
        cursor = span.end
    out.append(statement[cursor:])
    return "".join(out)


def attribute(statement: str, facts: FactTable) -> Attribution:
    """Every numeric span of `statement`, classified against `facts`."""
    citable, quoted = _values(facts)
    spans = tuple(
        _classify(match, statement, facts, citable, quoted)
        for match in _numeric_spans_in(_masked(statement))
    )
    return Attribution(statement, _reconstructed(statement, spans), spans)


def flags(statement: str, facts: FactTable) -> tuple[Flag, ...]:
    """The attribution flags of one statement, as a grader returns them."""
    return attribute(statement, facts).flags


def grade_reconstructed(narrative: Narrative, facts: FactTable) -> tuple[StatementGrade, ...]:
    """Grade a narrative the model wrote values into, as `grade_narrative` does.

    Each statement is matched against the table first, then D1-D5 run on the
    reconstructed form, where an attributed value carries the placeholder it
    would have had. So the graders see what they were written to read, and
    their rates on this arm mean the same thing as on a grounded one, with
    the caveat that here the citations are inferred rather than declared.

    The grade keeps the statement **as the model wrote it**: that is what a
    labeller is shown and what the report quotes. The reconstruction is an
    intermediate, and a flag naming `{est.ipw.point}` names the fact the
    matcher attributed, not a citation the model made.
    """
    grades = []
    for position, (section, statement) in enumerate(narrative.statements()):
        attribution = attribute(statement, facts)
        grades.append(
            StatementGrade(
                section,
                position,
                statement,
                attribution.flags + grade_statement(attribution.reconstructed, facts),
            )
        )
    return tuple(grades)
