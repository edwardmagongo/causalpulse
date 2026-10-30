"""Matching the numbers in a narrative back to the facts behind them.

The grounded arms need no such thing: a placeholder names its fact, so a
grader knows the value behind it without reading prose. The ungrounded arm
writes values itself, and every grader in `deterministic` reads placeholders,
so without this module they all see an empty statement and score it perfect.

So each numeric span is matched against the table and the statement is
rewritten with the placeholder it would have carried, which lets D1-D5 run on
this arm unchanged. A span that matches nothing is the arm's own finding.

Four outcomes, because they are four different things:

- **exact** -- the span is a fact's rendered form. Nothing to report.
- **rounded** -- the span is a correct rounding of exactly one fact, at the
  model's own precision rather than the renderer's. The quantity is right and
  the digits shown are the model's choice. Reported as `D0_ROUNDED`, and
  deliberately **not** as `D0`: it is not a value the table fails to support,
  and folding it into the primary endpoint would inflate that endpoint with a
  milder defect.
- **ambiguous** -- two or more facts carry the span's value, so which one it
  cites cannot be recovered. Not a misreading; counted so a reader knows how
  far D1-D5 can be read on this arm.
- **unsupported** -- no fact supports the value at any rounding. This is
  `D0`, the primary endpoint of the control arm.

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
from cp_llm.render import PLACEHOLDER, format_fact

from cp_llm_eval.deterministic import Flag

# A value no fact supports: the control arm's primary endpoint.
D0 = "D0"
# Not misreadings. Counted so the report can say what the arm's D1-D5 rest on.
D0_AMBIGUOUS = "D0-ambiguous"
D0_ROUNDED = "D0-rounded"

Outcome = Literal["exact", "rounded", "ambiguous", "unsupported"]

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
    # ambiguous, none when unsupported.
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
        return tuple(_flag(span) for span in self.spans if span.outcome != "exact")


def _flag(span: Span) -> Flag:
    if span.outcome == "unsupported":
        return Flag(D0, f"no fact supports the value {span.value!r}")
    if span.outcome == "ambiguous":
        cited = ", ".join(span.fact_ids)
        return Flag(D0_AMBIGUOUS, f"{span.value!r} is the value of several facts: {cited}")
    return Flag(D0_ROUNDED, f"{span.value!r} is {span.fact_ids[0]} at a coarser rounding")


def _numeric_spans_in(text: str) -> Iterator[re.Match[str]]:
    return NUMERIC.finditer(text)


def _rendered_values(facts: FactTable) -> Mapping[str, tuple[str, ...]]:
    """Normalised rendered value -> the ids of every fact that shows it.

    A `label`, `name` or `text` fact holds prose, which may itself contain a
    number ("the threshold was 0.05"). Those numbers are values the table
    supports, so each one is indexed to that fact as well; otherwise a model
    repeating a sentence out of the table would be charged with inventing the
    number inside it.
    """
    found: dict[str, list[str]] = {}
    for fact in facts:
        keys = [_normalised(format_fact(fact))]
        if fact.kind not in _FLOAT_KINDS and fact.kind != "count":
            keys += [_normalised(m.group(0)) for m in _numeric_spans_in(str(fact.value))]
        for key in keys:
            ids = found.setdefault(key, [])
            if fact.id not in ids:
                ids.append(fact.id)
    return {key: tuple(ids) for key, ids in found.items()}


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


def _classify(
    match: re.Match[str], facts: FactTable, rendered: Mapping[str, tuple[str, ...]]
) -> Span:
    key = _normalised(match.group(0))
    exact = rendered.get(key, ())
    if len(exact) == 1:
        return Span(match.start(), match.end(), match.group(0), "exact", exact)
    if len(exact) > 1:
        return Span(match.start(), match.end(), match.group(0), "ambiguous", exact)
    parsed = _parsed(match)
    if parsed is None:
        return Span(match.start(), match.end(), match.group(0), "unsupported", ())
    value, is_percent, is_bounded, digits = parsed
    if is_bounded:
        # `< 0.001` is the renderer's floor form; it matches that form or nothing.
        return Span(match.start(), match.end(), match.group(0), "unsupported", ())
    rounded = tuple(
        fact.id for fact in facts if _rounds_to(fact, value, is_percent, digits)
    )
    if len(rounded) == 1:
        return Span(match.start(), match.end(), match.group(0), "rounded", rounded)
    if len(rounded) > 1:
        return Span(match.start(), match.end(), match.group(0), "ambiguous", rounded)
    return Span(match.start(), match.end(), match.group(0), "unsupported", ())


def _reconstructed(statement: str, spans: Sequence[Span]) -> str:
    """`statement` with every exact span rewritten as the placeholder it cites.

    Only an exact, unambiguous span is rewritten. An ambiguous one cannot be
    attributed to one fact, and a rounded or unsupported one does not carry
    the table's value, so each is left as the model wrote it -- which is what
    D1-D5 then see: a number with no citation.
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
    rendered = _rendered_values(facts)
    spans = tuple(
        _classify(match, facts, rendered) for match in _numeric_spans_in(_masked(statement))
    )
    return Attribution(statement, _reconstructed(statement, spans), spans)


def flags(statement: str, facts: FactTable) -> tuple[Flag, ...]:
    """The attribution flags of one statement, as a grader returns them."""
    return attribute(statement, facts).flags
