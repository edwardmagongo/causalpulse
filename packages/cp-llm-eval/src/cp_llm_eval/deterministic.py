"""Misreadings a program can find, because it knows what each placeholder cites.

Each grader reads one statement as the model wrote it, with its placeholders,
and the fact table. A placeholder names its fact, so a grader knows the value
behind it without parsing a number out of prose. Each grader is registered by
name so that a test can remove one and show that its planted defect is then
not counted: no grader is covered only by another.

The graders find patterns. A misreading phrased outside a grader's pattern is
the judge's (J6), and the labelled sample measures how much each grader
misses and how often it flags something that is fine.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from cp_llm.facts import Fact, FactTable
from cp_llm.narrative import Narrative
from cp_llm.render import PLACEHOLDER

DETERMINISTIC_CATEGORIES: tuple[str, ...] = ("D1", "D2", "D3", "D4", "D5")
# Not a misreading: a comparison D1 could not attribute to two cited facts,
# counted so the report can say how many it left to the judge.
D1_UNPARSED = "D1-unparsed"

Truth = Literal["true", "false", "undetermined"]


@dataclass(frozen=True)
class Flag:
    category: str
    reason: str
    # D4 only: whether the uncited summary is true of the table.
    truth: Truth | None = None


@dataclass(frozen=True)
class StatementGrade:
    section: str
    # Position in reading order across the whole narrative, from 0.
    position: int
    statement: str
    flags: tuple[Flag, ...]


_WORD = re.compile(r"[a-z]+(?:['’-][a-z]+)*")


def _masked(statement: str) -> str:
    """The statement with each placeholder blanked to spaces of the same length.

    Positions in the masked text are positions in the statement, so a word can
    be placed before or after a placeholder without the placeholder's own
    text (`{est.ipw.point}` holds `point`) being read as prose.
    """
    return PLACEHOLDER.sub(lambda m: " " * len(m.group(0)), statement)


@dataclass(frozen=True)
class _Cited:
    start: int
    end: int
    fact: Fact


def _cited(statement: str, facts: FactTable) -> list[_Cited]:
    found = []
    for match in PLACEHOLDER.finditer(statement):
        fact = facts.get(match.group(1))
        if fact is not None:
            found.append(_Cited(match.start(), match.end(), fact))
    return found


def _words_before(masked: str, cited: list[_Cited], index: int, count: int) -> list[str]:
    """The last `count` words before cited placeholder `index`.

    The window stops at the placeholder before it, so words that belong to an
    earlier value are not read as describing this one.
    """
    start = cited[index - 1].end if index > 0 else 0
    return _WORD.findall(masked[start : cited[index].start].lower())[-count:]


# D1 ------------------------------------------------------------------------

_GREATER = frozenset({"larger", "higher", "greater", "more", "bigger", "above"})
_GREATER |= {"exceeds", "exceed", "exceeded", "exceeding"}
_LESS = frozenset({"smaller", "lower", "less", "fewer", "below"})
_COMPARATIVES = _GREATER | _LESS
_BAND_ORDER = {"INSUFFICIENT": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}
_NUMERIC_KINDS = frozenset({"number", "count", "level", "p_value"})
_BAND_FACT = re.compile(r"score\.[a-z0-9_]+\.(band|ceiling)")


def _comparable(fact: Fact) -> tuple[str, float] | None:
    """The fact's value on a scale where `<` means something, with that scale's name."""
    if fact.kind in _NUMERIC_KINDS:
        assert isinstance(fact.value, int | float)
        return "number", float(fact.value)
    if _BAND_FACT.fullmatch(fact.id) and str(fact.value) in _BAND_ORDER:
        return "band", float(_BAND_ORDER[str(fact.value)])
    return None


def _comparative(statement: str, facts: FactTable) -> Iterator[Flag]:
    """A comparative between two cited values that their values contradict.

    The comparative is attributed to the nearest comparable value before it
    and the nearest after it. "No higher" and "not above" are read as "at
    most". A comparative with a value on one side only, in a statement that
    cites two or more, is unparsed: the comparison may be with something the
    grader cannot see.
    """
    masked = _masked(statement)
    items = [(c, _comparable(c.fact)) for c in _cited(statement, facts)]
    comparables = [(c, v) for c, v in items if v is not None]
    unparsed = False
    for match in _WORD.finditer(masked.lower()):
        word = match.group(0)
        if word not in _GREATER and word not in _LESS:
            continue
        left = [(c, v) for c, v in comparables if c.end <= match.start()]
        right = [(c, v) for c, v in comparables if c.start >= match.end()]
        if not left or not right:
            unparsed = unparsed or len(comparables) >= 2
            continue
        (a, (scale_a, value_a)), (b, (scale_b, value_b)) = left[-1], right[0]
        if scale_a != scale_b:
            unparsed = True
            continue
        preceding = _WORD.findall(masked[: match.start()].lower())[-1:]
        negated = preceding in (["no"], ["not"])
        holds = value_a > value_b if word in _GREATER else value_a < value_b
        if holds == negated:
            relation = f"{'not ' if negated else ''}{word}"
            yield Flag(
                "D1",
                f"{{{a.fact.id}}} is called {relation} than {{{b.fact.id}}}, "
                f"but their values are {a.fact.value} and {b.fact.value}",
            )
    if unparsed:
        yield Flag(D1_UNPARSED, "a comparative that is not between two cited values")


# D2 ------------------------------------------------------------------------

_NEGATIONS = frozenset(
    {"not", "no", "never", "neither", "nor", "cannot"}
    | {"isn't", "wasn't", "aren't", "weren't", "doesn't", "didn't", "don't",
      "can't", "won't", "couldn't", "wouldn't", "shouldn't", "hasn't", "haven't"}
    | {"isn’t", "wasn’t", "aren’t", "weren’t", "doesn’t", "didn’t", "don’t",
      "can’t", "won’t", "couldn’t", "wouldn’t", "shouldn’t", "hasn’t", "haven’t"}
)
# A value the analysis computed. A name fact is excluded: "no estimate from
# {route.aipw.estimator}" negates nothing the table states.
_VALUE_KINDS = frozenset({"label", "number", "count", "level", "p_value"})


def _negated(statement: str, facts: FactTable) -> Iterator[Flag]:
    """A negation within three words before a cited value.

    A label whose own value begins with a negation ("not eligible") is
    exempt: the negation is the system's, not the model's. So is a negation
    followed by a comparative ("no higher than {ceiling}"): it belongs to the
    comparison, which is D1's to check.
    """
    masked = _masked(statement)
    cited = _cited(statement, facts)
    for index, item in enumerate(cited):
        if item.fact.kind not in _VALUE_KINDS:
            continue
        first = _WORD.findall(str(item.fact.value).lower())[:1]
        if first and first[0] in _NEGATIONS:
            continue
        window = _words_before(masked, cited, index, 3)
        hits = [
            word
            for position, word in enumerate(window)
            if word in _NEGATIONS
            and not (window[position + 1 : position + 2] and window[position + 1] in _COMPARATIVES)
        ]
        if hits:
            yield Flag("D2", f"{hits[-1]!r} before {{{item.fact.id}}} negates the cited value")


# D3 ------------------------------------------------------------------------

_LABELS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("p_value", re.compile(r"\bp[- ]?values?\b")),
    ("se", re.compile(r"\bstandard errors?\b")),
    ("bound", re.compile(r"\b(bounds?|interval)\b")),
    ("point", re.compile(r"\b(point|estimates?|effect)\b")),
)
_SWAPPED = re.compile(r"\bfrom\s+\{([^{}]+)\}\s+to\s+\{([^{}]+)\}")
_SWAPPED_PAIRS = ((".ci_upper", ".ci_lower"), (".point_high", ".point_low"))


def _field(fact: Fact) -> str:
    return fact.id.rsplit(".", 1)[-1]


def _mislabelled(statement: str, facts: FactTable) -> Iterator[Flag]:
    """A number called by the name of a different quantity, or a range written backwards.

    The label word nearest a number, within the four words before it, names
    what the sentence says the number is.
    """
    masked = _masked(statement)
    cited = _cited(statement, facts)
    for index, item in enumerate(cited):
        if item.fact.kind not in ("number", "p_value"):
            continue
        window = " ".join(_words_before(masked, cited, index, 4))
        nearest: tuple[int, str] | None = None
        for name, pattern in _LABELS:
            for match in pattern.finditer(window):
                if nearest is None or match.end() > nearest[0]:
                    nearest = (match.end(), name)
        if nearest is None:
            continue
        label, field = nearest[1], _field(item.fact)
        wrong = (
            (label == "p_value" and field != "p_value")
            or (label == "se" and field != "se")
            or (label == "bound" and field == "point")
            or (label == "point" and field in ("ci_lower", "ci_upper"))
        )
        if wrong:
            yield Flag("D3", f"{{{item.fact.id}}} is labelled as a {label.replace('_', '-')}")
    for match in _SWAPPED.finditer(statement):
        first, second = match.group(1), match.group(2)
        if any(first.endswith(hi) and second.endswith(lo) for hi, lo in _SWAPPED_PAIRS):
            yield Flag("D3", f"a range written from {{{first}}} down to {{{second}}}")


# D4 ------------------------------------------------------------------------

# Up to three words may stand between the quantifier and its noun ("none of
# the five robustness checks"). "Not a single" and "not one of the" read as
# "no": they are quantifiers themselves, matched from their "not", so the
# "preceded by not" rule below never sees that "not".
_QUANTIFIER = re.compile(
    r"\b(no|none of the|not a single|not one of the|every|all|all of the|all the|each)\s+"
    r"(?:[a-z-]+\s+){0,3}(checks?|diagnostics?)\b"
)
_PROBLEM = re.compile(
    r"\b(fail\w*|warn\w*|flag\w*|problems?|concerns?|issues?|violations?|alarms?)\b"
)
_UNTESTED = re.compile(
    r"\b(untest(?:ed|able)|not testable|not checkable|cannot be (?:tested|checked)"
    r"|could not be (?:tested|checked))\b"
)
_PASSED = re.compile(r"\b(pass(?:ed|es|ing)?|cleared|satisfied)\b")
_STATUS_WORD = re.compile(r"\bstatus(?:es)?\b")
# A quantified check is a status summary only when the statement says
# something about status: "each of the checks is listed below" is not one.
_STATUS_PREDICATES = (_PROBLEM, _PASSED, _UNTESTED, _STATUS_WORD)
# The quantifiers that claim no check has the property ("no check failed").
_NEGATIVE_QUANTIFIERS = frozenset({"no", "none of the", "not a single", "not one of the"})
_STATUS_FACT = re.compile(r"check\.([a-z0-9_]+)\.[a-z0-9_]+\.status")


def _summary_truth(quantifier: str, text: str, statuses: list[str]) -> Truth:
    if not statuses:
        return "undetermined"
    negative = quantifier in _NEGATIVE_QUANTIFIERS
    if negative and _PROBLEM.search(text):
        return "true" if not {"FAIL", "WARN"} & set(statuses) else "false"
    if not negative:
        if _UNTESTED.search(text):
            return "true" if set(statuses) == {"NOT_TESTABLE"} else "false"
        if _PASSED.search(text):
            return "true" if set(statuses) == {"PASS"} else "false"
    return "undetermined"


def _status_summary(statement: str, facts: FactTable) -> Iterator[Flag]:
    """A claim about all or none of the checks' statuses that cites no status.

    The statement must carry a status predicate (a problem, a pass, untested,
    or the word status); a quantified check without one says nothing a status
    could contradict.

    Its truth is read from the table: the checks of the one method the
    statement names, or every check when it names none or several.
    """
    cited = _cited(statement, facts)
    if any(_STATUS_FACT.fullmatch(c.fact.id) for c in cited):
        return
    text = _masked(statement).lower()
    match = _QUANTIFIER.search(text)
    if match is None or not any(p.search(text) for p in _STATUS_PREDICATES):
        return
    methods = {c.fact.id.split(".")[1] for c in cited if c.fact.id.startswith("est.")}
    statuses = [
        str(f.value)
        for f in facts
        if (m := _STATUS_FACT.fullmatch(f.id)) and (len(methods) != 1 or m.group(1) in methods)
    ]
    # "not every check passed" is true when any check did not: the flag
    # stands, but the table cannot call the sentence false.
    negated = _WORD.findall(text[: match.start()])[-1:] == ["not"]
    truth = "undetermined" if negated else _summary_truth(match.group(1), text, statuses)
    yield Flag("D4", f"{match.group(0)!r} summarises check statuses without citing one", truth)


# D5 ------------------------------------------------------------------------

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z{(\"'])")


def _cross_sentence(statement: str, facts: FactTable) -> Iterator[Flag]:
    """A fact whose method is named only in another sentence of the statement."""
    sentences = _SENTENCE_BREAK.split(statement)
    if len(sentences) < 2:
        return
    cited = [{c.fact.id for c in _cited(sentence, facts)} for sentence in sentences]
    everywhere = set().union(*cited)
    for ids in cited:
        for fact_id in sorted(ids):
            fact = facts.get(fact_id)
            assert fact is not None
            subject = fact.subject
            if subject is not None and subject not in ids and subject in everywhere:
                yield Flag("D5", f"{{{fact_id}}} is in a sentence that does not name {{{subject}}}")


Grader = Callable[[str, FactTable], Iterator[Flag]]

GRADERS: dict[str, Grader] = {
    "D1": _comparative,
    "D2": _negated,
    "D3": _mislabelled,
    "D4": _status_summary,
    "D5": _cross_sentence,
}


def grade_statement(statement: str, facts: FactTable) -> tuple[Flag, ...]:
    return tuple(flag for grader in GRADERS.values() for flag in grader(statement, facts))


def grade_narrative(narrative: Narrative, facts: FactTable) -> tuple[StatementGrade, ...]:
    return tuple(
        StatementGrade(section, position, statement, grade_statement(statement, facts))
        for position, (section, statement) in enumerate(narrative.statements())
    )
