"""Words a narrative may use only through a placeholder.

A band, a status, an agreement verdict, a direction and a significance claim
are each a finding. Written as free text, any of them is the model's own
assertion; written as a placeholder, it is the system's. So the words that
state them are refused outside a placeholder, and the model reaches for the
fact instead.

These lists are a starting point, not a measurement. The design calls for
them to be settled against what the model actually writes; the record of that
calibration belongs beside them.
"""

from __future__ import annotations

from collections.abc import Mapping

NUMBER_WORDS: frozenset[str] = frozenset(
    {
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "half",
        "twice",
        "double",
        "minus",
        "times",
        "percent",
        "percentage",
    }
)

_BAND = "states a validity band; cite the band or ceiling fact"
_STATUS = "states a check status; cite the status fact"
_AGREEMENT = "states an agreement verdict; cite the agreement fact"
_DIRECTION = "states a direction; cite the direction fact"
_SIGNIFICANCE = "claims significance, which no fact supplies; cite interval_excludes_zero"
_ZERO = "states a zero or null result; cite the direction or interval_excludes_zero fact"
_ELIGIBILITY = "states eligibility; cite the eligible fact"

CONTROLLED_WORDS: Mapping[str, str] = {
    **dict.fromkeys(
        ("strong", "moderate", "weak", "insufficient", "strongly", "weakly", "moderately"),
        _BAND,
    ),
    **dict.fromkeys(
        (
            "pass",
            "passes",
            "passed",
            "passing",
            "warn",
            "warns",
            "warned",
            "fail",
            "fails",
            "failed",
            "failing",
            "testable",
            "untestable",
        ),
        _STATUS,
    ),
    **dict.fromkeys(
        (
            "consistent",
            "inconsistent",
            "assessable",
            "divergent",
            "diverge",
            "diverges",
            "diverged",
            "agree",
            "agrees",
            "agreed",
            "disagree",
            "disagrees",
            "disagreed",
        ),
        _AGREEMENT,
    ),
    **dict.fromkeys(
        (
            "positive",
            "negative",
            "increase",
            "increases",
            "increased",
            "decrease",
            "decreases",
            "decreased",
            "increasing",
            "decreasing",
        ),
        _DIRECTION,
    ),
    **dict.fromkeys(
        ("significant", "significantly", "significance", "insignificant"), _SIGNIFICANCE
    ),
    **dict.fromkeys(("eligible", "ineligible"), _ELIGIBILITY),
    "zero": _ZERO,
}

# "consistent with" is one of the language lint's own hedges, and refusing it
# would take away a phrasing the lint asks for.
ALLOWED_PHRASES: tuple[str, ...] = ("consistent with",)
