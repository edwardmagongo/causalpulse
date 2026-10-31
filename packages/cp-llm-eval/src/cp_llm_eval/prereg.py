"""The control arm's pre-registered endpoints, scored against a finished run.

Written down before the arm was billed, so a result cannot be fitted to them
afterwards. `P0` gates `P1`: it runs the attribution matcher's own output over
the grounded narratives, where an unsupported value cannot exist, and a
non-zero count there means the matcher is measuring itself rather than the
model. `P1` is the primary endpoint, scored against a threshold fixed in
advance.

`P1` is one-sample by design. The grounded arms' unsupported-value rate is
zero as a property of verifier rules 1 and 2, not an estimate of anything, so
pairing against it would attach a standard error to a logical fact. `P0`
establishes that property holds of the stored narratives; `P1` measures the
arm that has no verifier.

The flags scored here are produced by the attribution matcher, which must
emit exactly `D0` for a value the table does not support and `D0_AMBIGUOUS`
for one that matches two or more facts, and `D0_ROUNDED` for a value that is
some fact at a coarser rounding. Only `D0` is the primary endpoint's event:
`attribution`'s own docstring says why a rounded value is reported apart from
it. This module reads flag categories off each statement, so it does not
require any of them to have been added to
`deterministic.DETERMINISTIC_CATEGORIES`.

`graded.grade_records` is what puts those flags there, and it runs the
matcher on every arm: over the model's own text on the ungrounded arm, and
over the rendered text on a grounded one -- which is the text P0 is a claim
about, since that is where every value is some fact's rendered form exactly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from cp_llm_eval.attribution import D0, D0_AMBIGUOUS, D0_ROUNDED
from cp_llm_eval.graded import GradedNarrative, GradedStatement
from cp_llm_eval.stats import Z_95, Rate, clustered_rate, wilson

__all__ = [
    "D0",
    "D0_AMBIGUOUS",
    "D0_ROUNDED",
    "AMBIGUITY_CONCERN",
    "P1_THRESHOLD",
    "UNGROUNDED_ARM",
    "Bound",
    "P0Result",
    "P1Result",
    "has_flag",
    "prereg_markdown",
    "score_p0",
    "score_p1",
]

# The arm with no verifier. Its narratives are the P1 population.
UNGROUNDED_ARM = "ungrounded"

# P1's pre-registered threshold, and the direction predicted: the per-narrative
# unsupported-value rate in the ungrounded arm is greater than this.
P1_THRESHOLD = 0.10

# Above this, the ambiguity rate is reported as a limit on the arm's D1-D5
# figures rather than as a footnote. Not a pass condition.
AMBIGUITY_CONCERN = 0.10

Verdict = Literal["met", "failed", "inconclusive"]
IntervalBasis = Literal["clustered", "rule_of_three", "wilson_unclustered"]


@dataclass(frozen=True)
class Bound:
    """The interval a verdict was read from, and which form produced it.

    `clustered` is the pre-registered form: the CR1 standard error over cases.
    `rule_of_three` is the bound `clustered_rate` gives a rate of zero or one,
    which has no useful standard error. `wilson_unclustered` is the fallback
    when every case has the same proportion: the CR1 form is then exactly 0,
    which would read as certainty, so the interval is taken at the unclustered
    count instead and is optimistic, because it ignores the clustering.
    """

    lower: float
    upper: float
    basis: IntervalBasis


@dataclass(frozen=True)
class P0Result:
    """The instrument check. `passed` is false if any grounded narrative has a D0."""

    passed: bool
    unsupported: int
    statements: int
    narratives: int
    ambiguity: Rate | None
    note: str = ""


@dataclass(frozen=True)
class P1Result:
    """The primary endpoint: the per-narrative unsupported-value rate, ungrounded arm."""

    verdict: Verdict
    threshold: float
    rate: Rate | None
    bound: Bound | None
    excluded_no_statements: int
    note: str = ""


def has_flag(statement: GradedStatement, category: str) -> bool:
    """Whether a deterministic flag of `category` is on this statement.

    Only the deterministic side is read. The matcher is a program, and
    `judge.Finding` is typed to the judge's own categories, so neither `D0`
    nor `D0_AMBIGUOUS` can ever arrive on a verdict.
    """
    return any(flag.category == category for flag in statement.deterministic)


def _readable(narratives: Iterable[GradedNarrative]) -> list[GradedNarrative]:
    """Narratives the model actually wrote statements for.

    A narrative with no statements -- the model gave no answer, or was cut off
    -- stated no value at all. Counting it in the denominator would move the
    rate toward zero for a reason that has nothing to do with grounding, so it
    is excluded and counted separately.
    """
    return [n for n in narratives if n.statements]


def _per_narrative_groups(
    narratives: Sequence[GradedNarrative], category: str
) -> list[tuple[int, int]]:
    """`(flagged, total)` per case: a narrative counts once however many statements carry it."""
    cases: dict[str, list[int]] = {}
    for narrative in narratives:
        flagged = any(has_flag(s, category) for s in narrative.statements)
        cases.setdefault(narrative.case_id, []).append(int(flagged))
    return [(sum(flags), len(flags)) for flags in cases.values()]


def _per_statement_groups(
    narratives: Sequence[GradedNarrative], category: str
) -> list[tuple[int, int]]:
    cases: dict[str, list[int]] = {}
    for narrative in narratives:
        for statement in narrative.statements:
            cases.setdefault(narrative.case_id, []).append(int(has_flag(statement, category)))
    return [(sum(flags), len(flags)) for flags in cases.values()]


def score_p0(narratives: Sequence[GradedNarrative]) -> P0Result:
    """Score the instrument check over every grounded narrative given.

    Every value in a verified narrative came from `render.substitute` and so
    is some fact's rendered form exactly. Both `D0` and `D0_ROUNDED` are
    therefore impossible here, and either one is a defect in the matcher
    rather than a finding about the model.

    `D0_AMBIGUOUS` is not a defect: two facts can genuinely carry the same
    value, and the renderer then writes the same characters for both. Its rate
    is measured instead, because it bounds how far the ungrounded arm's D1-D5
    figures can be read.
    """
    grounded = _readable(n for n in narratives if n.variant != UNGROUNDED_ARM)
    statements = [s for n in grounded for s in n.statements]
    unsupported = sum(has_flag(s, D0) or has_flag(s, D0_ROUNDED) for s in statements)
    ambiguity = (
        clustered_rate(_per_statement_groups(grounded, D0_AMBIGUOUS)) if statements else None
    )
    if not grounded:
        note = "no grounded narrative with statements: the check could not run"
    elif unsupported:
        note = (
            f"{unsupported} of {len(statements)} grounded statements carry {D0} or "
            f"{D0_ROUNDED}, which the verifier makes impossible; the matcher is at fault "
            "and P1 is void until it is fixed"
        )
    elif ambiguity is not None and ambiguity.rate > AMBIGUITY_CONCERN:
        note = (
            f"ambiguity is {ambiguity.rate:.1%} of grounded statements, above "
            f"{AMBIGUITY_CONCERN:.0%}: report it as a limit on the ungrounded arm's D1-D5 figures"
        )
    else:
        note = ""
    return P0Result(
        passed=bool(grounded) and unsupported == 0,
        unsupported=unsupported,
        statements=len(statements),
        narratives=len(grounded),
        ambiguity=ambiguity,
        note=note,
    )


def _bound(rate: Rate) -> Bound:
    """The 95% interval a verdict is read from, by the rules fixed in advance."""
    if rate.se is not None:
        return Bound(
            max(0.0, rate.rate - Z_95 * rate.se),
            min(1.0, rate.rate + Z_95 * rate.se),
            "clustered",
        )
    if rate.upper_bound is not None:
        return Bound(0.0, rate.upper_bound, "rule_of_three")
    if rate.lower_bound is not None:
        return Bound(rate.lower_bound, 1.0, "rule_of_three")
    interval = wilson(rate.numerator, rate.denominator)
    return Bound(interval.lower, interval.upper, "wilson_unclustered")


def score_p1(
    narratives: Sequence[GradedNarrative], threshold: float = P1_THRESHOLD
) -> P1Result:
    """Score the primary endpoint over the ungrounded arm.

    Met when the interval's lower bound clears `threshold`, failed when its
    upper bound falls below it, inconclusive when it spans it. The verdict is
    read from the interval, never from the point rate, so a rate above the
    threshold with an interval straddling it is inconclusive and says so.
    """
    arm = [n for n in narratives if n.variant == UNGROUNDED_ARM]
    readable = _readable(arm)
    excluded = len(arm) - len(readable)
    if not readable:
        return P1Result(
            "inconclusive", threshold, None, None, excluded,
            f"no {UNGROUNDED_ARM} narrative with statements",
        )
    rate = clustered_rate(_per_narrative_groups(readable, D0))
    bound = _bound(rate)
    if bound.lower > threshold:
        verdict: Verdict = "met"
    elif bound.upper < threshold:
        verdict = "failed"
    else:
        verdict = "inconclusive"
    notes = []
    if bound.basis == "wilson_unclustered":
        notes.append(
            "every case had the same proportion, so the clustered standard error is exactly "
            "0; the interval is Wilson at the unclustered count and ignores the clustering"
        )
    if excluded:
        notes.append(f"{excluded} narrative(s) had no statements and are not in the denominator")
    return P1Result(verdict, threshold, rate, bound, excluded, "; ".join(notes))


def prereg_markdown(p0: P0Result, p1: P1Result) -> str:
    """The two endpoints as the report prints them, P0 first because it gates P1."""
    lines = [
        "# Pre-registered endpoints",
        "",
        f"P0 {'passed' if p0.passed else 'FAILED'} · P1 {p1.verdict}.",
        "",
        "## P0 - the attribution matcher finds no unsupported value in a grounded narrative",
        "",
        f"{p0.unsupported} of {p0.statements} statements over {p0.narratives} grounded "
        "narratives. Passes only at exactly 0.",
        "",
    ]
    if p0.ambiguity is not None:
        lines += [f"Ambiguous values: {p0.ambiguity.rate:.1%} of grounded statements.", ""]
    if p0.note:
        lines += [f"Note: {p0.note}", ""]
    lines += [
        f"## P1 - the ungrounded arm's per-narrative unsupported-value rate exceeds "
        f"{p1.threshold:.0%}",
        "",
    ]
    if not p0.passed:
        lines += ["P0 did not pass, so this endpoint is void.", ""]
    if p1.rate is None or p1.bound is None:
        lines += [f"Not scored: {p1.note or 'no data'}", ""]
        return "\n".join(lines)
    lines += [
        f"{p1.rate.numerator} of {p1.rate.denominator} narratives over {p1.rate.clusters} "
        f"cases: {p1.rate.rate:.1%} [{p1.bound.lower:.1%}, {p1.bound.upper:.1%}], "
        f"{p1.bound.basis.replace('_', ' ')}.",
        "",
        f"Verdict: **{p1.verdict}** against a threshold of {p1.threshold:.0%}, fixed before "
        "the arm was run.",
        "",
    ]
    if p1.note:
        lines += [f"Note: {p1.note}", ""]
    return "\n".join(lines)
