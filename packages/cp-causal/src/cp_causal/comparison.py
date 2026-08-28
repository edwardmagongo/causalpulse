"""Cross-method comparison.

Estimates are not averaged. The methods share data and are not independent, so
a pooled estimate would be a number with no defensible interpretation and a
meta-analytic heterogeneity test would be invalid. What this layer reports
instead is the spread, which intervals overlap, and -- the actually useful
output when methods disagree -- which assumptions each of them relies on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from cp_core.results import (
    BAND_ORDER,
    AssumptionKind,
    AssumptionStatus,
    CausalValidityScore,
    EffectEstimate,
)

Agreement = Literal["consistent", "divergent", "not_assessable"]

# A fit the rubric has called INSUFFICIENT must not drive the verdict: its
# number is not usable, so letting it make a set look divergent would report a
# disagreement between one estimate and one non-estimate.
_MINIMUM_USABLE_BAND = BAND_ORDER["WEAK"]


@dataclass(frozen=True)
class AssumptionDivergence:
    """One assumption the compared methods do not treat identically.

    `by_method` maps method name to status, where `None` means that method does
    not rely on the assumption at all. One shape covers both "they disagree
    about an assumption" and "one needs an assumption the other never makes",
    and the second is usually the more interesting case.
    """

    check_name: str
    kind: AssumptionKind
    by_method: Mapping[str, AssumptionStatus | None]


@dataclass(frozen=True, eq=False)
class MethodComparison:
    """What several methods said about the same question, without pooling them."""

    estimates: tuple[EffectEstimate, ...]
    scores: tuple[CausalValidityScore, ...]
    point_range: tuple[float, float]
    overlap: np.ndarray
    dispersion: float
    agreement: Agreement
    assumption_diff: tuple[AssumptionDivergence, ...]
    verdict: str


def _validate(
    estimates: tuple[EffectEstimate, ...], scores: tuple[CausalValidityScore, ...]
) -> None:
    if len(estimates) < 2:
        raise ValueError(f"comparison needs at least two estimates, got {len(estimates)}")
    if len(estimates) != len(scores):
        raise ValueError(
            f"got {len(estimates)} estimates but {len(scores)} scores; each estimate "
            "needs its own validity score"
        )
    names = [e.method for e in estimates]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate method names in comparison: {sorted(names)}")
    estimands = {e.estimand for e in estimates}
    if len(estimands) > 1:
        raise ValueError(
            f"cannot compare a mixed estimand: {sorted(estimands)}. An ATE and an ATT "
            "are different quantities, not two answers to one question."
        )
    scales = {e.scale for e in estimates}
    if len(scales) > 1:
        raise ValueError(
            f"cannot compare a mixed scale: {sorted(scales)}. A percentage-point "
            "effect and a log ratio are not on the same axis."
        )


def _overlap_matrix(estimates: tuple[EffectEstimate, ...]) -> np.ndarray:
    # Two closed intervals overlap iff each one's lower bound does not exceed
    # the other's upper bound; touching endpoints count as overlap. Checked by
    # hand against a touching pair and a disjoint pair before trusting it (see
    # test_overlap_matrix_values_reflect_pairwise_interval_overlap).
    lower = np.array([e.ci.lower for e in estimates])
    upper = np.array([e.ci.upper for e in estimates])
    return (lower[:, None] <= upper[None, :]) & (lower[None, :] <= upper[:, None])


def _assumption_diff(
    estimates: tuple[EffectEstimate, ...],
) -> tuple[AssumptionDivergence, ...]:
    kinds: dict[str, AssumptionKind] = {}
    by_check: dict[str, dict[str, AssumptionStatus | None]] = {}
    for estimate in estimates:
        for assumption in estimate.assumptions:
            kinds.setdefault(assumption.name, assumption.kind)
            by_check.setdefault(assumption.name, {})[estimate.method] = assumption.status

    divergences = []
    for name in sorted(by_check):
        statuses: dict[str, AssumptionStatus | None] = {
            estimate.method: by_check[name].get(estimate.method) for estimate in estimates
        }
        if len(set(statuses.values())) > 1:
            divergences.append(
                AssumptionDivergence(check_name=name, kind=kinds[name], by_method=statuses)
            )
    return tuple(divergences)


def _verdict(
    estimates: tuple[EffectEstimate, ...],
    scores: tuple[CausalValidityScore, ...],
    usable: tuple[int, ...],
    agreement: Agreement,
    point_range: tuple[float, float],
    dispersion: float,
    n_divergences: int,
) -> str:
    bases = ", ".join(sorted({e.ci_basis for e in estimates}))
    excluded = [estimates[i].method for i in range(len(estimates)) if i not in usable]
    note = (
        f" Excluded as INSUFFICIENT and not counted toward agreement: "
        f"{', '.join(sorted(excluded))}."
        if excluded
        else ""
    )
    if agreement == "not_assessable":
        return (
            "Fewer than two methods scored WEAK or better, so agreement cannot be "
            f"assessed.{note}"
        )
    spread = (
        f"Point estimates range from {point_range[0]:.4g} to {point_range[1]:.4g}; "
        f"descriptive dispersion (standard deviation across methods, not a test of "
        f"heterogeneity) is {dispersion:.4g}. Interval bases in this comparison: {bases}."
    )
    if agreement == "consistent":
        best = max(usable, key=lambda i: BAND_ORDER[scores[i].band])
        return (
            f"The {len(usable)} methods scored WEAK or better report intervals that "
            "overlap pairwise, which is consistent with a common effect under each "
            f"method's own assumptions. Highest validity band: {estimates[best].method} "
            f"at {scores[best].band}. {spread}{note}"
        )
    divergence_note = (
        f"{n_divergences} assumption(s) differ across methods; read assumption_diff "
        "rather than treating the spread as uncertainty. "
        if n_divergences > 0
        else "No assumption checks differ across these methods, so the disagreement's "
        "source is not visible in assumption_diff. "
    )
    return (
        f"The {len(usable)} methods scored WEAK or better do not all overlap, so they "
        "are not mutually consistent under their own assumptions. "
        f"{divergence_note}"
        f"{spread}{note}"
    )


def compare(
    estimates: tuple[EffectEstimate, ...], scores: tuple[CausalValidityScore, ...]
) -> MethodComparison:
    """Compare several fitted estimates of the same quantity without pooling them."""
    _validate(estimates, scores)

    # point_range and dispersion are computed over every estimate handed in,
    # not just the usable ones. `_validate` guarantees at least two estimates,
    # so `np.std(..., ddof=1)` always sees at least two samples; restricting
    # this to the usable subset would divide by zero whenever exactly one
    # method survives the WEAK-or-better filter -- exactly the
    # not_assessable case below -- and raise under this project's
    # filterwarnings=error.
    points = np.array([e.point for e in estimates])
    overlap = _overlap_matrix(estimates)
    usable = tuple(
        i for i, score in enumerate(scores) if BAND_ORDER[score.band] >= _MINIMUM_USABLE_BAND
    )

    if len(usable) < 2:
        agreement: Agreement = "not_assessable"
    else:
        block = overlap[np.ix_(usable, usable)]
        agreement = "consistent" if bool(block.all()) else "divergent"

    point_range = (float(points.min()), float(points.max()))
    dispersion = float(np.std(points, ddof=1))
    diff = _assumption_diff(estimates)

    return MethodComparison(
        estimates=estimates,
        scores=scores,
        point_range=point_range,
        overlap=overlap,
        dispersion=dispersion,
        agreement=agreement,
        assumption_diff=diff,
        verdict=_verdict(
            estimates, scores, usable, agreement, point_range, dispersion, len(diff)
        ),
    )
