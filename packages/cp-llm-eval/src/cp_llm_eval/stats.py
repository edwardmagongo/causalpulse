"""The uncertainty attached to every rate the report states.

The four replies to one fact table are not independent draws, so a rate's
standard error is clustered by case. A rate of zero has no useful standard
error (it is zero), so it carries the rule-of-three upper bound instead, and
a unanimous rate carries the mirrored lower bound. A
judge's sensitivity and specificity are proportions of a small labelled,
stratified sample, weighted back to the population, and carry Wilson
intervals at the effective sample size. The prompt comparison is paired by case.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Z_95 = 1.959963984540054
_SAME_DIFFERENCE = 1e-12


@dataclass(frozen=True)
class Rate:
    """`numerator` of `denominator` units, over `clusters` cases.

    `se` is the cluster-robust standard error, or None when it would say
    nothing: the numerator is zero or equals the denominator, there is one
    cluster, or every cluster has the same proportion (the CR1 form is then
    exactly 0, which would read as certainty). `upper_bound` is the
    rule-of-three 95% upper bound, 3 / denominator capped at 1, given only
    when the numerator is zero; `lower_bound` is its mirror, 1 - 3 /
    denominator floored at 0, given only when the numerator equals the
    denominator. Both treat the units as independent, so with clustering they
    are optimistic.
    """

    numerator: int
    denominator: int
    clusters: int
    rate: float
    se: float | None
    upper_bound: float | None
    lower_bound: float | None = None


def clustered_rate(groups: Sequence[tuple[int, int]]) -> Rate:
    """The rate over clusters given as `(numerator, denominator)` pairs.

    The standard error is the CR1 ratio-estimator form: residuals
    `n_g - rate * d_g` summed in squares, scaled by `G / (G - 1)`, over the
    squared total denominator.
    """
    groups = [(n, d) for n, d in groups if d > 0]
    numerator = sum(n for n, _ in groups)
    denominator = sum(d for _, d in groups)
    if denominator == 0:
        raise ValueError("a rate needs at least one unit")
    rate = numerator / denominator
    clusters = len(groups)
    if numerator == 0:
        return Rate(numerator, denominator, clusters, 0.0, None, min(1.0, 3 / denominator))
    if numerator == denominator:
        lower = max(0.0, 1 - 3 / denominator)
        return Rate(numerator, denominator, clusters, 1.0, None, None, lower)
    # Every residual n - rate * d is zero exactly when n * D == N * d, checked
    # in integers so float rounding cannot hide it.
    if clusters < 2 or all(n * denominator == numerator * d for n, d in groups):
        return Rate(numerator, denominator, clusters, rate, None, None)
    squares = sum((n - rate * d) ** 2 for n, d in groups)
    variance = clusters / (clusters - 1) * squares / denominator**2
    return Rate(numerator, denominator, clusters, rate, math.sqrt(variance), None)


@dataclass(frozen=True)
class Interval:
    """`n` is the number of trials, or the effective sample size of a weighted one."""

    estimate: float
    lower: float
    upper: float
    n: float


def wilson(successes: int, n: int, z: float = Z_95) -> Interval:
    """The Wilson score interval for `successes` of `n`."""
    if n < 1:
        raise ValueError("a proportion needs at least one trial")
    if not 0 <= successes <= n:
        raise ValueError(f"{successes} successes of {n} trials")
    return wilson_weighted(successes / n, n, z)


def wilson_weighted(p: float, n_eff: float, z: float = Z_95) -> Interval:
    """The Wilson score interval for a weighted proportion `p`, at effective size `n_eff`.

    `n_eff` is Kish's `(sum w)^2 / sum w^2`; at equal weights it is the count,
    and this is `wilson`.
    """
    if not n_eff > 0:
        raise ValueError(f"an effective sample size must be positive, got {n_eff}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"a proportion must lie in [0, 1], got {p}")
    centre = (p + z**2 / (2 * n_eff)) / (1 + z**2 / n_eff)
    half = z * math.sqrt(p * (1 - p) / n_eff + z**2 / (4 * n_eff**2)) / (1 + z**2 / n_eff)
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), n_eff)


@dataclass(frozen=True)
class Difference:
    """The mean of `after - before` over paired cases, with a normal 95% interval.

    `degenerate` is true when every case has the same difference: `se` is then
    0 and the interval is a point, which says no spread, not certainty.
    """

    mean: float
    se: float
    lower: float
    upper: float
    pairs: int
    degenerate: bool = False


def paired_difference(pairs: Sequence[tuple[float, float]]) -> Difference:
    """`after - before` per case, averaged; the standard error is that of the mean.

    The interval is normal rather than t: with about 25 cases the two differ
    in the second decimal of the half-width, and the report says which it is.
    """
    if len(pairs) < 2:
        raise ValueError("a paired difference needs at least two pairs")
    differences = [after - before for before, after in pairs]
    mean = sum(differences) / len(differences)
    # Equal differences can come out of float subtraction a rounding apart
    # (0.1 - 0.0 and 0.3 - 0.2); distinct shares of a few replies differ by far
    # more than this tolerance.
    if max(differences) - min(differences) <= _SAME_DIFFERENCE:
        return Difference(mean, 0.0, mean, mean, len(differences), degenerate=True)
    variance = sum((d - mean) ** 2 for d in differences) / (len(differences) - 1)
    se = math.sqrt(variance / len(differences))
    return Difference(mean, se, mean - Z_95 * se, mean + Z_95 * se, len(differences))
