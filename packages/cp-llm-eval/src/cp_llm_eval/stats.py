"""The uncertainty attached to every rate the report states.

The four replies to one fact table are not independent draws, so a rate's
standard error is clustered by case. A rate of zero has no useful standard
error (it is zero), so it carries the rule-of-three upper bound instead. A
judge's sensitivity and specificity are proportions of a small labelled
sample and carry Wilson intervals. The prompt comparison is paired by case.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Rate:
    """`numerator` of `denominator` units, over `clusters` cases.

    `se` is the cluster-robust standard error, or None when the numerator is
    zero or there is one cluster. `upper_bound` is the rule-of-three 95% upper
    bound, 3 / denominator capped at 1, given only when the numerator is zero;
    it treats the units as independent, so with clustering it is optimistic.
    """

    numerator: int
    denominator: int
    clusters: int
    rate: float
    se: float | None
    upper_bound: float | None


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
    if clusters < 2:
        return Rate(numerator, denominator, clusters, rate, None, None)
    squares = sum((n - rate * d) ** 2 for n, d in groups)
    variance = clusters / (clusters - 1) * squares / denominator**2
    return Rate(numerator, denominator, clusters, rate, math.sqrt(variance), None)


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    n: int


def wilson(successes: int, n: int, z: float = Z_95) -> Interval:
    """The Wilson score interval for `successes` of `n`."""
    if n < 1:
        raise ValueError("a proportion needs at least one trial")
    if not 0 <= successes <= n:
        raise ValueError(f"{successes} successes of {n} trials")
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), n)


@dataclass(frozen=True)
class Difference:
    """The mean of `after - before` over paired cases, with a normal 95% interval."""

    mean: float
    se: float
    lower: float
    upper: float
    pairs: int


def paired_difference(pairs: Sequence[tuple[float, float]]) -> Difference:
    """`after - before` per case, averaged; the standard error is that of the mean.

    The interval is normal rather than t: with about 25 cases the two differ
    in the second decimal of the half-width, and the report says which it is.
    """
    if len(pairs) < 2:
        raise ValueError("a paired difference needs at least two pairs")
    differences = [after - before for before, after in pairs]
    mean = sum(differences) / len(differences)
    variance = sum((d - mean) ** 2 for d in differences) / (len(differences) - 1)
    se = math.sqrt(variance / len(differences))
    return Difference(mean, se, mean - Z_95 * se, mean + Z_95 * se, len(differences))
