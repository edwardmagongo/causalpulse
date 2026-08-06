"""Interval estimates for proportions and differences of proportions.

Wald intervals are deliberately not offered. They undercover badly at the small
counts and extreme proportions that conversion-rate experiments routinely
produce, and there is no case where they are the right default.
"""

from __future__ import annotations

import math

from cp_core.values import Interval
from scipy.stats import norm


def _validate_counts(successes: int, trials: int) -> None:
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(
            f"successes must lie in [0, trials]; got successes={successes}, trials={trials}"
        )


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a single proportion.

    The score interval inverts the score test rather than assuming normality of
    the estimate, so it stays inside [0, 1] and retains near-nominal coverage at
    proportions close to 0 or 1.
    """
    _validate_counts(successes, trials)

    z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
    p_hat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p_hat + z * z / (2.0 * trials)) / denominator
    half_width = (z / denominator) * math.sqrt(
        p_hat * (1.0 - p_hat) / trials + z * z / (4.0 * trials * trials)
    )
    lower = max(0.0, centre - half_width)
    upper = min(1.0, centre + half_width)

    # Handle floating point precision at boundaries
    if successes == 0:
        lower = 0.0
    if successes == trials:
        upper = 1.0

    return Interval(lower=lower, upper=upper, level=level)


def newcombe_interval(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    level: float = 0.95,
) -> Interval:
    """Newcombe hybrid-score interval for the difference `p_a - p_b`.

    Method 10 of Newcombe (1998). Combines the separate Wilson intervals rather
    than assuming normality of the difference, which keeps coverage honest when
    either arm is near a boundary.
    """
    _validate_counts(successes_a, trials_a)
    _validate_counts(successes_b, trials_b)

    p_a = successes_a / trials_a
    p_b = successes_b / trials_b
    arm_a = wilson_interval(successes_a, trials_a, level)
    arm_b = wilson_interval(successes_b, trials_b, level)

    difference = p_a - p_b
    lower = difference - math.sqrt(
        (p_a - arm_a.lower) ** 2 + (arm_b.upper - p_b) ** 2
    )
    upper = difference + math.sqrt(
        (arm_a.upper - p_a) ** 2 + (p_b - arm_b.lower) ** 2
    )
    return Interval(lower=max(-1.0, lower), upper=min(1.0, upper), level=level)
