"""The one rule every figure in this package obeys.

The parent design document states it directly: a figure that displays an
effect estimate must render its uncertainty, and a plotting helper handed an
estimate without an interval raises rather than drawing a bare point. A point
without an interval is a number with no indication of how well it is known,
and a reader cannot tell the difference by looking.

`EffectEstimate.ci` is typed `Interval`, not `Interval | None` -- callers are
expected to always have one. But `dataclasses` does not enforce field types
at runtime, only `__post_init__` does, and `EffectEstimate.__post_init__`
does not check `ci`. So a hand-built or `dataclasses.replace`-mutated
`EffectEstimate` can carry `ci=None` despite what the type says, and this is
exactly the case this guard exists to catch before a figure draws from it.

`SpecificationPoint.ci` (`cp_core.results`) carries the identical hazard: it
too is typed `Interval` rather than `Interval | None`, and
`SpecificationCurve.__post_init__` validates `specifications` and
`original_index` but never inspects any specification's `ci`. So
`require_specification_interval` closes the same gap for
`cp_figures.sensitivity.specification_curve_data`, which reads
`specifications[i].ci` for every row.

Both guards live in this one module rather than in each figure so that a
figure added later cannot quietly omit them.
"""

from __future__ import annotations

from cp_core.results import EffectEstimate, SpecificationPoint
from cp_core.values import Interval

# The substantive half of both guards' refusal -- why a missing interval
# matters, not merely that one is missing -- stated once so the two guards
# cannot drift apart. Each guard still writes "without an interval" itself
# (see `_refuse_to_draw`), so the phrase stays visible in each guard's own
# source, not only inside this shared helper.
_MISSING_INTERVAL_EXPLANATION = (
    "a point drawn alone states a number with no indication of how well it "
    "is known"
)


def _refuse_to_draw(prefix: str, reason: str) -> ValueError:
    """Build the `ValueError` both guards raise, from a caller-specific prefix."""
    return ValueError(f"{prefix} {reason}: {_MISSING_INTERVAL_EXPLANATION}")


def require_interval(estimate: EffectEstimate, figure: str) -> Interval:
    """Return the estimate's interval, or refuse to let it be drawn."""
    interval = estimate.ci
    if interval is None:
        raise _refuse_to_draw(
            f"cannot draw a {figure} for {estimate.method!r}", "without an interval"
        )
    return interval


def require_specification_interval(
    point: SpecificationPoint, index: int, figure: str
) -> Interval:
    """Return one specification's interval, or refuse to let it be drawn."""
    interval = point.ci
    if interval is None:
        raise _refuse_to_draw(
            f"cannot draw a {figure} for specification {index} "
            f"({point.specification!r})",
            "without an interval",
        )
    return interval
