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

`require_ci_level` closes a narrower version of the same gap for
`cp_figures.panel.event_study_data`, which never reads `estimate.ci`'s
*bounds* (see that module's own docstring for why) but does read its
*level*, to build every per-horizon interval at the same confidence level
the fit was actually run at. `estimate.ci` being `None` is exactly as fatal
to that read as it is to `require_interval`'s -- with no level to build a
margin from, every horizon would draw as a bare point -- so it is refused
the same way, distinctly from `require_interval` so a missing level is
never described as a missing interval when only the level was ever needed.

All three guards live in this one module rather than in each figure so that
a figure added later cannot quietly omit them.
"""

from __future__ import annotations

from cp_core.results import EffectEstimate, SpecificationPoint
from cp_core.values import Interval

# The substantive half of the refusal -- why missing uncertainty matters,
# not merely that it is missing -- stated once so the three guards cannot
# drift apart. Each guard still writes its own reason in its own source
# ("without an interval", "without a confidence level"), so the phrase a
# reader greps for stays visible there rather than only in this helper.
_MISSING_INTERVAL_EXPLANATION = (
    "a point drawn alone states a number with no indication of how well it "
    "is known"
)


def _refuse_to_draw(prefix: str, reason: str) -> ValueError:
    """Build the `ValueError` every guard raises, from a caller-specific reason."""
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


def require_ci_level(estimate: EffectEstimate, figure: str) -> float:
    """Return the estimate's confidence level, or refuse to let it be drawn.

    Distinct from `require_interval`: a caller here reads only the *level*
    `estimate.ci` was computed at, never its bounds (see the module
    docstring). But building any interval at all -- even one this function's
    caller assembles itself from other diagnostics -- still needs a level to
    compute a critical value from, and `estimate.ci` is the only place one is
    recorded. A missing `estimate.ci` is therefore exactly as fatal here as
    it is to `require_interval`, so it is refused the same way.
    """
    interval = estimate.ci
    if interval is None:
        # "the {figure}" rather than "a {figure}" (require_interval's form):
        # this guard's one caller passes "event study", and "a event study"
        # is bad grammar -- the definite article reads correctly regardless
        # of what figure name a future caller passes.
        raise _refuse_to_draw(
            f"cannot draw the {figure} for {estimate.method!r}", "without a confidence level"
        )
    return interval.level
