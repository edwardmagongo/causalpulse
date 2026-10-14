"""The event study: per-horizon coefficients around a structural reference.

The hazard this module exists to close off: the
reference period is normalised to zero *by construction*, not estimated.
`cp_causal.estimators.event_study.EventStudy.fit` initialises the coefficient
and standard-error arrays with `np.zeros` and never writes to the reference
column (`REFERENCE_HORIZON`, `t - 1`) -- so that horizon carries no
estimation uncertainty to report, because none was estimated. Drawing a
confidence band around it would invite a reader to test a hypothesis that
cannot be false. `event_study_data` flags which horizon that is, and
`draw_event_study` marks it with a plain point at zero and no error bar,
while every estimated horizon gets one.

This module deliberately never reads the *bounds* of `estimate.ci`: that
interval bounds the single averaged post-treatment effect, not any
individual event-time coefficient, and calling `require_interval` here
would be answering the wrong question. It does read `estimate.ci.level`,
though -- the confidence level the fit was actually run at, not the bounds
computed from it -- because that level is what every per-horizon band in
this figure is built at (see `event_study_data`), and `EstimatorConfig.ci_level`
is configurable rather than fixed at 0.95. Reading the level carries none of
the hazard reading the bounds would: it is a single scalar the whole fit
shares, not a value specific to the post-treatment average this module must
not conflate with a per-horizon coefficient.

That read still needs `estimate.ci` to exist, though: `EffectEstimate.ci` is
typed `Interval`, not `Interval | None`, and nothing in
`EffectEstimate.__post_init__` checks it (`cp_figures.guards`'s module
docstring), so a hand-built or `dataclasses.replace`-mutated estimate can
carry `ci=None` despite the type. `event_study_data` reads the level through
`cp_figures.guards.require_ci_level` rather than `estimate.ci.level`
directly, so that case is refused with a `ValueError` naming this figure
rather than raising an unhelpful `AttributeError` for want of `.level` on
`None`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cp_core.results import EffectEstimate
from matplotlib.axes import Axes
from scipy.stats import norm

from cp_figures.guards import require_ci_level


def _critical_value(level: float) -> float:
    """The two-sided normal critical value at `level`.

    Computed the same way `cp_causal._support.critical_value` and
    `cp_statistics.proportions` compute it: `norm.ppf(1 - (1 - level) / 2)`.
    Reused as the same formula rather than a textbook-rounded literal (e.g.
    1.96 for 95%), so an interval drawn here uses the same convention as
    every other interval in the project, at whatever level the fit actually
    used.
    """
    return float(norm.ppf(1 - (1 - level) / 2))


def _format_level(level: float) -> str:
    """Render `level` as a percentage without ever rounding it up to 100%.

    `f"{level:.0%}"` rounds to the nearest whole percent, so a level of
    0.999 -- reachable, since `EstimatorConfig.ci_level` is validated only
    as `0.0 < ci_level < 1.0` -- renders as "100%", claiming a certainty the
    fit was never actually run at. `:g`'s default precision (six
    significant digits) keeps whatever digits `level * 100` actually has,
    so it prints "95%" for the common case and "99.9%" for 0.999 rather
    than rounding either to 100. It is not an absolute guarantee: a level
    needing more than six significant digits, such as 0.9999999, still
    renders as "100%". That is a far-fetched `ci_level` and the six-digit
    bound is stated here rather than defended against.
    """
    return f"{level * 100:g}%"


@dataclass(frozen=True, eq=False)
class EventStudyData:
    """Everything an event-study figure draws, one entry per event-time horizon.

    `eq=False` for the same reason as `CounterfactualPathData` in
    `cp_figures.effects`: every array field would otherwise generate an
    `__eq__` that raises on the ambiguous truth value of an array comparison.

    `is_reference` flags the one horizon normalised to zero by construction
    (see the module docstring) so a drawing function never has to recompute
    which one it was; `reference_horizon` names it directly.

    `level` is the confidence level `lower`/`upper` were built at --
    `estimate.ci.level`, not a hardcoded constant (see the module docstring)
    -- so a fit run at a non-default level (`EstimatorConfig.ci_level`) is
    drawn at that level rather than silently widened or narrowed to 95%.
    `draw_event_study` names it in the legend so a reader is never shown an
    unlabelled band.
    """

    horizons: np.ndarray
    coefficients: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    is_reference: tuple[bool, ...]
    reference_horizon: float
    level: float


def event_study_data(estimate: EffectEstimate) -> EventStudyData:
    """Read the three event-time diagnostics and locate the reference horizon.

    Reads `event_time_horizons`, `event_time_coefficients` and
    `event_time_standard_errors` by name and lets a `KeyError` propagate when
    any is absent: an estimate without them is not an event study, and this
    function does not guess at a substitute.

    Builds a normal interval from each horizon's own standard error, at
    `estimate.ci.level` -- reading the level, never the bounds, of
    `estimate.ci`, which bounds the single averaged post-treatment effect
    rather than any individual horizon (see the module docstring). A fit run
    at a non-default `EstimatorConfig.ci_level` is therefore drawn at that
    same level here, rather than silently widened or narrowed to 95%.

    The level is read through `guards.require_ci_level` rather than
    `estimate.ci.level` directly: `estimate.ci` can be `None` on a
    hand-built or `dataclasses.replace`-mutated estimate despite its
    non-optional type (see the module docstring and `cp_figures.guards`),
    and without a level there is no critical value to build any horizon's
    margin from -- every horizon would draw as a bare point. That refuses
    with a `ValueError` naming this figure and the method, rather than
    raising an unhelpful `AttributeError` for `None.level`.

    The reference horizon is identified empirically, not assumed at a fixed
    index or offset: it is the one horizon whose coefficient *and* standard
    error are both exactly zero. Verified against a real `EventStudy.fit`
    (panel scenario P3, `n_units=150, n_periods=7`): the arrays it produces
    initialise the reference column with `np.zeros` and never write to it, so
    the pair is an exact `(0.0, 0.0)` there and nowhere else -- comparing
    with `==` rather than a tolerance is correct here, not a shortcut.

    Raises `ValueError` if that pair does not identify exactly one horizon.
    Zero matches means this estimate was not produced by an estimator that
    pins a reference this way; more than one match is ambiguous about which
    horizon is structural (a genuinely null pre-period coefficient can land
    on exactly zero, but landing its standard error on exactly zero as well
    only happens for a column that was never estimated). Either way, this
    function refuses to guess rather than silently picking one.
    """
    horizons = np.asarray(estimate.diagnostics["event_time_horizons"].value, dtype=float)
    coefficients = np.asarray(
        estimate.diagnostics["event_time_coefficients"].value, dtype=float
    )
    standard_errors = np.asarray(
        estimate.diagnostics["event_time_standard_errors"].value, dtype=float
    )

    reference_mask = (coefficients == 0.0) & (standard_errors == 0.0)
    matches = int(reference_mask.sum())
    if matches != 1:
        raise ValueError(
            "event study must have exactly one reference horizon (coefficient "
            f"and standard error both exactly zero), found {matches}"
        )

    level = require_ci_level(estimate, "event study")
    margin = _critical_value(level) * standard_errors
    return EventStudyData(
        horizons=horizons,
        coefficients=coefficients,
        lower=coefficients - margin,
        upper=coefficients + margin,
        is_reference=tuple(bool(flag) for flag in reference_mask),
        reference_horizon=float(horizons[reference_mask][0]),
        level=level,
    )


def draw_event_study(data: EventStudyData, ax: Axes) -> Axes:
    """Draw one point per horizon against event time, on `ax`.

    Estimated horizons get a point and an error bar built from their
    interval. The reference horizon gets a plain filled point at zero and no
    error bar, drawn with its own `errorbar` call so its bar-line collection
    is empty rather than merely omitted -- it is normalised to zero by
    construction, and a band around it would invite a reader to test a
    hypothesis that cannot be false (see the module docstring). A horizontal
    line at zero marks the no-effect baseline.

    The estimated-horizons legend entry names `data.level` (e.g. "estimated
    (90% CI)") rather than just "estimated": a band with no stated level
    leaves a reader unable to say what was actually rendered, and
    `EstimatorConfig.ci_level` is not always the project's 95% default. The
    level is rendered by `_format_level`, not `f"{level:.0%}"` directly --
    `.0%`'s round-to-nearest-whole-percent would print "100%" for a
    reachable level like 0.999, claiming a certainty the fit never actually
    ran at.
    """
    horizons = np.asarray(data.horizons)
    coefficients = np.asarray(data.coefficients)
    lower = np.asarray(data.lower)
    upper = np.asarray(data.upper)
    is_reference = np.asarray(data.is_reference, dtype=bool)
    estimated = ~is_reference

    lower_err = coefficients[estimated] - lower[estimated]
    upper_err = upper[estimated] - coefficients[estimated]
    ax.errorbar(
        horizons[estimated],
        coefficients[estimated],
        yerr=(lower_err, upper_err),
        fmt="o",
        capsize=3,
        color="tab:blue",
        label=f"estimated ({_format_level(data.level)} CI)",
    )
    ax.errorbar(
        [data.reference_horizon],
        [0.0],
        yerr=None,
        fmt="o",
        color="black",
        markersize=8,
        label="reference (fixed at zero by construction)",
    )

    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("event time (periods since adoption)")
    ax.set_ylabel("coefficient")
    ax.legend()
    return ax
