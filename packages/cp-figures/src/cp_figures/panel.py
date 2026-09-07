"""The event study: per-horizon coefficients around a structural reference.

The hazard this module exists to close off (Phase 16 design, section 6): the
reference period is normalised to zero *by construction*, not estimated.
`cp_causal.estimators.event_study.EventStudy.fit` initialises the coefficient
and standard-error arrays with `np.zeros` and never writes to the reference
column (`REFERENCE_HORIZON`, `t - 1`) -- so that horizon carries no
estimation uncertainty to report, because none was estimated. Drawing a
confidence band around it would invite a reader to test a hypothesis that
cannot be false. `event_study_data` flags which horizon that is, and
`draw_event_study` marks it with a plain point at zero and no error bar,
while every estimated horizon gets one.

This module deliberately never reads `estimate.ci`: that interval bounds the
single averaged post-treatment effect, not any individual event-time
coefficient, and calling `require_interval` here would be answering the
wrong question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cp_core.results import EffectEstimate
from matplotlib.axes import Axes
from scipy.stats import norm

# The two-sided normal critical value at this project's default 95% level,
# computed the same way `cp_causal._support.critical_value` and
# `cp_statistics.proportions` compute it: `norm.ppf(1 - (1 - level) / 2)`.
# Reused as the same formula rather than the textbook-rounded 1.96, so an
# interval drawn here uses the same convention as every other interval in
# the project.
_CI_LEVEL = 0.95
_CRITICAL_VALUE = float(norm.ppf(1 - (1 - _CI_LEVEL) / 2))


@dataclass(frozen=True, eq=False)
class EventStudyData:
    """Everything an event-study figure draws, one entry per event-time horizon.

    `eq=False` for the same reason as `CounterfactualPathData` in
    `cp_figures.effects`: every array field would otherwise generate an
    `__eq__` that raises on the ambiguous truth value of an array comparison.

    `is_reference` flags the one horizon normalised to zero by construction
    (see the module docstring) so a drawing function never has to recompute
    which one it was; `reference_horizon` names it directly.
    """

    horizons: np.ndarray
    coefficients: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    is_reference: tuple[bool, ...]
    reference_horizon: float


def event_study_data(estimate: EffectEstimate) -> EventStudyData:
    """Read the three event-time diagnostics and locate the reference horizon.

    Reads `event_time_horizons`, `event_time_coefficients` and
    `event_time_standard_errors` by name and lets a `KeyError` propagate when
    any is absent: an estimate without them is not an event study, and this
    function does not guess at a substitute.

    Builds a 95% normal interval from each horizon's own standard error --
    not from `estimate.ci`, which bounds the single averaged post-treatment
    effect rather than any individual horizon (see the module docstring).

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

    margin = _CRITICAL_VALUE * standard_errors
    return EventStudyData(
        horizons=horizons,
        coefficients=coefficients,
        lower=coefficients - margin,
        upper=coefficients + margin,
        is_reference=tuple(bool(flag) for flag in reference_mask),
        reference_horizon=float(horizons[reference_mask][0]),
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
        label="estimated",
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
