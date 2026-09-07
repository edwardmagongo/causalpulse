"""The forest plot and the counterfactual path.

Both figures compare an estimator's output against its own uncertainty, and
both carry a hazard this package exists to close off:

- The forest plot draws rows from estimators that do not share an estimand --
  cross-sectional methods report `ATE`, panel methods report `ATT`. Rendering
  them on one axis without saying so per row would imply a single quantity
  where there are two, which is the defect family this project has hit three
  times.
- The counterfactual path draws a shaded band that is the uncertainty in the
  *predicted-absent-treatment* series, not in the effect itself. Labelling it
  as the effect's interval would misstate what the band shows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from cp_core.results import CounterfactualPath, EffectEstimate
from matplotlib.axes import Axes

from cp_figures.guards import require_interval


@dataclass(frozen=True)
class ForestPlotData:
    """Everything a forest plot draws, one tuple entry per estimate.

    `estimands` is carried alongside `methods` rather than assumed constant,
    because the estimators being compared do not all target the same
    estimand -- see the module docstring.
    """

    methods: tuple[str, ...]
    points: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    estimands: tuple[str, ...]
    bases: tuple[str, ...]


def forest_plot_data(estimates: Sequence[EffectEstimate]) -> ForestPlotData:
    """Build one forest-plot row per estimate.

    Calls `require_interval` for every estimate before anything is drawn, so
    an estimate without an interval refuses here rather than being silently
    plotted as a bare point.
    """
    methods: list[str] = []
    points: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    estimands: list[str] = []
    bases: list[str] = []

    for estimate in estimates:
        interval = require_interval(estimate, "forest plot")
        methods.append(estimate.method)
        points.append(estimate.point)
        lower.append(interval.lower)
        upper.append(interval.upper)
        estimands.append(estimate.estimand)
        bases.append(estimate.ci_basis)

    return ForestPlotData(
        methods=tuple(methods),
        points=tuple(points),
        lower=tuple(lower),
        upper=tuple(upper),
        estimands=tuple(estimands),
        bases=tuple(bases),
    )


def draw_forest_plot(data: ForestPlotData, ax: Axes) -> Axes:
    """Draw one row per estimate: a point and its interval, on `ax`.

    Each row's y-tick label combines the method with its estimand --
    `f"{method} ({estimand})"` -- because these estimators do not share an
    estimand: cross-sectional methods report `ATE`, panel methods report
    `ATT`. Without the per-row label a reader would take the whole column as
    one comparable quantity, which it is not.
    """
    y = np.arange(len(data.methods))
    points = np.asarray(data.points)
    lower_err = points - np.asarray(data.lower)
    upper_err = np.asarray(data.upper) - points

    ax.errorbar(
        points,
        y,
        xerr=(lower_err, upper_err),
        fmt="o",
        capsize=3,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(
        [
            f"{method} ({estimand})"
            for method, estimand in zip(data.methods, data.estimands, strict=True)
        ]
    )
    ax.axvline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("effect estimate")
    return ax


@dataclass(frozen=True, eq=False)
class CounterfactualPathData:
    """Everything a counterfactual-path figure draws.

    `lower`/`upper` bound the *counterfactual* series (`predicted`), not the
    effect -- see the module docstring.

    `eq=False` for the same reason as `CounterfactualPath` in `cp_core.results`:
    every field is an ndarray, and a generated `__eq__` would raise on the
    ambiguous truth value of an array comparison rather than compare.
    """

    periods: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def counterfactual_path_data(path: CounterfactualPath) -> CounterfactualPathData:
    """Pull the observed series, the counterfactual series, and its band out of a fit result."""
    return CounterfactualPathData(
        periods=path.periods,
        observed=path.observed,
        predicted=path.predicted_absent_treatment,
        lower=path.lower,
        upper=path.upper,
    )


def draw_counterfactual_path(data: CounterfactualPathData, ax: Axes) -> Axes:
    """Draw the observed series against its counterfactual, with the counterfactual's band.

    The filled region is labelled as the *counterfactual's* interval, never
    as the effect's: the band bounds `predicted_absent_treatment`, the series
    that says what would have happened without treatment, not the treatment
    effect itself. Calling it the effect's interval would misstate what it
    shows.
    """
    ax.plot(data.periods, data.observed, label="observed")
    ax.plot(data.periods, data.predicted, label="counterfactual (predicted absent treatment)")
    ax.fill_between(
        data.periods,
        data.lower,
        data.upper,
        alpha=0.2,
        label="counterfactual uncertainty band",
    )
    ax.set_xlabel("period")
    ax.set_ylabel("outcome")
    ax.legend()
    return ax
