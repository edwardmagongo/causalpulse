"""Benchmark heatmaps: one cell per (method, scenario), read from `cp_evaluation.metrics.summarise`.

The hazard this module exists to close off (Phase 16 design, section 6): a
cell with no fitted rows is not a cell with a value of zero.
`cp_evaluation.metrics.summarise` carries `n_fitted`, `n_ineligible` and
`n_refused` alongside every metric for exactly this reason -- its own module
docstring puts it directly: "a metric computed over zero usable rows has to
be distinguishable from one that happens to equal zero". A benchmark heatmap
that painted "this method could not run here" the same colour as "this
method scored zero here" would be actively misleading, which is why Phase 14
recorded three statuses per row rather than dropping the ones that produced
no estimate.

`heatmap_data` therefore marks a cell absent when its `n_fitted` is zero (or
when the (method, scenario) combination has no row in the summary at all),
never when the metric's value happens to be NaN. Those are genuinely
different conditions: a metric can be legitimately NaN with rows present.
`accuracy_metrics` in `cp_evaluation.metrics` returns `relative_bias` as NaN
whenever the group's mean truth is exactly zero -- the null scenarios the
false-positive test exists for -- specifically to avoid a `bias / 0` that
would read as infinite rather than undefined; those rows still have
`n_fitted > 0` and must be drawn as present, unplottable-numbered cells, not
hidden as if the method never ran there. `draw_heatmap` renders over a
masked array (masked by `absent`, not by `numpy.isnan`) so an absent cell
takes the colormap's "bad" colour instead of landing on the value scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.axes import Axes


@dataclass(frozen=True, eq=False)
class HeatmapData:
    """Everything a benchmark heatmap draws: one cell per (method, scenario).

    `eq=False` for the same reason as `CounterfactualPathData` in
    `cp_figures.effects`: `values` and `absent` are ndarrays, and a generated
    `__eq__` would raise on the ambiguous truth value of an array comparison
    rather than compare.

    `absent[i, j]` is `True` exactly when the (`methods[i]`, `scenarios[j]`)
    cell had zero fitted rows in the source summary -- including a
    combination that is missing from the frame entirely -- and never merely
    because `values[i, j]` happens to be NaN (see the module docstring).
    `values[i, j]` is NaN on an absent cell only incidentally, from there
    being nothing to read; it is `absent`, not `numpy.isnan(values)`, that a
    drawing function must consult to decide whether a cell is plottable.
    """

    methods: tuple[str, ...]
    scenarios: tuple[str, ...]
    values: np.ndarray
    absent: np.ndarray
    metric: str


def heatmap_data(summary: pd.DataFrame, metric: str) -> HeatmapData:
    """Pivot a summary frame to one cell per (method, scenario) for `metric`.

    Refuses by raising `KeyError` (naming `metric`) when the summary carries
    no such column, rather than silently drawing an empty heatmap -- the same
    refuse-rather-than-guess posture as `love_plot_data`'s length check in
    `cp_figures.balance`.

    A cell is marked absent in the returned `HeatmapData.absent` when its
    `n_fitted` is zero, or when the (method, scenario) combination has no row
    in `summary` at all -- never when `metric`'s value happens to be NaN (see
    the module docstring and the hazard it exists to close off). `methods`
    and `scenarios` are each sorted, giving a deterministic axis order that
    does not depend on the row order `summary` happens to arrive in.
    """
    if metric not in summary.columns:
        raise KeyError(metric)

    methods = tuple(sorted(str(m) for m in summary["method"].unique()))
    scenarios = tuple(sorted(str(s) for s in summary["scenario"].unique()))
    method_index = {method: i for i, method in enumerate(methods)}
    scenario_index = {scenario: j for j, scenario in enumerate(scenarios)}

    values = np.full((len(methods), len(scenarios)), np.nan, dtype=float)
    # Absence defaults to True for every cell and is only cleared below, for
    # a (method, scenario) pair that has a row *and* a nonzero n_fitted -- a
    # combination with no row at all is left exactly as absent as one with a
    # row reporting n_fitted == 0.
    absent = np.ones((len(methods), len(scenarios)), dtype=bool)

    for _, row in summary.iterrows():
        i = method_index[str(row["method"])]
        j = scenario_index[str(row["scenario"])]
        values[i, j] = float(row[metric])
        absent[i, j] = int(row["n_fitted"]) == 0

    return HeatmapData(
        methods=methods,
        scenarios=scenarios,
        values=values,
        absent=absent,
        metric=metric,
    )


def draw_heatmap(data: HeatmapData, ax: Axes) -> Axes:
    """Draw one cell per (method, scenario) as a colour, on `ax`.

    Renders over a masked array (`numpy.ma.masked_array`, masked by
    `data.absent`) rather than the bare `values` array, so `ax.imshow` paints
    an absent cell in the colormap's "bad" colour instead of placing it on
    the value scale -- the whole point of tracking absence separately from
    NaN (see the module docstring). `with_extremes` returns a new colormap
    rather than mutating the shared, globally registered "viridis" instance
    that other figures elsewhere might also read (`Colormap.set_bad` does
    the same mutation and is pending deprecation for exactly that reason).
    """
    masked_values = np.ma.masked_array(data.values, mask=data.absent)
    cmap = colormaps["viridis"].with_extremes(bad="lightgrey")

    ax.imshow(masked_values, cmap=cmap)
    ax.set_xticks(list(range(len(data.scenarios))))
    ax.set_xticklabels(list(data.scenarios))
    ax.set_yticks(list(range(len(data.methods))))
    ax.set_yticklabels(list(data.methods))
    ax.set_xlabel("scenario")
    ax.set_ylabel("method")
    ax.set_title(data.metric)
    return ax
