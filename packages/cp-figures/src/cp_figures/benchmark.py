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
from cp_evaluation.metrics import GROUP_COLUMNS
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

    Also refuses, by raising `ValueError`, when `summary` carries more than
    one row for the same (method, scenario) pair. Phase 18's grid can widen
    `n_units`, `confounding_strength` or any of the four new axes, so a
    `summarise()` output that has not been filtered to one grid slice has
    several rows per (method, scenario) -- e.g. one per `effect_scale` value.
    Without this check, the loop below silently overwrites `values[i, j]`
    and `absent[i, j]` on every row it visits, so the pivot quietly keeps
    only the last row it happened to see and reports whatever that row's
    `n_fitted` was for the whole (method, scenario) cell -- exactly the
    absent-versus-zero confusion this module's docstring exists to prevent
    (see `docs/specs/2026-09-13-causal-core-phase18-design.md` section 12,
    "Findings from the whole-branch review": a measured example collapsed
    three scenario-C cells, n_fitted 10/10/0, into one cell marked absent).
    A per-axis heatmap is a separate, not-yet-built feature; the caller must
    filter to one slice (e.g. one `n_units`/`confounding_strength`/axis
    value) before pivoting.

    The refusal names only the *identity* columns that differ across the
    duplicate rows -- `cp_evaluation.metrics.GROUP_COLUMNS` (`shape`,
    `n_units`, `n_periods`, and the axis names in
    `cp_simulation.params.AXIS_NAMES`), not the dozens of metric
    columns `summarise` also produces (`bias`, `rmse_mcse`,
    `realised_treated_share`, ...), which are never what a caller needs to
    filter on and are not identity in the first place: two rows can differ
    in `bias` while describing the very same grid cell. When none of those
    identity columns differ either -- the duplicate rows share the same
    cell identity outright -- the message says so instead of reporting an
    empty `differing in []`.

    A cell is marked absent in the returned `HeatmapData.absent` when its
    `n_fitted` is zero, or when the (method, scenario) combination has no row
    in `summary` at all -- never when `metric`'s value happens to be NaN (see
    the module docstring and the hazard it exists to close off). `methods`
    and `scenarios` are each sorted, giving a deterministic axis order that
    does not depend on the row order `summary` happens to arrive in.
    """
    if metric not in summary.columns:
        raise KeyError(metric)

    group_sizes = summary.groupby(["method", "scenario"]).size()
    duplicated = group_sizes[group_sizes > 1]
    if not duplicated.empty:
        method, scenario = duplicated.index[0]
        group = summary[(summary["method"] == method) & (summary["scenario"] == scenario)]
        # Only the grid's own cell-identity columns -- never a metric column
        # like `bias` or `realised_treated_share`, which can legitimately
        # differ between rows that describe the very same cell.
        differing_identity_columns = [
            column
            for column in GROUP_COLUMNS
            if column not in ("method", "scenario")
            and column in group.columns
            and group[column].nunique(dropna=False) > 1
        ]
        if differing_identity_columns:
            raise ValueError(
                f"heatmap_data needs exactly one row per (method, scenario), but "
                f"({method!r}, {scenario!r}) has {len(group)} rows, differing in "
                f"{differing_identity_columns}. Filter the summary to one grid "
                "slice (e.g. one value of each of those columns) before "
                "pivoting to a heatmap."
            )
        raise ValueError(
            f"heatmap_data needs exactly one row per (method, scenario), but "
            f"({method!r}, {scenario!r}) has {len(group)} rows that share the "
            "same grid-cell identity (no identity column differs between "
            "them) -- deduplicate the summary before pivoting to a heatmap."
        )

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
