"""Benchmark figures over `cp_evaluation.metrics.summarise`: the heatmap and the profile.

The heatmap draws one cell per (method, scenario) at a single grid slice.
The axis profile draws one line per method along one axis of the grid, for
one scenario -- the figure `heatmap_data`'s duplicate refusal tells a caller
to build rather than widening the heatmap into.

The hazard both exist to close off (Phase 16 design, section 6): a
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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from cp_evaluation.grid import AXIS_BASELINES
from cp_evaluation.metrics import GROUP_COLUMNS
from matplotlib import colormaps
from matplotlib.axes import Axes


def _differing_identity_columns(group: pd.DataFrame, exclude: Sequence[str]) -> list[str]:
    """Which of the grid's own cell-identity columns differ across `group`'s rows.

    Only `cp_evaluation.metrics.GROUP_COLUMNS` -- `shape`, `n_units`,
    `n_periods`, and the axis names in `cp_simulation.params.AXIS_NAMES` --
    never one of the dozens of metric columns `summarise` also produces
    (`bias`, `rmse_mcse`, `realised_treated_share`, ...). Two rows can
    legitimately differ in `bias` while describing the very same grid cell,
    so a metric column is not identity and is never something a caller could
    "filter to one value of".

    Shared by both refusals in this module -- the heatmap's
    one-row-per-(method, scenario) check and the profile's
    one-row-per-(method, axis value) check -- so the two cannot drift apart
    about what counts as identity. `exclude` is what the caller has already
    keyed on and so cannot differ by construction; each caller still writes
    its own message, keeping the phrase a reader greps for at the site that
    raises it (the arrangement `cp_figures.guards` uses for its three guards).
    """
    return [
        column
        for column in GROUP_COLUMNS
        if column not in exclude
        and column in group.columns
        and group[column].nunique(dropna=False) > 1
    ]


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
        differing_identity_columns = _differing_identity_columns(
            group, exclude=("method", "scenario")
        )
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


# The axes a profile can be drawn along: every axis the grid varies, plus
# `n_units`. `n_units` is crossed with `confounding_strength` rather than
# varied one at a time, but with every `AXIS_BASELINES` axis pinned to its
# baseline the surviving rows are a genuine one-at-a-time profile, and
# sample-size scaling is among the most useful curves a benchmark produces.
PROFILE_AXES: tuple[str, ...] = (*AXIS_BASELINES, "n_units")


@dataclass(frozen=True, eq=False)
class AxisProfileData:
    """Everything an axis profile draws: one metric per (method, axis value).

    `eq=False` for `HeatmapData`'s reason: `values` and `absent` are
    ndarrays, and a generated `__eq__` would raise on the ambiguous truth
    value of an array comparison rather than compare.

    `absent[i, j]` is `True` exactly when the (`methods[i]`,
    `axis_values[j]`) cell had zero fitted rows in the source summary --
    including a combination missing from the frame entirely -- and never
    merely because `values[i, j]` happens to be NaN. That is the same
    guarantee `HeatmapData` carries, and it matters more here, not less: a
    line chart has no "bad colour" to fall back on, so absence has to be
    kept off the value scale by construction (see `draw_axis_profile`).
    `relative_bias` is legitimately NaN on a null-truth cell with
    `n_fitted > 0`, and such a cell is present-but-unplottable, not missing.

    `axis_values` is `float` even for `n_units`, whose grid values are
    `int`: the field holds positions on one numeric axis, and one element
    type keeps the array arithmetic and the tick formatting uniform.

    `realised_treated_share[j]` is the mean realised treated share at that
    axis value, or `None` when the summary carries no such column. See
    `axis_profile_data` for why it is recorded and why a mean is exact here.
    """

    axis: str
    scenario: str
    metric: str
    axis_values: tuple[float, ...]
    methods: tuple[str, ...]
    values: np.ndarray
    absent: np.ndarray
    realised_treated_share: tuple[float, ...] | None


def axis_profile_data(
    summary: pd.DataFrame, metric: str, *, axis: str, scenario: str
) -> AxisProfileData:
    """Select one scenario's one-at-a-time slice along `axis`, per method.

    `axis` and `scenario` are keyword-only. Three bare strings in a row is a
    silent-swap hazard, and this package already takes that precaution --
    `cp_figures.sensitivity.placebo_data` makes `statistic` keyword-only
    with no default "so a caller cannot forget to say which one it is".

    **The slice.** `cp_evaluation.grid._expand_cross_section` varies each
    axis alone, against a fixed anchor. Recovering that slice from a summary
    means keeping the rows where every *other* axis sits at its baseline
    (`cp_evaluation.grid.AXIS_BASELINES`) and `n_units` sits at the grid's
    anchor, `max(axes.n_units)`.

    The `n_units` pin is not a detail. The grid emits the baseline cell at
    *every* `n_units` (it comes from the `n_units` x `confounding_strength`
    cross-product) but anchors the perturbed cells at the largest one.
    Without the pin, the axis's own default value would carry two rows per
    method and the two would describe cells differing in two things at once.
    The anchor is computed over the rows that survive the axis pins rather
    than over the whole summary, so the perturbed cells -- which exist only
    at the anchor -- cannot drag it.

    That pinning also recovers the **baseline point**, which is easy to miss:
    `_expand_cross_section` explicitly skips the axis's own default in its
    one-at-a-time loop (`if value == baseline_value: continue`), so the
    baseline comes from the cross-product instead. A rule that gathered only
    the perturbed cells would silently drop the point the whole comparison
    is anchored on.

    `n_units` itself is selectable, and is the rule's one branch: it pins all
    five `AXIS_BASELINES` axes and has no `n_units` pin of its own.

    **Absence.** A cell is absent when its `n_fitted` is zero, or when the
    (method, axis value) combination has no row at all -- never when
    `metric`'s value happens to be NaN. See `AxisProfileData` and this
    module's docstring for why those are different conditions.

    **The realised treated share** travels alongside, per axis value, because
    section 4.1 of `docs/specs/2026-09-13-causal-core-phase18-design.md` says
    the `treatment_intercept` axis "sets the intercept, not a target share",
    and that the realised share is recorded beside it "because an intercept
    is not a quantity a reader of the eventual research report thinks in". A
    profile labelled only `-1.5, 0.0, 1.5` reproduces exactly the gap Phase
    18's review found in `bench report`. Taking the mean across the rows at
    one axis value is exact rather than an aggregation hiding disagreement:
    `summarise` computes this figure over *every* row in a (method, cell)
    group -- fitted, ineligible and refused alike, since it is a property of
    the cell's generated data rather than of any fit -- so every method
    sharing a cell already carries the identical value.

    `methods` and `axis_values` are both sorted, giving an order that does
    not depend on the row order `summary` happens to arrive in.
    """
    if metric not in summary.columns:
        raise KeyError(metric)

    rows = summary
    if "shape" in rows.columns:
        # Panel groups carry NaN for every axis -- `PanelAxes` has none of
        # them -- so they would survive no pin anyway; dropping them here
        # makes the reason explicit rather than incidental.
        rows = rows[rows["shape"] == "cross_section"]

    rows = rows[rows["scenario"] == scenario]

    for name, baseline in AXIS_BASELINES.items():
        if name != axis and name in rows.columns:
            rows = rows[rows[name] == baseline]

    if axis != "n_units" and "n_units" in rows.columns and not rows.empty:
        rows = rows[rows["n_units"] == rows["n_units"].max()]

    axis_values = tuple(sorted(float(value) for value in rows[axis].unique()))
    methods = tuple(sorted(str(method) for method in rows["method"].unique()))
    value_index = {value: j for j, value in enumerate(axis_values)}
    method_index = {method: i for i, method in enumerate(methods)}

    values = np.full((len(methods), len(axis_values)), np.nan, dtype=float)
    # Absence defaults to True everywhere and is only cleared below, for a
    # (method, axis value) that has a row *and* a nonzero n_fitted -- so a
    # combination with no row at all is left exactly as absent as one whose
    # row reports n_fitted == 0.
    absent = np.ones((len(methods), len(axis_values)), dtype=bool)

    for _, row in rows.iterrows():
        i = method_index[str(row["method"])]
        j = value_index[float(row[axis])]
        values[i, j] = float(row[metric])
        absent[i, j] = int(row["n_fitted"]) == 0

    return AxisProfileData(
        axis=axis,
        scenario=scenario,
        metric=metric,
        axis_values=axis_values,
        methods=methods,
        values=values,
        absent=absent,
        realised_treated_share=None,
    )
