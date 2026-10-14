"""Benchmark figures over `cp_evaluation.metrics.summarise`: the heatmap and the profile.

The heatmap draws one cell per (method, scenario) at a single grid slice.
The axis profile draws one line per method along one axis of the grid, for
one scenario -- the figure `heatmap_data`'s duplicate refusal tells a caller
to build rather than widening the heatmap into.

The hazard both exist to close off: a
cell with no fitted rows is not a cell with a value of zero.
`cp_evaluation.metrics.summarise` carries `n_fitted`, `n_ineligible` and
`n_refused` alongside every metric for exactly this reason -- its own module
docstring puts it directly: "a metric computed over zero usable rows has to
be distinguishable from one that happens to equal zero". A benchmark heatmap
that painted "this method could not run here" the same colour as "this
method scored zero here" would be actively misleading, which is why Phase 14
recorded three statuses per row rather than dropping the ones that produced
no estimate.

Both `heatmap_data` and `axis_profile_data` therefore mark a cell absent when
its `n_fitted` is zero (or when the (method, scenario) or (method, axis
value) combination has no row in the summary at all), never when the
metric's value happens to be NaN. Those are genuinely different conditions:
a metric can be legitimately NaN with rows present. `accuracy_metrics` in
`cp_evaluation.metrics` returns `relative_bias` as NaN whenever the group's
mean truth is exactly zero -- the null scenarios the false-positive test
exists for -- specifically to avoid a `bias / 0` that would read as infinite
rather than undefined; those rows still have `n_fitted > 0` and must be
drawn as present, unplottable-numbered cells, not hidden as if the method
never ran there.

`draw_heatmap` renders over a masked array (masked by `absent`, not by
`numpy.isnan`) so an absent cell takes the colormap's "bad" colour instead
of landing on the value scale. A line chart has no such colour to fall back
on -- `draw_axis_profile` keeps an absent position off the value scale by
NaN-ing it out of the plotted line (so the line breaks there rather than
drawing straight across it) and marking it instead with an absence rug: a
row of marks drawn in blended coordinates, x on the real axis value but y a
fixed fraction of the axes height, so it has no position on the metric scale
at all and cannot be misread as a measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from cp_evaluation.grid import AXIS_BASELINES, SCENARIO_AXES
from cp_evaluation.metrics import GROUP_COLUMNS
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory


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
    (a measured example from the Phase 18 whole-branch review collapsed
    three scenario-C cells, n_fitted 10/10/0, into one cell marked absent).
    A per-axis view is `axis_profile_data`/`draw_axis_profile`, elsewhere in
    this module: one scenario's one-at-a-time slice along one axis, drawn as
    a line per method rather than pivoted into cells. A caller who only needs
    to keep pivoting to a heatmap must still filter to one slice (e.g. one
    `n_units`/`confounding_strength`/axis value) before doing so.

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


def _realised_treated_share(
    rows: pd.DataFrame, axis: str, axis_values: tuple[float, ...]
) -> tuple[float, ...] | None:
    """The mean realised treated share at each axis value, or `None` if unrecorded.

    A mean rather than a single row's value, and exact rather than an
    aggregation papering over disagreement: `summarise` computes this figure
    over *every* row in a (method, cell) group -- fitted, ineligible and
    refused alike, since it is a property of the cell's generated data
    rather than of any fit -- so every method sharing a cell already carries
    the identical number.

    NaN at an axis value with nothing finite to average, which keeps the
    tuple positionally aligned with `axis_values` rather than short.
    """
    if "realised_treated_share" not in rows.columns:
        return None

    shares: list[float] = []
    for value in axis_values:
        at_value = rows[rows[axis] == value]["realised_treated_share"].to_numpy(dtype=float)
        finite = at_value[np.isfinite(at_value)]
        shares.append(float(finite.mean()) if finite.size else float("nan"))
    return tuple(shares)


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
    the `treatment_intercept` axis sets the intercept, not a target share,
    and the realised share is recorded beside it because an intercept
    is not a quantity a reader of the eventual research report thinks in. A
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
    if axis not in PROFILE_AXES:
        raise ValueError(
            f"unknown axis {axis!r}; available: {list(PROFILE_AXES)}"
        )
    if axis not in summary.columns:
        raise KeyError(axis)

    rows = summary
    if "shape" in rows.columns:
        # Panel groups carry NaN for every axis -- `PanelAxes` has none of
        # them -- so they would survive no pin anyway; dropping them here
        # makes the reason explicit rather than incidental.
        rows = rows[rows["shape"] == "cross_section"]

    available_scenarios = sorted(str(s) for s in rows["scenario"].unique())
    rows = rows[rows["scenario"] == scenario]
    if rows.empty:
        raise ValueError(
            f"no cross-sectional rows for scenario {scenario!r}; "
            f"available: {available_scenarios}"
        )

    # A scenario that does not declare an axis has nothing to vary along it.
    # The grid as it stands gives such a scenario a single baseline value on
    # every axis, so the single-value refusal below would catch a summary the
    # current grid produced. It would not catch one persisted before
    # `confounding_strength` was gated by declaration
    # (`cp_evaluation.grid._confounding_values`): that grid crossed A, B, C
    # and F -- whose RandomAssignment makes the axis meaningless
    # (`cp_simulation.scenarios`) -- against every strength, so the column
    # varies even though the generated data behind it is identical, the
    # single-value check cannot fire, and a profile would draw a flat line a
    # reader cannot tell apart from a measured null result. Checking
    # declaration up front catches both cases at the same place, and names
    # the reason rather than a symptom.
    #
    # `n_units` is exempt: no scenario declares it -- it is crossed against
    # `confounding_strength` rather than varied one-at-a-time -- but it
    # genuinely moves every scenario's data regardless of declaration.
    #
    # Skipped outright when `scenario` is not a key in `SCENARIO_AXES` at
    # all: this function cannot vouch for a scenario the registry does not
    # know, and the axis-column, empty-slice and single-value refusals
    # elsewhere in this function still apply to it.
    if axis != "n_units" and scenario in SCENARIO_AXES and axis not in SCENARIO_AXES[scenario]:
        declared = sorted(SCENARIO_AXES[scenario])
        raise ValueError(
            f"scenario {scenario!r} does not declare axis {axis!r} "
            f"(cp_simulation.scenarios.SCENARIOS[{scenario!r}].axes); it "
            f"declares {declared}. This scenario never reads {axis!r}, so any "
            "rows a summary carries along it describe identical data (an "
            "older grid crossed confounding_strength against every "
            "scenario, declared or not). A profile would draw a flat line a "
            "reader cannot tell apart from a measured null result. Choose "
            f"one of {declared}, or axis='n_units'."
        )

    pinned: dict[str, float] = {}
    for name, baseline in AXIS_BASELINES.items():
        if name != axis and name in rows.columns:
            pinned[name] = baseline
            rows = rows[rows[name] == baseline]

    if axis != "n_units" and "n_units" in rows.columns and not rows.empty:
        anchor = float(rows["n_units"].max())
        pinned["n_units"] = anchor
        rows = rows[rows["n_units"] == anchor]

    if rows.empty:
        raise ValueError(
            f"no rows for scenario {scenario!r} along {axis!r} once every other "
            f"axis was pinned to its baseline ({pinned}). The summary carries no "
            "cell at that anchor -- check it came from a grid that varies this "
            "axis one at a time."
        )

    axis_values = tuple(sorted(float(value) for value in rows[axis].unique()))
    if len(axis_values) < 2:
        raise ValueError(
            f"scenario {scenario!r} does not vary along {axis!r}: every row sits "
            f"at {axis_values[0]}. For a scenario the registry knows, this is not "
            "a declaration problem -- an axis such a scenario does not declare is "
            "refused earlier, before this check runs. Two callers reach here with "
            "that check skipped rather than passed: `n_units`, which no scenario "
            "declares, and a scenario absent from the registry. The usual reason is "
            "a summary already narrowed to one point on "
            f"{axis!r}, the shape a `ci_config()` run, or any summary pre-filtered "
            "to one value, would carry. This profile would otherwise be a single "
            "point drawn as though it were a measured flat line -- which a reader "
            "cannot tell apart from an axis that genuinely moved nothing."
        )

    group_sizes = rows.groupby(["method", axis]).size()
    duplicated = group_sizes[group_sizes > 1]
    if not duplicated.empty:
        method, value = duplicated.index[0]
        group = rows[(rows["method"] == method) & (rows[axis] == value)]
        differing = _differing_identity_columns(
            group, exclude=("method", "scenario", axis)
        )
        if differing:
            raise ValueError(
                f"axis_profile_data needs exactly one row per (method, {axis}), "
                f"but ({method!r}, {value}) has {len(group)} rows, differing in "
                f"{differing}. Filter the summary to one value of each of those "
                "columns before building a profile."
            )
        raise ValueError(
            f"axis_profile_data needs exactly one row per (method, {axis}), but "
            f"({method!r}, {value}) has {len(group)} rows that share the same "
            "grid-cell identity (no identity column differs between them) -- "
            "deduplicate the summary before building a profile."
        )

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
        realised_treated_share=_realised_treated_share(rows, axis, axis_values),
    )


# The fraction of the axes height the absence rug occupies, bottom and top.
# Fixed rather than proportional to the method count so ten methods take the
# same strip two do, instead of turning the rug into a third of the figure.
_RUG_BAND: tuple[float, float] = (0.02, 0.12)


def _rug_positions(n_methods: int) -> np.ndarray:
    """One rug row per method, evenly spread inside `_RUG_BAND`.

    Distinct rows rather than one shared line, so two methods absent at the
    same axis value draw two visible marks rather than one on top of
    another. A single method sits on the band's floor; `numpy.linspace`
    would otherwise put it at the top.
    """
    low, high = _RUG_BAND
    if n_methods < 2:
        return np.array([low])
    return np.linspace(low, high, n_methods)


def _tick_labels(data: AxisProfileData) -> list[str]:
    """One x tick label per axis value, annotated where the raw number does not read.

    `treatment_intercept` is a logit-scale parameter. The axis
    sets the intercept, not a target share, and the realised share is recorded
    beside it because an intercept is not a quantity a reader of the
    eventual research report thinks in -- so the share goes on the tick,
    where a reader meets the number.

    Only that axis. `effect_scale` does not touch assignment at all, and
    annotating every tick of every axis with a constant would be noise
    rather than a mitigation. A missing or non-finite
    share falls back to the bare value rather than printing "nan% treated".
    """
    bare = [f"{value:g}" for value in data.axis_values]
    shares = data.realised_treated_share
    if data.axis != "treatment_intercept" or shares is None:
        return bare
    return [
        f"{label}\n({share:.0%} treated)" if np.isfinite(share) else label
        for label, share in zip(bare, shares, strict=True)
    ]


def draw_axis_profile(data: AxisProfileData, ax: Axes) -> Axes:
    """Draw one line per method along `data.axis`, on `ax`.

    Plots `numpy.where(data.absent, nan, data.values)`, never `data.values`.
    That is the drawing half of the absent-versus-zero guarantee, and two
    things follow from it. A cell marked absent draws no marker even if its
    `values` entry is finite -- matplotlib will not auto-hide a finite
    number, so a function that plotted `values` directly and relied on NaN
    would put a point on the value scale where nothing was measured. And the
    line *breaks* at an absent value rather than being drawn straight
    across it, which would assert a value between two measurements where
    there is none.

    A point marker, not a bare line, for `draw_counterfactual_path`'s
    reason: a profile can legitimately have a single plottable value between
    two absent ones, and `ax.plot` with the default style draws nothing
    visible for a lone point.

    Absence gets its own visual channel rather than a y-position. The
    heatmap keeps an absent cell off the value scale by painting it in the
    colormap's "bad" colour; a line chart has no such channel, so an absent
    (method, axis value) is marked in *blended* coordinates -- x through
    `ax.transData` so it lines up with the axis value it belongs to, y
    through `ax.transAxes` so it sits at a fixed fraction of the axes height
    and has no position on the metric scale at all. Each method gets its own
    row inside `_RUG_BAND`, coloured from its own line, so a mark is
    attributable without a second legend.
    """
    x = np.asarray(data.axis_values, dtype=float)
    plottable = np.where(data.absent, np.nan, data.values)

    rug_y = _rug_positions(len(data.methods))
    # x in data coordinates, y in axes fractions: a rug mark lines up with
    # the axis value it belongs to but has no position on the metric scale
    # at all, so it can never be read as a measurement.
    rug_transform = blended_transform_factory(ax.transData, ax.transAxes)

    for i, method in enumerate(data.methods):
        (line,) = ax.plot(x, plottable[i], marker="o", label=method)
        absent_x = x[np.asarray(data.absent[i], dtype=bool)]
        if absent_x.size:
            ax.plot(
                absent_x,
                np.full(absent_x.shape, rug_y[i]),
                marker="x",
                linestyle="none",
                color=line.get_color(),
                transform=rug_transform,
                # The rug is an annotation on its method's line, not a
                # series of its own; a second legend entry per method would
                # double the legend for no information.
                label="_nolegend_",
            )

    # set_xticks before set_xticklabels: a FixedLocator must be in place
    # first, or matplotlib warns -- and this suite turns warnings into
    # errors.
    ax.set_xticks(list(x))
    ax.set_xticklabels(_tick_labels(data))
    ax.set_xlabel(data.axis)
    ax.set_ylabel(data.metric)
    ax.set_title(f"{data.metric} along {data.axis} (scenario {data.scenario})")
    ax.legend()
    return ax
