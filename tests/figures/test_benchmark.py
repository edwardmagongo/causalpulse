import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from cp_evaluation.metrics import GROUP_COLUMNS, summarise
from cp_evaluation.run import ROW_COLUMNS
from cp_figures.benchmark import (
    AxisProfileData,
    HeatmapData,
    axis_profile_data,
    draw_axis_profile,
    draw_heatmap,
    heatmap_data,
)


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def _summary():
    return pd.DataFrame(
        [
            {
                "method": "ipw",
                "scenario": "A",
                "bias": 0.01,
                "n_fitted": 15,
                "n_ineligible": 0,
                "n_refused": 0,
            },
            {
                "method": "ipw",
                "scenario": "B",
                "bias": 0.00,
                "n_fitted": 15,
                "n_ineligible": 0,
                "n_refused": 0,
            },
            {
                "method": "twfe",
                "scenario": "A",
                "bias": float("nan"),
                "n_fitted": 0,
                "n_ineligible": 15,
                "n_refused": 0,
            },
            {
                "method": "twfe",
                "scenario": "B",
                "bias": 0.04,
                "n_fitted": 15,
                "n_ineligible": 0,
                "n_refused": 0,
            },
        ]
    )


def test_it_pivots_to_method_by_scenario():
    data = heatmap_data(_summary(), metric="bias")

    assert data.methods == ("ipw", "twfe")
    assert data.scenarios == ("A", "B")
    assert data.values.shape == (2, 2)


def test_a_cell_with_no_fitted_rows_is_absent_not_zero():
    # "could not run here" and "scored zero here" must not be the same colour.
    # bool(...) rather than `is True`: `data.absent` is a numpy boolean
    # array, and indexing it yields numpy.bool_, which is never identical to
    # Python's True singleton even when it compares equal.
    data = heatmap_data(_summary(), metric="bias")

    row = data.methods.index("twfe")
    column = data.scenarios.index("A")
    assert np.isnan(data.values[row][column])
    assert bool(data.absent[row][column]) is True


def test_a_cell_with_fitted_rows_is_present():
    data = heatmap_data(_summary(), metric="bias")

    row = data.methods.index("ipw")
    column = data.scenarios.index("A")
    assert data.values[row][column] == pytest.approx(0.01)
    assert bool(data.absent[row][column]) is False


def test_a_genuine_zero_is_present_not_absent():
    # ipw on scenario B has bias exactly 0.0 with 15 fitted rows. It must not
    # be confused with a cell that never ran. A mutation that marks a cell
    # absent whenever its value is falsy (rather than reading n_fitted) would
    # pass every other test here but fail this one.
    data = heatmap_data(_summary(), metric="bias")

    row = data.methods.index("ipw")
    column = data.scenarios.index("B")
    assert data.values[row][column] == pytest.approx(0.0)
    assert bool(data.absent[row][column]) is False


def test_every_fitted_cell_is_correctly_placed_not_just_one():
    # A pivot bug that transposes rows and columns, or that assigns the last
    # row seen to every cell, could still pass the single-cell checks above
    # if it happened to get ipw/A and ipw/B right by coincidence. Checking
    # twfe/B too pins down that each (method, scenario) pair lands in its own
    # cell rather than all rows collapsing onto one.
    data = heatmap_data(_summary(), metric="bias")

    row = data.methods.index("twfe")
    column = data.scenarios.index("B")
    assert data.values[row][column] == pytest.approx(0.04)
    assert bool(data.absent[row][column]) is False


def test_a_combination_missing_from_the_frame_entirely_is_absent():
    # summarise only emits a row for a (method, scenario) pair that actually
    # appears in the grid. A combination that never appears at all -- not
    # even as a refused/ineligible row -- must still read as absent, not
    # default to a plotted NaN-as-zero or crash on a missing key.
    summary = pd.DataFrame(
        [
            {
                "method": "ipw",
                "scenario": "A",
                "bias": 0.01,
                "n_fitted": 15,
                "n_ineligible": 0,
                "n_refused": 0,
            },
            {
                "method": "twfe",
                "scenario": "B",
                "bias": 0.02,
                "n_fitted": 15,
                "n_ineligible": 0,
                "n_refused": 0,
            },
        ]
    )

    data = heatmap_data(summary, metric="bias")

    row = data.methods.index("ipw")
    column = data.scenarios.index("B")
    assert data.absent[row][column]
    assert np.isnan(data.values[row][column])


def test_a_legitimately_nan_value_with_fitted_rows_is_present_not_absent():
    # relative_bias in a null-truth cell is nan even though rows were fitted
    # (mean_truth == 0.0 divides nothing meaningful, see
    # cp_evaluation.metrics.accuracy_metrics). absent must track n_fitted,
    # never np.isnan(value), or this cell would be wrongly hidden as if the
    # method could not run here.
    summary = pd.DataFrame(
        [
            {
                "method": "ipw",
                "scenario": "A",
                "relative_bias": float("nan"),
                "n_fitted": 15,
                "n_ineligible": 0,
                "n_refused": 0,
            },
        ]
    )

    data = heatmap_data(summary, metric="relative_bias")

    assert np.isnan(data.values[0][0])
    assert bool(data.absent[0][0]) is False


def test_an_unknown_metric_is_refused_by_name():
    with pytest.raises(KeyError, match="frobnicate"):
        heatmap_data(_summary(), metric="frobnicate")


def _row(**overrides) -> dict[str, object]:
    """A minimal `ROW_COLUMNS` row, everything not overridden left NaN --
    mirrors `test_metrics.py`'s own `_frame` helper, but building the
    dict-per-row form `summarise()` groups over rather than a frame."""
    row: dict[str, object] = dict.fromkeys(ROW_COLUMNS, np.nan)
    row.update(shape="cross_section", scenario="C", method="lin_regression", status="fitted")
    row.update(overrides)
    return row


def _real_summary_with_two_effect_scale_cells() -> pd.DataFrame:
    """A genuine `cp_evaluation.metrics.summarise()` output -- not a
    hand-built fixture -- carrying two rows for (`lin_regression`, `C`) that
    differ only in `effect_scale`. This is the actual shape `full_config()`'s
    widened grid produces before a caller filters to one slice: one row per
    axis value, dozens of columns wide (the method, one identity per
    `GROUP_COLUMNS`, and every metric column `summarise` computes -- `bias`,
    `rmse_mcse`, `realised_treated_share`, ... -- which are most of them). A fixture built by hand
    with only a couple of columns could pass a comprehension that lists
    every differing column, metric or not; this can't, because leaking even
    one metric column into the message is what the test below checks for.
    """
    rows = [
        _row(effect_scale=0.5, n_units=2000, seed=0, point=0.30, truth=0.30),
        _row(effect_scale=0.5, n_units=2000, seed=1, point=0.32, truth=0.30),
        _row(effect_scale=2.0, n_units=2000, seed=0, point=-0.10, truth=-0.06),
        _row(effect_scale=2.0, n_units=2000, seed=1, point=-0.08, truth=-0.06),
    ]
    frame = pd.DataFrame(rows, columns=list(ROW_COLUMNS))
    return summarise(frame)


def test_more_than_one_row_per_method_and_scenario_is_refused_naming_only_identity_columns():
    # A real summarise() output for two effect_scale values necessarily
    # disagrees on far more than effect_scale alone: bias, rmse_mcse,
    # realised_treated_share and every other metric family differ between
    # the two rows too, since they were computed from different data. The
    # refusal must name only effect_scale -- the one column that actually
    # identifies the grid cell -- and never a metric column, which a caller
    # cannot "filter to one value of" in any meaningful sense.
    summary = _real_summary_with_two_effect_scale_cells()
    assert summary.shape[0] == 2
    # Sanity check this really is a >20-column real summarise() output, not
    # an accidental hand-built stand-in that would let this test pass for
    # the wrong reason.
    assert len(summary.columns) >= 20

    with pytest.raises(ValueError, match=r"differing in \['effect_scale'\]") as exc_info:
        heatmap_data(summary, metric="bias")

    message = str(exc_info.value)
    metric_columns = [
        column
        for column in summary.columns
        if column not in ("method", "scenario", "shape", "n_units", "n_periods", "effect_scale")
    ]
    assert metric_columns, "fixture sanity: summarise() must have produced metric columns"
    for column in metric_columns:
        assert column not in message, f"metric column {column!r} leaked into the refusal"


def test_more_than_one_row_per_method_and_scenario_with_no_identity_difference_says_so():
    # Two rows for the same (method, scenario) that carry none of
    # cp_evaluation.metrics.GROUP_COLUMNS's identity columns at all share
    # the same grid-cell identity outright. The old comprehension reported
    # this as "differing in []", which names nothing a caller could act on;
    # the refusal must say plainly that the rows are identity-duplicates.
    summary = pd.DataFrame(
        [
            {
                "method": "ipw",
                "scenario": "A",
                "bias": 0.01,
                "n_fitted": 15,
                "n_ineligible": 0,
                "n_refused": 0,
            },
            {
                "method": "ipw",
                "scenario": "A",
                "bias": 0.05,
                "n_fitted": 15,
                "n_ineligible": 0,
                "n_refused": 0,
            },
        ]
    )

    with pytest.raises(ValueError, match="same grid-cell identity"):
        heatmap_data(summary, metric="bias")


def test_a_summary_already_filtered_to_one_slice_still_pivots():
    # The guard must not fire on the ordinary case: a summary that already
    # carries axis-identity columns but has been filtered to a single value
    # of each, so every (method, scenario) pair still has exactly one row.
    summary = pd.DataFrame(
        [
            {
                "method": "lin_regression",
                "scenario": "C",
                "effect_scale": 1.0,
                "bias": 0.00,
                "n_fitted": 10,
                "n_ineligible": 0,
                "n_refused": 0,
            },
            {
                "method": "ipw",
                "scenario": "C",
                "effect_scale": 1.0,
                "bias": 0.02,
                "n_fitted": 10,
                "n_ineligible": 0,
                "n_refused": 0,
            },
        ]
    )

    data = heatmap_data(summary, metric="bias")

    assert data.values.shape == (2, 1)
    assert not data.absent.any()


def test_drawing_produces_one_cell_per_method_and_scenario(ax):
    data = heatmap_data(_summary(), metric="bias")

    draw_heatmap(data, ax)

    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape == (2, 2)


def test_drawing_masks_absent_cells_rather_than_colouring_them(ax):
    # The whole point of tracking absence separately from NaN: an
    # implementation that just did ax.imshow(data.values) would still
    # produce one image of the right shape (passing the test above) but
    # would not mask anything, painting the absent cell some colour on the
    # scale instead of the colormap's "bad" colour.
    data = heatmap_data(_summary(), metric="bias")

    draw_heatmap(data, ax)

    drawn = ax.images[0].get_array()
    assert isinstance(drawn, np.ma.MaskedArray)
    mask = np.ma.getmaskarray(drawn)
    np.testing.assert_array_equal(mask, np.asarray(data.absent))

    present_row = data.methods.index("ipw")
    present_column = data.scenarios.index("B")
    assert not mask[present_row][present_column]
    assert drawn[present_row][present_column] == pytest.approx(0.0)


def test_drawing_masks_by_the_absent_flag_even_when_the_value_is_finite(ax):
    # matplotlib's own imshow auto-masks NaN and inf values regardless of
    # any mask passed in (`cbook.safe_masked_invalid`, confirmed empirically
    # against this matplotlib install: an explicit mask=[[False]] over a NaN
    # value still comes back masked). That makes a NaN-valued fixture
    # useless for telling "draw_heatmap passes data.absent through as the
    # mask" apart from "draw_heatmap ignores data.absent and relies on
    # imshow's own NaN handling" -- both draw the same masked cell either
    # way, so a test built on one would never be able to fail.
    #
    # A *finite* value marked absent breaks that coincidence: imshow never
    # auto-masks a finite number (confirmed against this install too), so
    # this cell comes back masked only if draw_heatmap actually threads
    # data.absent through to imshow as the mask.
    data = HeatmapData(
        methods=("ipw",),
        scenarios=("A",),
        values=np.array([[0.0]]),
        absent=np.array([[True]]),
        metric="bias",
    )

    draw_heatmap(data, ax)

    drawn = ax.images[0].get_array()
    mask = np.ma.getmaskarray(drawn)
    assert mask[0][0]


def test_drawing_labels_the_axes_with_methods_and_scenarios(ax):
    data = heatmap_data(_summary(), metric="bias")

    draw_heatmap(data, ax)

    assert [t.get_text() for t in ax.get_yticklabels()] == ["ipw", "twfe"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]


def test_drawing_returns_the_axes_and_leaves_pyplot_alone(ax):
    before = plt.get_fignums()
    data = heatmap_data(_summary(), metric="bias")

    assert draw_heatmap(data, ax) is ax
    assert plt.get_fignums() == before


def test_heatmap_data_is_a_dataclass_with_the_documented_fields():
    data = heatmap_data(_summary(), metric="bias")

    assert isinstance(data, HeatmapData)
    assert data.metric == "bias"


def _profile_frame():
    """A real `ROW_COLUMNS` frame for scenario C's effect_scale profile.

    The five (n_units, effect_scale, noise_scale) combinations below are
    exactly what `full_config()` emits for scenario C along this axis, plus
    the two rows that must be *excluded*:

      - (500, 1.0, 0.0) is the baseline cell at a non-anchor n_units. The
        grid's cross-product produces the baseline at every n_units, but
        `_expand_cross_section` anchors the perturbed cells at
        max(axes.n_units). Without the n_units pin, effect_scale=1.0 would
        carry two rows per method and the duplicate refusal would fire.
      - (2000, 1.0, 1.3) is a *different* axis perturbed. One-at-a-time
        means it belongs to the noise_scale profile, not this one.

    Two seeds per cell so `summarise` has something to average; bias is
    exact because both seeds carry the same point and truth.
    """
    # Keyed on the whole cell, not just (method, effect_scale): the two
    # rows that must be EXCLUDED sit at effect_scale=1.0 alongside the row
    # that must be kept, so they need their own distinguishable biases. If
    # all three carried the same number, a rule that anchored on min(n_units)
    # instead of max, or that skipped the noise_scale pin, would select the
    # wrong row and every assertion would still pass.
    biases = {
        ("lin_regression", 2000, 0.5, 0.0): 0.01,
        ("lin_regression", 2000, 1.0, 0.0): 0.00,  # a genuine zero, not an absent cell
        ("lin_regression", 2000, 2.0, 0.0): 0.04,
        ("ipw", 2000, 0.5, 0.0): 0.02,
        ("ipw", 2000, 1.0, 0.0): 0.03,
        ("ipw", 2000, 2.0, 0.0): 0.05,
        # Must never reach the profile:
        ("lin_regression", 500, 1.0, 0.0): 0.91,
        ("ipw", 500, 1.0, 0.0): 0.92,
        ("lin_regression", 2000, 1.0, 1.3): 0.81,
        ("ipw", 2000, 1.0, 1.3): 0.82,
    }
    rows = []
    for method in ("ipw", "lin_regression"):
        for n_units, effect_scale, noise_scale in (
            (2000, 0.5, 0.0),
            (2000, 1.0, 0.0),
            (2000, 2.0, 0.0),
            (500, 1.0, 0.0),
            (2000, 1.0, 1.3),
        ):
            truth = 0.03 * effect_scale
            bias = biases[(method, n_units, effect_scale, noise_scale)]
            for seed in (0, 1):
                rows.append(
                    _row(
                        method=method,
                        n_units=n_units,
                        confounding_strength=1.0,
                        treatment_intercept=0.0,
                        noise_scale=noise_scale,
                        effect_scale=effect_scale,
                        heterogeneity_scale=1.0,
                        realised_treated_share=0.5,
                        seed=seed,
                        truth=truth,
                        point=truth + bias,
                    )
                )
    return pd.DataFrame(rows, columns=list(ROW_COLUMNS))


def _profile_summary():
    return summarise(_profile_frame())


def test_the_profile_selects_one_axis_and_sorts_its_values():
    data = axis_profile_data(
        _profile_summary(), "bias", axis="effect_scale", scenario="C"
    )

    assert data.axis == "effect_scale"
    assert data.scenario == "C"
    assert data.metric == "bias"
    assert data.axis_values == (0.5, 1.0, 2.0)
    assert data.methods == ("ipw", "lin_regression")
    assert data.values.shape == (2, 3)


def test_the_profile_includes_the_baseline_point_from_the_cross_product():
    # `_expand_cross_section` SKIPS the axis's own default in its
    # one-at-a-time loop (`if value == baseline_value: continue`); the
    # baseline cell comes from the n_units x confounding_strength
    # cross-product instead. A rule that gathered only the perturbed cells
    # would drop the one point the whole comparison is anchored on, and
    # would do it silently -- the figure would still draw, with two points.
    data = axis_profile_data(
        _profile_summary(), "bias", axis="effect_scale", scenario="C"
    )

    assert 1.0 in data.axis_values
    row = data.methods.index("lin_regression")
    column = data.axis_values.index(1.0)
    assert data.values[row][column] == pytest.approx(0.00)


def test_a_baseline_row_at_a_smaller_n_units_is_excluded_by_the_anchor_pin():
    # The grid produces the baseline cell at EVERY n_units, but anchors the
    # perturbed cells at max(axes.n_units). Mixing a 500-unit baseline in
    # beside 2000-unit perturbed cells would compare cells that differ in
    # two things at once. With the pin, effect_scale=1.0 has exactly one row
    # per method; without it, two -- which the duplicate refusal would catch,
    # so this test pins the *correct* behaviour rather than the crash.
    summary = _profile_summary()
    assert (summary["n_units"] == 500).any(), "fixture sanity: a 500-unit row must exist"
    # summarise() itself emits rows sorted ascending by n_units, so the
    # 500-unit row always precedes the 2000-unit one in iteration order --
    # meaning a plain "last write wins" placement loop would coincidentally
    # land on the correct 2000-unit value even with the anchor pin deleted,
    # and this test would never be able to fail. Reversing the row order
    # here removes that coincidence without touching what data means: the
    # anchor pin must filter to the max-n_units row regardless of the order
    # rows arrive in, while a mutant relying on iteration order to save it
    # would now keep the wrong (500-unit) row instead.
    summary = summary.iloc[::-1].reset_index(drop=True)

    data = axis_profile_data(summary, "bias", axis="effect_scale", scenario="C")

    assert data.values.shape == (2, 3)
    # The 500-unit baseline row carries bias 0.91/0.92, the 2000-unit one
    # 0.00/0.03. Asserting WHICH value landed is what distinguishes the
    # anchor being max(n_units) from it being min -- a shape check alone
    # would pass either way.
    for method, expected in (("lin_regression", 0.00), ("ipw", 0.03)):
        row = data.methods.index(method)
        column = data.axis_values.index(1.0)
        assert data.values[row][column] == pytest.approx(expected), method


def test_a_row_with_another_axis_perturbed_is_excluded():
    # (2000, effect_scale=1.0, noise_scale=1.3) belongs to the noise_scale
    # profile. Leaving it in would put two rows on effect_scale=1.0 and
    # average a noisy cell into the baseline point.
    summary = _profile_summary()
    assert (summary["noise_scale"] == 1.3).any(), "fixture sanity: a noisy row must exist"

    data = axis_profile_data(summary, "bias", axis="effect_scale", scenario="C")

    assert data.values.shape == (2, 3)
    # The noisy row carries bias 0.81/0.82. Asserting the quiet row's value
    # landed is what makes this test able to fail: dropping the noise_scale
    # pin selects the wrong row, not merely an extra one.
    for method, expected in (("lin_regression", 0.00), ("ipw", 0.03)):
        row = data.methods.index(method)
        column = data.axis_values.index(1.0)
        assert data.values[row][column] == pytest.approx(expected), method


def test_every_cell_is_correctly_placed_not_just_one():
    # A selection bug that transposes methods and values, or assigns the
    # last row seen to every cell, could still pass a single-cell check.
    data = axis_profile_data(
        _profile_summary(), "bias", axis="effect_scale", scenario="C"
    )

    expected = {
        ("ipw", 0.5): 0.02,
        ("ipw", 1.0): 0.03,
        ("ipw", 2.0): 0.05,
        ("lin_regression", 0.5): 0.01,
        ("lin_regression", 1.0): 0.00,
        ("lin_regression", 2.0): 0.04,
    }
    for (method, value), bias in expected.items():
        row = data.methods.index(method)
        column = data.axis_values.index(value)
        assert data.values[row][column] == pytest.approx(bias), (method, value)
        assert bool(data.absent[row][column]) is False


def test_a_genuine_zero_is_present_not_absent_along_an_axis():
    # lin_regression at effect_scale=1.0 has bias exactly 0.0 with fitted
    # rows. An implementation that marked a cell absent whenever its value
    # was falsy, rather than reading n_fitted, would pass every other test
    # here and fail this one.
    data = axis_profile_data(
        _profile_summary(), "bias", axis="effect_scale", scenario="C"
    )

    row = data.methods.index("lin_regression")
    column = data.axis_values.index(1.0)
    assert data.values[row][column] == pytest.approx(0.0)
    assert bool(data.absent[row][column]) is False


def test_a_cell_with_no_fitted_rows_is_absent_not_zero_along_an_axis():
    # The heatmap's guarantee, carried across. Every ipw row at
    # effect_scale=2.0 is ineligible, so that cell never ran -- it is not a
    # cell that scored zero.
    frame = _profile_frame()
    ineligible = (
        (frame["method"] == "ipw")
        & (frame["effect_scale"] == 2.0)
        & (frame["noise_scale"] == 0.0)
    )
    assert ineligible.any(), "fixture sanity"
    frame.loc[ineligible, "status"] = "ineligible"

    data = axis_profile_data(summarise(frame), "bias", axis="effect_scale", scenario="C")

    row = data.methods.index("ipw")
    column = data.axis_values.index(2.0)
    assert bool(data.absent[row][column]) is True
    assert np.isnan(data.values[row][column])


def test_a_legitimately_nan_metric_with_fitted_rows_is_present_along_an_axis():
    # relative_bias is NaN wherever a group's mean truth is exactly zero
    # (cp_evaluation.metrics.accuracy_metrics), and those rows have
    # n_fitted > 0. absent must track n_fitted, never numpy.isnan(value),
    # or a null-truth cell would be wrongly hidden as if it never ran.
    frame = _profile_frame()
    frame["truth"] = 0.0
    summary = summarise(frame)

    data = axis_profile_data(summary, "relative_bias", axis="effect_scale", scenario="C")

    row = data.methods.index("ipw")
    column = data.axis_values.index(1.0)
    assert np.isnan(data.values[row][column])
    assert bool(data.absent[row][column]) is False


def test_a_method_missing_at_one_axis_value_entirely_is_absent():
    # summarise only emits a row for a (method, cell) that appears in the
    # results. A method with no row at one axis value must read as absent,
    # not crash on a missing key and not default to a plotted zero.
    frame = _profile_frame()
    keep = ~(
        (frame["method"] == "ipw")
        & (frame["effect_scale"] == 0.5)
    )
    summary = summarise(frame[keep])

    data = axis_profile_data(summary, "bias", axis="effect_scale", scenario="C")

    row = data.methods.index("ipw")
    column = data.axis_values.index(0.5)
    assert bool(data.absent[row][column]) is True
    assert np.isnan(data.values[row][column])


def test_n_units_is_a_selectable_axis_with_no_anchor_pin():
    # n_units is crossed rather than varied one-at-a-time, but with every
    # AXIS_NAMES axis pinned to its baseline the surviving rows are a
    # genuine profile. This is the rule's one branch: there is no n_units
    # pin when n_units is itself the axis.
    data = axis_profile_data(
        _profile_summary(), "bias", axis="n_units", scenario="C"
    )

    assert data.axis_values == (500.0, 2000.0)
    assert data.values.shape == (2, 2)
    assert not data.absent.any()


def test_axis_profile_data_is_a_dataclass_with_the_documented_fields():
    data = axis_profile_data(
        _profile_summary(), "bias", axis="effect_scale", scenario="C"
    )

    assert isinstance(data, AxisProfileData)
    assert isinstance(data.values, np.ndarray)
    assert isinstance(data.absent, np.ndarray)
    assert data.absent.dtype == bool


def test_an_unknown_profile_metric_is_refused_by_name():
    # scenario="Z" is not in the summary either, and that guard runs later
    # (after the metric check). If the metric guard were deleted, execution
    # would reach the scenario guard instead and raise a ValueError -- not a
    # KeyError -- so this test would fail rather than pass for the wrong
    # reason. Without the mismatched scenario, deleting the metric guard
    # would let placement's own `row[metric]` raise the identical
    # KeyError('frobnicate'), and this test would pass either way.
    with pytest.raises(KeyError, match="frobnicate"):
        axis_profile_data(
            _profile_summary(), "frobnicate", axis="effect_scale", scenario="Z"
        )


def test_an_unknown_axis_is_refused_naming_it_and_the_permitted_set():
    with pytest.raises(ValueError, match="wobble") as exc_info:
        axis_profile_data(_profile_summary(), "bias", axis="wobble", scenario="C")

    message = str(exc_info.value)
    # The permitted set must be listed -- a caller who guessed wrong needs
    # to see what the alternatives are, not just that the guess failed.
    assert "effect_scale" in message
    assert "n_units" in message


def test_an_axis_column_missing_from_the_summary_is_refused_by_name():
    # A pre-Phase-18 summary carries no heterogeneity_scale column at all.
    # That is a different failure from "the axis is not a real axis", and
    # gets the KeyError an absent column deserves.
    #
    # scenario="Z" is also not in the summary, and that guard runs later
    # (after the axis-column check) without ever touching the axis column
    # itself. If the axis-column guard were deleted, the pin loop would skip
    # heterogeneity_scale (it never pins the axis under study) and execution
    # would reach the scenario guard instead, raising a ValueError -- not a
    # KeyError -- so this test would fail rather than pass for the wrong
    # reason. Without the mismatched scenario, deleting the guard would let
    # `rows[axis].unique()` raise the identical KeyError('heterogeneity_scale')
    # from pandas' own column access, and this test would pass either way.
    summary = _profile_summary().drop(columns=["heterogeneity_scale"])

    with pytest.raises(KeyError, match="heterogeneity_scale"):
        axis_profile_data(summary, "bias", axis="heterogeneity_scale", scenario="Z")


def test_a_scenario_not_in_the_summary_is_refused_naming_what_is_available():
    with pytest.raises(ValueError, match="'Z'") as exc_info:
        axis_profile_data(_profile_summary(), "bias", axis="effect_scale", scenario="Z")

    assert "C" in str(exc_info.value)


def test_a_scenario_that_does_not_vary_along_the_axis_is_refused():
    # Scenario C declares effect_scale, treatment_intercept and noise_scale
    # -- not heterogeneity_scale (cp_simulation.scenarios: C's effect is a
    # ConstantEffect, whose CATE spread is structurally zero, so there is no
    # increment for the axis to scale). Every C row therefore sits at the
    # heterogeneity_scale baseline, and a profile along it would be a single
    # point drawn as if it were a measured flat line. A reader cannot tell
    # "measured, did not move" from "never varied here" by looking.
    with pytest.raises(ValueError, match="heterogeneity_scale") as exc_info:
        axis_profile_data(
            _profile_summary(), "bias", axis="heterogeneity_scale", scenario="C"
        )

    message = str(exc_info.value)
    assert "'C'" in message
    assert "1.0" in message  # the single value found


def test_an_empty_slice_is_refused_rather_than_returning_an_empty_profile():
    # Every row perturbed on noise_scale, so pinning noise_scale to its
    # baseline leaves nothing. Returning an empty AxisProfileData would draw
    # a blank axes that looks like a successful figure.
    frame = _profile_frame()
    frame["noise_scale"] = 1.3

    with pytest.raises(ValueError, match="no rows"):
        axis_profile_data(summarise(frame), "bias", axis="effect_scale", scenario="C")


def test_more_than_one_row_per_method_and_axis_value_is_refused_naming_identity_columns():
    # The fixture duplicates every row at effect_scale=1.0, n_units=2000,
    # noise_scale=0.0 -- both ipw and lin_regression -- differing only in
    # n_periods, an identity column the profile does not pin. The guard
    # fires on the first duplicated group groupby.size() finds, which is
    # ('ipw', 1.0), not lin_regression. Either way the refusal must name
    # n_periods and no metric column: a caller cannot "filter to one value
    # of" bias.
    frame = _profile_frame()
    extra = frame[
        (frame["effect_scale"] == 1.0)
        & (frame["n_units"] == 2000)
        & (frame["noise_scale"] == 0.0)
    ].copy()
    extra["n_periods"] = 7.0
    # An exact copy of `frame`'s rows would carry identical point/truth, so
    # summarise() would give the n_periods=7.0 cell metric aggregates
    # numerically identical to the original's -- defining away the very
    # leak this test exists to catch: no metric column could ever differ,
    # so a mutant that checked every column instead of only GROUP_COLUMNS
    # would still slip through. Perturbing `point` gives the duplicate
    # cell its own, genuinely different metrics, the way two real grid
    # cells would differ.
    extra["point"] = extra["point"] + 0.5
    summary = summarise(pd.concat([frame, extra], ignore_index=True))
    assert len(summary.columns) >= 20, "fixture sanity: a real summarise() output"

    with pytest.raises(ValueError, match=r"differing in \['n_periods'\]") as exc_info:
        axis_profile_data(summary, "bias", axis="effect_scale", scenario="C")

    message = str(exc_info.value)
    metric_columns = [
        column
        for column in summary.columns
        if column not in GROUP_COLUMNS and column != "method"
    ]
    assert metric_columns, "fixture sanity: summarise() must have produced metric columns"
    for column in metric_columns:
        assert column not in message, f"metric column {column!r} leaked into the refusal"


def _intercept_frame():
    """Scenario C's treatment_intercept profile, with a different realised
    share at each intercept -- which is the whole reason the share is
    recorded. A logit intercept of -1.5/0.0/1.5 is not a quantity a reader
    thinks in; "21% treated" is."""
    shares = {-1.5: 0.21, 0.0: 0.50, 1.5: 0.79}
    rows = []
    for method in ("ipw", "lin_regression"):
        for intercept, share in shares.items():
            for seed in (0, 1):
                rows.append(
                    _row(
                        method=method,
                        n_units=2000,
                        confounding_strength=1.0,
                        treatment_intercept=intercept,
                        noise_scale=0.0,
                        effect_scale=1.0,
                        heterogeneity_scale=1.0,
                        realised_treated_share=share,
                        seed=seed,
                        truth=0.03,
                        point=0.04,
                    )
                )
    return pd.DataFrame(rows, columns=list(ROW_COLUMNS))


def test_the_realised_treated_share_travels_with_the_axis_values():
    data = axis_profile_data(
        summarise(_intercept_frame()), "bias", axis="treatment_intercept", scenario="C"
    )

    assert data.axis_values == (-1.5, 0.0, 1.5)
    assert data.realised_treated_share == pytest.approx((0.21, 0.50, 0.79))


def test_the_realised_treated_share_is_none_when_the_summary_has_no_such_column():
    summary = _profile_summary().drop(columns=["realised_treated_share"])

    data = axis_profile_data(summary, "bias", axis="effect_scale", scenario="C")

    assert data.realised_treated_share is None


def test_the_realised_treated_share_is_carried_for_every_axis_not_just_the_intercept():
    # It is a measurement summarise has already made, and an axis along
    # which the treated share stays flat is itself worth being able to read.
    # Only the display is axis-specific.
    data = axis_profile_data(
        _profile_summary(), "bias", axis="effect_scale", scenario="C"
    )

    assert data.realised_treated_share == pytest.approx((0.5, 0.5, 0.5))


def test_realised_treated_share_stays_aligned_when_one_axis_value_has_no_finite_values():
    # The function must emit NaN for an axis value with no finite
    # realised_treated_share values, preserving the tuple's positional alignment
    # with axis_values. A regression that skipped the NaN-bearing axis value
    # instead of appending NaN would return a shorter tuple, misaligning every
    # later entry: the 1.5 intercept's 0.79 share would be reported against
    # the 0.0 intercept's axis value, not its own.
    frame = _intercept_frame()
    # Make the 0.0 intercept's realised_treated_share all NaN
    frame.loc[frame["treatment_intercept"] == 0.0, "realised_treated_share"] = float("nan")

    data = axis_profile_data(
        summarise(frame), "bias", axis="treatment_intercept", scenario="C"
    )

    # The returned tuple must have the same length as axis_values
    assert len(data.realised_treated_share) == len(data.axis_values)

    # Find the index of the 0.0 intercept (the axis value with no finite shares)
    zero_index = data.axis_values.index(0.0)

    # The entry at that position must be NaN
    assert np.isnan(data.realised_treated_share[zero_index])

    # The other entries must still carry their correct values, matched to their
    # own axis values (not misaligned to a neighbour's). This catches the
    # specific failure mode where a skipped NaN-value would shift later entries.
    for i, axis_value in enumerate(data.axis_values):
        if axis_value != 0.0:
            expected_share = {-1.5: 0.21, 1.5: 0.79}[axis_value]
            assert data.realised_treated_share[i] == pytest.approx(
                expected_share
            ), f"axis_value {axis_value} must map to share {expected_share}"


def test_drawing_a_profile_draws_one_line_per_method_at_the_axis_values(ax):
    data = axis_profile_data(
        _profile_summary(), "bias", axis="effect_scale", scenario="C"
    )

    draw_axis_profile(data, ax)

    lines = [line for line in ax.lines if line.get_label() in data.methods]
    assert len(lines) == 2
    by_method = {line.get_label(): line for line in lines}
    np.testing.assert_allclose(by_method["ipw"].get_xdata(), [0.5, 1.0, 2.0])
    np.testing.assert_allclose(by_method["ipw"].get_ydata(), [0.02, 0.03, 0.05])
    np.testing.assert_allclose(
        by_method["lin_regression"].get_ydata(), [0.01, 0.00, 0.04]
    )


def test_drawing_a_profile_masks_absent_positions_not_dropping_them(ax):
    # An absent point must be kept in place but NaN'd, not dropped from the
    # arrays. A draw_ that filtered absent values out would plot only the
    # present points and draw a line straight from one neighbour to the other,
    # visually asserting a trend across a cell that never ran. With the point
    # kept and NaN'd, the line breaks and a reader can see the gap.
    #
    # This test verifies the structural requirement (x array is complete, not
    # filtered). The sibling test verifies the value masking (y is NaN on an
    # absent position). Together they guard both halves of "mask, not drop".
    data = AxisProfileData(
        axis="effect_scale",
        scenario="C",
        metric="bias",
        axis_values=(0.5, 1.0, 2.0),
        methods=("ipw",),
        values=np.array([[0.02, 0.0, 0.05]]),  # finite value at absent pos
        absent=np.array([[False, True, False]]),
        realised_treated_share=None,
    )

    draw_axis_profile(data, ax)

    line = next(line for line in ax.lines if line.get_label() == "ipw")
    x = np.asarray(line.get_xdata(), dtype=float)
    y = np.asarray(line.get_ydata(), dtype=float)

    # The absent position is kept in the x array; it is not dropped
    np.testing.assert_allclose(x, [0.5, 1.0, 2.0])

    # The absent position is NaN'd in the y array
    assert np.isnan(y[1])
    assert not np.isnan(y[0]) and not np.isnan(y[2])


def test_drawing_a_profile_masks_by_the_absent_flag_even_when_the_value_is_finite(ax):
    # The mutation-catcher, mirroring the heatmap's own. matplotlib will not
    # auto-hide a finite number, so a draw_ that plotted data.values
    # directly and relied on NaN would draw a marker at 0.0 where nothing
    # was ever measured. Only threading data.absent through can hide it.
    data = AxisProfileData(
        axis="effect_scale",
        scenario="C",
        metric="bias",
        axis_values=(0.5, 1.0, 2.0),
        methods=("ipw",),
        values=np.array([[0.02, 0.0, 0.05]]),
        absent=np.array([[False, True, False]]),
        realised_treated_share=None,
    )

    draw_axis_profile(data, ax)

    line = next(line for line in ax.lines if line.get_label() == "ipw")
    y = np.asarray(line.get_ydata(), dtype=float)
    assert np.isnan(y[1]), "a finite value marked absent was plotted on the value scale"


def test_drawing_a_profile_labels_the_axes_and_names_the_scenario(ax):
    data = axis_profile_data(
        _profile_summary(), "bias", axis="effect_scale", scenario="C"
    )

    draw_axis_profile(data, ax)

    assert ax.get_xlabel() == "effect_scale"
    assert ax.get_ylabel() == "bias"
    # A profile read without knowing its scenario says nothing.
    title = ax.get_title()
    assert "bias" in title and "effect_scale" in title and "C" in title


def test_drawing_a_profile_returns_the_axes_and_leaves_pyplot_alone(ax):
    before = plt.get_fignums()
    data = axis_profile_data(
        _profile_summary(), "bias", axis="effect_scale", scenario="C"
    )

    assert draw_axis_profile(data, ax) is ax
    assert plt.get_fignums() == before


def _one_absent_profile():
    return AxisProfileData(
        axis="effect_scale",
        scenario="C",
        metric="bias",
        axis_values=(0.5, 1.0, 2.0),
        methods=("ipw", "lin_regression"),
        values=np.array([[0.02, np.nan, 0.05], [0.01, 0.00, 0.04]]),
        absent=np.array([[False, True, False], [False, False, False]]),
        realised_treated_share=None,
    )


def _rug_lines(ax):
    """The artists drawn in blended (data-x, axes-y) coordinates.

    `Axes.add_line` sets a plain line's transform to `ax.transData` itself,
    so identity separates the rug from the profile lines. If that ever
    stopped holding, `test_a_method_with_no_absent_values_draws_no_rug`
    fails first -- it asserts this returns an empty list when nothing is
    absent -- so the helper cannot silently start matching everything.
    """
    return [
        line
        for line in ax.lines
        if line.get_transform() is not ax.transData
    ]


def test_an_absent_value_gets_a_rug_mark_at_its_own_axis_position(ax):
    draw_axis_profile(_one_absent_profile(), ax)

    rugs = _rug_lines(ax)
    assert len(rugs) == 1, "only the one method with an absent value gets a rug"
    np.testing.assert_allclose(rugs[0].get_xdata(), [1.0])


def test_the_rug_is_drawn_off_the_value_scale_not_at_a_metric_value(ax):
    # The whole point. A marker at a y-position on the metric scale would be
    # read as a measurement; this one is positioned in axes fractions, so it
    # cannot be. x stays in data coordinates so it still lines up with the
    # axis value it belongs to.
    draw_axis_profile(_one_absent_profile(), ax)

    rug = _rug_lines(ax)[0]
    y = np.asarray(rug.get_ydata(), dtype=float)
    assert np.all((y >= 0.0) & (y <= 1.0)), "rug y must be an axes fraction"

    transform = rug.get_transform()
    # x through the data transform, y through the axes transform: a point at
    # data-x 1.0 and fraction-y 0.02 must land where the x-axis says 1.0 is,
    # and 2% up from the bottom spine.
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(-100.0, 100.0)
    px, py = transform.transform((1.0, 0.02))
    data_px, _ = ax.transData.transform((1.0, 0.0))
    _, axes_py = ax.transAxes.transform((0.0, 0.02))
    assert px == pytest.approx(data_px)
    assert py == pytest.approx(axes_py)


def test_a_method_with_no_absent_values_draws_no_rug(ax):
    data = axis_profile_data(
        _profile_summary(), "bias", axis="effect_scale", scenario="C"
    )

    draw_axis_profile(data, ax)

    assert _rug_lines(ax) == []


def test_each_methods_rug_matches_its_own_line_colour(ax):
    # A rug mark has to be attributable to a method without a second legend.
    data = AxisProfileData(
        axis="effect_scale",
        scenario="C",
        metric="bias",
        axis_values=(0.5, 1.0, 2.0),
        methods=("ipw", "lin_regression"),
        values=np.array([[0.02, np.nan, 0.05], [np.nan, 0.00, 0.04]]),
        absent=np.array([[False, True, False], [True, False, False]]),
        realised_treated_share=None,
    )

    draw_axis_profile(data, ax)

    lines = {line.get_label(): line for line in ax.lines if line.get_label() in data.methods}
    rugs = _rug_lines(ax)
    assert len(rugs) == 2
    rug_colours = {rug.get_color() for rug in rugs}
    assert rug_colours == {
        lines["ipw"].get_color(),
        lines["lin_regression"].get_color(),
    }


def test_the_rug_band_stays_the_same_height_whatever_the_method_count(ax):
    # Ten methods must not turn the rug into a third of the figure.
    from cp_figures.benchmark import _RUG_BAND, _rug_positions

    for n_methods in (1, 2, 10):
        positions = _rug_positions(n_methods)
        assert len(positions) == n_methods
        assert positions.min() >= _RUG_BAND[0]
        assert positions.max() <= _RUG_BAND[1]
    # Distinct rows, so two methods absent at the same axis value do not
    # draw one marker on top of another.
    assert len(set(_rug_positions(10).tolist())) == 10


def test_the_rug_does_not_add_a_legend_entry(ax):
    draw_axis_profile(_one_absent_profile(), ax)

    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ["ipw", "lin_regression"]
