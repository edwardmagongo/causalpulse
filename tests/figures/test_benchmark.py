import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from cp_evaluation.metrics import summarise
from cp_evaluation.run import ROW_COLUMNS
from cp_figures.benchmark import HeatmapData, draw_heatmap, heatmap_data


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
    axis value, ~24 columns wide (one identity per `GROUP_COLUMNS`, plus
    every metric family `summarise` computes -- `bias`, `rmse_mcse`,
    `realised_treated_share`, ... -- 23 of the 24). A fixture built by hand
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
