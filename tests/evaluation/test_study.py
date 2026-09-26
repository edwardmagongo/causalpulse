import numpy as np
import pandas as pd
import pytest
from cp_evaluation.metrics import summarise
from cp_evaluation.store import RunStatus
from cp_evaluation.study import (
    NOT_COVERED,
    baseline_slice,
    coverage_of_study_markdown,
    headline_tables,
    markdown_table,
)

from evaluation._study_frames import frame, rows

COMPLETE = RunStatus(expected=10, done=10, orphans=0, by_scenario={})


# -- pooling ---------------------------------------------------------------


def test_summarise_by_pools_replications_across_the_groups_it_drops():
    # Two sample sizes, 50 rows each: coverage 0.90 at one and 1.00 at the other.
    data = frame(
        rows("aipw", "A", 50, hit_fraction=0.9, n_units=500),
        rows("aipw", "A", 50, hit_fraction=1.0, n_units=2000),
    )

    pooled = summarise(data, by=["scenario"])

    assert len(pooled) == 1
    assert pooled["coverage_analytic"].iloc[0] == pytest.approx(0.95)
    assert pooled["coverage_analytic_n_used"].iloc[0] == 100


def test_a_pooled_mcse_is_not_the_average_of_the_per_cell_mcses():
    # The reason roll-ups start from rows. Coverage 0.5 over 20 rows and 1.0
    # over 80: the pooled share is 0.9, and its MCSE sqrt(.9*.1/100) = 0.03 is
    # neither cell's MCSE nor their mean.
    data = frame(
        rows("aipw", "A", 20, hit_fraction=0.5, n_units=500),
        rows("aipw", "A", 80, hit_fraction=1.0, n_units=2000),
    )

    pooled = summarise(data, by=["scenario"]).iloc[0]
    per_cell = summarise(data)

    assert pooled["coverage_analytic"] == pytest.approx(0.9)
    assert pooled["coverage_analytic_mcse"] == pytest.approx(np.sqrt(0.9 * 0.1 / 100))
    assert pooled["coverage_analytic"] != pytest.approx(per_cell["coverage_analytic"].mean())


def test_a_unanimous_pooled_share_gets_the_rule_of_three_not_a_zero_mcse():
    data = frame(rows("aipw", "A", 60, hit_fraction=1.0))

    pooled = summarise(data, by=["scenario"]).iloc[0]

    assert pooled["coverage_analytic"] == 1.0
    assert pooled["coverage_analytic_mcse"] == pytest.approx(3 / 60)


def test_grouping_by_a_column_the_frame_lacks_is_refused():
    with pytest.raises(KeyError, match="no_such_column"):
        summarise(frame(rows("aipw", "A", 5)), by=["no_such_column"])


def test_by_none_is_the_cell_level_summary_it_always_was():
    data = frame(rows("aipw", "A", 5), rows("aipw", "A", 5, n_units=2000))

    assert len(summarise(data)) == 2
    assert len(summarise(data, by=None)) == 2


# -- the baseline slice ------------------------------------------------------


def test_the_baseline_slice_drops_cells_that_move_an_axis_and_keeps_panel_rows():
    data = frame(
        rows("aipw", "A", 5),
        rows("aipw", "A", 5, axes={"effect_scale": 2.0}),
        rows("aipw", "D", 5, axes={"confounding_strength": 0.5}),
        rows("twfe", "P1", 5, panel=True),
    )

    kept = baseline_slice(data)

    assert sorted(kept["scenario"].unique()) == ["A", "P1"]
    assert len(kept) == 10


def test_the_baseline_slice_keeps_every_sample_size():
    data = frame(rows("aipw", "A", 5, n_units=500), rows("aipw", "A", 5, n_units=2000))

    assert sorted(baseline_slice(data)["n_units"].unique()) == [500, 2000]


# -- the tables ----------------------------------------------------------------


def test_every_table_pools_the_baseline_slice_only():
    data = frame(
        rows("aipw", "A", 50, bias=0.0),
        # A one-at-a-time cell with a huge bias: it must not reach a headline table.
        rows("aipw", "A", 50, bias=5.0, axes={"noise_scale": 1.3}),
    )

    tables = headline_tables(data)

    assert tables["accuracy__scenario"]["bias"].iloc[0] == pytest.approx(0.0, abs=0.02)
    assert tables["accuracy__scenario"]["bias_n_used"].iloc[0] == 50


def test_coverage_is_reported_separately_for_the_two_interval_types():
    # Analytic intervals cover on rows 0-44. Overwriting rows 0-9's bootstrap
    # intervals leaves the bootstrap covering on rows 10-44: 35 of 50.
    data = frame(rows("aipw", "A", 50, hit_fraction=0.9))
    data.loc[:9, ["bootstrap_ci_lower", "bootstrap_ci_upper"]] = 100.0

    table = headline_tables(data)["coverage__scenario"].iloc[0]

    assert table["coverage_analytic"] == pytest.approx(0.9)
    assert table["coverage_bootstrap"] == pytest.approx(0.7)
    assert "coverage_analytic_mcse" in table.index and "coverage_bootstrap_mcse" in table.index


def test_the_baseline_slice_keeps_donor_pool_rows():
    # A donor-pool row carries no cross-sectional axes (they are NaN), the
    # same reason a panel row passes.
    data = frame(rows("synthetic_control", "S1", 5, donor_pool=True))

    kept = baseline_slice(data)

    assert len(kept) == 5
    assert kept["scenario"].unique().tolist() == ["S1"]


def test_not_covered_no_longer_names_synthetic_control_or_bsts():
    names = {name for name, _ in NOT_COVERED}
    assert "synthetic_control" not in names
    assert "bsts" not in names


def test_the_coverage_table_reports_permutation_coverage_for_donor_pool_rows_only():
    donor = rows("synthetic_control", "S1", 20, donor_pool=True, hit_fraction=0.75)
    donor["ci_basis"] = "permutation"
    data = frame(donor, rows("aipw", "A", 20))

    table = headline_tables(data)["coverage__scenario"].set_index("method")

    assert table.loc["synthetic_control", "coverage_permutation"] == pytest.approx(0.75)
    assert table.loc["synthetic_control", "coverage_permutation_n_used"] == 20
    assert "coverage_permutation_mcse" in table.columns
    assert np.isnan(table.loc["synthetic_control", "coverage_posterior"])
    assert table.loc["synthetic_control", "coverage_posterior_n_used"] == 0
    assert np.isnan(table.loc["aipw", "coverage_permutation"])
    assert table.loc["aipw", "coverage_permutation_n_used"] == 0
    assert np.isnan(table.loc["aipw", "coverage_posterior"])


def test_the_false_positive_table_holds_only_null_scenarios():
    data = frame(
        rows("aipw", "A", 100, truth=1.0),
        rows("aipw", "B", 100, truth=0.0, hit_fraction=0.95),
    )

    table = headline_tables(data)["false_positive_rate__scenario"]

    assert list(table["scenario"]) == ["B"]
    assert table["false_positive_rate"].iloc[0] == pytest.approx(0.05)


def test_status_counts_travel_with_every_table():
    data = frame(rows("aipw", "A", 10), rows("aipw", "A", 3, status="refused"))

    row = headline_tables(data)["accuracy__scenario"].iloc[0]

    assert (row["n_fitted"], row["n_refused"], row["n_ineligible"]) == (10, 3, 0)


def test_a_table_by_size_has_one_row_per_sample_size():
    data = frame(rows("aipw", "A", 20, n_units=500), rows("aipw", "A", 20, n_units=2000))

    table = headline_tables(data)["coverage__scenario_by_size"]

    assert sorted(table["n_units"]) == [500, 2000]


def test_markdown_tables_render_headers_rows_and_blank_missing_values():
    text = markdown_table(pd.DataFrame({"a": [1.5, float("nan")], "b": ["x", "y"]}))

    assert text.splitlines()[0] == "| a | b |"
    assert text.splitlines()[2] == "| 1.5 | x |"
    assert text.splitlines()[3] == "|  | y |"


# -- coverage of the study ---------------------------------------------------


def test_the_coverage_document_names_what_was_not_run_and_the_mcse_it_implies():
    data = frame(rows("aipw", "A", 10), rows("aipw", "A", 3, status="refused"))

    text = coverage_of_study_markdown(data, COMPLETE, 50)

    assert "Replications per configuration: 50" in text
    assert "0.031" in text  # sqrt(.95 * .05 / 50)
    for name, _ in NOT_COVERED:
        assert name in text
    assert "aipw" in text and "refused" in text
