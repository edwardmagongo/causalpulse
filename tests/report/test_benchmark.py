"""Tests for `cp_report.benchmark`'s data layer: the five section dataclasses
(overview, findings, comparisons, limitations, CATE) and
`benchmark_evidence_from_run`."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from cp_core.language import causal_language_lint
from cp_evaluation.run import ROW_COLUMNS
from cp_evaluation.store import IncompleteRunError, RunStatus, RunStore
from cp_report.benchmark import (
    BIAS_TABLE,
    CATE_NOTE,
    COVERAGE_COLUMN_NOTE,
    COVERAGE_TABLE,
    P7_NOTE,
    CateSectionData,
    ExpectationRow,
    HeadlineFindingsData,
    OverviewData,
    P7SectionData,
    benchmark_evidence_from_run,
    cate_section_data,
    headline_findings_data,
    limitations_data,
    method_comparisons_data,
    overview_data,
    p7_section_data,
    write_markdown_benchmark_report,
)
from cp_report.language import ReportLanguageError, require_clean_prose

from report.benchmark_builders import result, tables, tiny_grid_with_scenario_d


def test_overview_data_reports_the_grid_shape_and_mcse():
    data = overview_data(n_estimators=4, n_scenarios=6, n_cells=4000, replications=200)

    assert data == OverviewData(
        n_estimators=4,
        n_scenarios=6,
        n_cells=4000,
        replications=200,
        coverage_mcse=data.coverage_mcse,
    )
    assert data.coverage_mcse == pytest.approx(0.015411, abs=1e-6)  # sqrt(.95*.05/200)


def test_overview_data_has_no_prose_fragments():
    assert overview_data(1, 1, 1, 10).prose_fragments() == ()


def test_headline_findings_data_tallies_verdicts_and_keeps_order():
    results = (
        result("E1", verdict="met"),
        result("E2", verdict="failed"),
        result("E3", verdict="inconclusive"),
    )

    data = headline_findings_data(results, narratives={})

    assert (data.n_met, data.n_failed, data.n_inconclusive) == (1, 1, 1)
    assert [row.id for row in data.rows] == ["E1", "E2", "E3"]


def test_headline_narrative_prefers_caller_override_then_note_then_statement():
    results = (
        result("E1", statement="stmt one", note=""),
        result("E2", statement="stmt two", note="a friendlier note"),
        result("E3", statement="plain stmt", note=""),
    )

    data = headline_findings_data(results, narratives={"E1": "caller override"})

    assert {row.id: row.narrative for row in data.rows} == {
        "E1": "caller override",
        "E2": "a friendlier note",
        "E3": "plain stmt",
    }


def test_headline_prose_fragments_repeat_nothing_when_narrative_equals_statement():
    same = ExpectationRow(id="E1", statement="same text", verdict="met", narrative="same text")
    different = ExpectationRow(id="E2", statement="stmt", verdict="failed", narrative="narrative")

    fragments = HeadlineFindingsData(
        rows=(same, different), n_met=1, n_failed=1, n_inconclusive=0
    ).prose_fragments()

    assert fragments == ("same text", "stmt", "narrative")


def test_method_comparisons_data_curates_a_bias_table_and_a_coverage_table():
    data = method_comparisons_data(tables(), "an exploratory note")

    assert list(data.tables) == [BIAS_TABLE, COVERAGE_TABLE]  # runtime__scenario is not curated in
    bias = data.tables[BIAS_TABLE]
    assert list(bias.columns) == ["method", "A", "B"]
    assert bias.set_index("method").loc["lin_regression", "B"] == 0.04


def _coverage_by_basis() -> pd.DataFrame:
    nan = float("nan")
    return pd.DataFrame(
        {
            "method": ["ipw", "ipw", "hierarchical_did", "hierarchical_did"]
            + ["callaway_santanna", "callaway_santanna", "aipw"],
            "scenario": ["A", "B", "P1", "P2", "P1", "P2", "H"],
            "coverage_analytic": [0.95, 0.94, nan, nan, nan, nan, nan],
            "coverage_posterior": [nan, nan, 0.97, 0.93, nan, nan, nan],
            "coverage_bootstrap_se": [nan, nan, nan, nan, 0.89, 0.91, nan],
            "coverage_permutation": [nan] * 7,
        }
    )


def test_the_coverage_table_shows_each_estimators_primary_interval_and_its_kind():
    data = method_comparisons_data({"coverage__scenario": _coverage_by_basis()}, "note")

    table = data.tables[COVERAGE_TABLE].set_index("method")
    assert list(data.tables[COVERAGE_TABLE].columns) == [
        "method",
        "interval",
        "A",
        "B",
        "H",
        "P1",
        "P2",
    ]
    assert table.loc["ipw", "interval"] == "analytic" and table.loc["ipw", "B"] == 0.94
    assert table.loc["hierarchical_did", "interval"] == "posterior"
    assert table.loc["hierarchical_did", "P1"] == 0.97
    assert table.loc["callaway_santanna", "interval"] == "bootstrap_se"
    assert table.loc["callaway_santanna", "P2"] == 0.91
    assert pd.isna(table.loc["hierarchical_did", "A"])  # not run there
    assert table.loc["aipw", "interval"] == ""  # no interval anywhere: nothing fitted


def test_method_comparisons_data_skips_a_source_table_missing_from_the_input():
    only_accuracy = {"accuracy__scenario": tables()["accuracy__scenario"]}

    data = method_comparisons_data(only_accuracy, "note")

    assert list(data.tables) == [BIAS_TABLE]


def test_the_coverage_note_travels_with_the_coverage_table_and_only_then():
    with_coverage = method_comparisons_data(tables(), "note")
    accuracy_only = {"accuracy__scenario": tables()["accuracy__scenario"]}
    without = method_comparisons_data(accuracy_only, "note")

    assert with_coverage.coverage_note == COVERAGE_COLUMN_NOTE
    assert with_coverage.prose_fragments() == ("note", COVERAGE_COLUMN_NOTE)
    assert without.coverage_note == ""
    assert without.prose_fragments() == ("note",)


def test_the_coverage_note_names_each_non_analytic_estimator_and_its_interval_kind():
    for phrase in (
        "hierarchical_did",
        "posterior",
        "callaway_santanna",
        "bootstrap_se",
        "synthetic_control",
        "permutation",
        "analytic",
    ):
        assert phrase in COVERAGE_COLUMN_NOTE
    assert causal_language_lint(COVERAGE_COLUMN_NOTE) == ()


def test_limitations_prose_is_every_reason_then_every_statement_but_no_names():
    data = limitations_data(
        not_covered=(("donor_pool", "Not yet implemented."),),
        statements=("hierarchical_did is excluded from scenario P6.",),
    )

    assert data.prose_fragments() == (
        "Not yet implemented.",
        "hierarchical_did is excluded from scenario P6.",
    )


def test_benchmark_evidence_from_run_reads_not_covered_per_frame(d_run):
    # The tiny D+P1 grid has no donor-pool cells, so `not_covered_for` reduces to exactly
    # `NOT_COVERED` -- proves the switch away from the bare constant did not silently change
    # what a donor-free run reports.
    from cp_evaluation.study import NOT_COVERED

    evidence = benchmark_evidence_from_run(d_run)

    assert evidence.not_covered == NOT_COVERED


def test_benchmark_evidence_from_run_drops_a_donor_estimator_once_it_has_a_fitted_row(monkeypatch):
    # A synthetic frame stands in for a real donor-pool run: real fitting takes hours and this
    # is `not_covered_for`'s own job to test in cp_evaluation (it already is,
    # tests/evaluation/test_study.py) -- this test only proves benchmark_evidence_from_run
    # calls it instead of returning the bare constant, on a frame it does not have to fit.
    from cp_evaluation.store import RunStatus

    class _FakeStore:
        replications = 1

        def __init__(self, frame):
            self._frame = frame

        def read_frame(self):
            return self._frame

        def status(self):
            return RunStatus(expected=1, done=1, orphans=0, by_scenario={})

    frame = pd.DataFrame(
        {
            "method": ["synthetic_control"],
            "scenario": ["S0"],
            "shape": ["donor_pool"],
            "status": ["fitted"],
        }
    )
    monkeypatch.setattr(
        "cp_report.benchmark.RunStore",
        type("_Open", (), {"open": staticmethod(lambda _directory: _FakeStore(frame))}),
    )
    monkeypatch.setattr("cp_report.benchmark.evaluate_expectations", lambda _frame, _status: ())
    monkeypatch.setattr("cp_report.benchmark.headline_tables", lambda _frame: {})
    monkeypatch.setattr("cp_report.benchmark._report_figures", lambda _frame: ())

    evidence = benchmark_evidence_from_run(Path("unused"))

    assert "synthetic_control" not in dict(evidence.not_covered)
    assert "bsts" in dict(evidence.not_covered)  # still absent from this frame entirely


def test_benchmark_evidence_from_run_builds_from_a_real_tiny_study(d_run):
    evidence = benchmark_evidence_from_run(
        d_run, limitations=("A stand-in limitation for this tiny fixture.",)
    )

    assert evidence.status.expected == evidence.status.done > 0
    assert set(evidence.scenarios_run) == {"D", "P1"}
    assert set(evidence.estimators_run) == {"difference_in_means", "lin_regression"}
    # Each scenario's shape is the one the grid declared it under, not a second groupby of
    # the frame the evidence was built from.
    grid = tiny_grid_with_scenario_d()
    declared = {name: "cross_section" for name in grid.cross_section.scenarios} | {
        name: "panel" for name in grid.panel.scenarios
    }
    assert evidence.scenario_shapes == {name: declared[name] for name in evidence.scenarios_run}
    assert evidence.replications == 2
    assert len(evidence.expectations) > 0
    assert evidence.limitations == ("A stand-in limitation for this tiny fixture.",)
    assert "coverage__scenario_by_size" in evidence.exploratory_note
    # One heatmap per REPORT_HEATMAP_METRICS, one profile per REPORT_PROFILE_SELECTIONS.
    assert [figure.caption for figure in evidence.figures] == [
        "Coverage of each estimator's primary interval, by method and scenario, pooled over "
        "each scenario's baseline slice. Lighter cells have higher coverage; grey cells have "
        "no fitted result.",
        "bias on scenario D, varying confounding_strength one value at a time.",
    ]
    comparisons = method_comparisons_data(evidence.tables, evidence.exploratory_note)
    bias = comparisons.tables[BIAS_TABLE]
    assert set(bias["method"]) == {"difference_in_means", "lin_regression"}
    assert {"D", "P1"} <= set(bias.columns)


def test_the_report_figures_leave_no_estimator_label_clipped(d_run):
    # A default-margin figure clips a long label such as `callaway_santanna`; the report's
    # figures lay themselves out so every tick label sits inside the figure.
    evidence = benchmark_evidence_from_run(d_run)
    fig = plt.figure()
    try:
        evidence.figures[0].draw(fig.subplots())
        fig.canvas.draw()
        inside = fig.bbox
        labels = fig.axes[0].get_yticklabels()
        assert labels
        for label in labels:
            box = label.get_window_extent()
            assert box.x0 >= inside.x0 and box.x1 <= inside.x1
    finally:
        plt.close(fig)


def test_narratives_reach_the_headline_rows(d_run):
    first = benchmark_evidence_from_run(d_run).expectations[0].id

    evidence = benchmark_evidence_from_run(d_run, narratives={first: "A reader-facing paragraph."})

    rows = headline_findings_data(evidence.expectations, evidence.narratives).rows
    assert rows[0].narrative == "A reader-facing paragraph."


def test_a_narrative_for_an_unknown_expectation_is_refused_not_ignored(d_run):
    with pytest.raises(ValueError, match="E99"):
        benchmark_evidence_from_run(d_run, narratives={"E99": "a typo'd id"})


def test_benchmark_evidence_from_run_refuses_an_incomplete_run(d_run):
    victim = sorted(RunStore.open(d_run).expected_cells())[0]
    (d_run / "cells" / f"{victim}.parquet").unlink()

    with pytest.raises(IncompleteRunError):  # propagated from RunStore.read_frame
        benchmark_evidence_from_run(d_run)


def _cate_rows(scenario, correlations, *, rmse=0.05, slope=1.2, method="causal_forest"):
    records = []
    for seed, correlation in enumerate(correlations):
        row = dict.fromkeys(ROW_COLUMNS, np.nan)
        row.update(
            shape="cate", scenario=scenario, n_units=2000, seed=seed, method=method,
            status="fitted", reason="", estimand="ATE", truth=0.1, point=0.1,
            runtime_seconds=1.0, cate_correlation=correlation, cate_rmse=rmse,
            cate_calibration_slope=slope,
        )
        records.append(row)
    return pd.DataFrame(records, columns=list(ROW_COLUMNS))


_CATE_STATUS = RunStatus(expected=4, done=4, orphans=0, by_scenario={})


def test_the_cate_section_reports_each_metric_with_its_mcse_and_count():
    rows = _cate_rows("RICH", [0.8, 0.9, 0.7, 0.8])

    data = cate_section_data(rows, _CATE_STATUS, replications=4)

    assert data is not None
    (row,) = data.table.to_dict("records")
    assert row["method"] == "causal_forest"
    assert row["scenario"] == "RICH"
    assert row["cate_correlation"] == pytest.approx(0.8)
    assert row["cate_correlation_mcse"] == pytest.approx(
        float(np.std([0.8, 0.9, 0.7, 0.8], ddof=1) / 2.0)
    )
    assert row["cate_correlation_n_used"] == 4
    assert row["cate_calibration_slope"] == pytest.approx(1.2)
    assert (data.replications, data.cells_done, data.cells_expected) == (4, 4, 4)


def test_null_keeps_its_rmse_and_leaves_correlation_and_slope_blank():
    rows = _cate_rows("NULL", [np.nan, np.nan], slope=np.nan)

    data = cate_section_data(rows, _CATE_STATUS, replications=2)

    assert data is not None
    (row,) = data.table.to_dict("records")
    assert row["cate_rmse_n_used"] == 2
    assert np.isnan(row["cate_correlation"])
    assert np.isnan(row["cate_calibration_slope"])


def test_a_method_with_no_per_unit_cate_is_left_out_of_the_table():
    rows = pd.concat(
        [
            _cate_rows("RICH", [0.8, 0.9]),
            _cate_rows("RICH", [np.nan, np.nan], rmse=np.nan, slope=np.nan,
                       method="difference_in_means"),
        ],
        ignore_index=True,
    )

    data = cate_section_data(rows, _CATE_STATUS, replications=2)

    assert data is not None
    assert list(data.table["method"]) == ["causal_forest"]


def test_no_scored_row_means_no_section():
    unscored = _cate_rows("RICH", [np.nan], rmse=np.nan, slope=np.nan)

    assert cate_section_data(unscored, _CATE_STATUS, replications=1) is None
    assert cate_section_data(unscored.iloc[0:0], _CATE_STATUS, replications=1) is None


def test_the_cate_note_passes_the_language_lint_and_the_lint_still_bites():
    require_clean_prose([("CATE", CATE_NOTE)])
    with pytest.raises(ReportLanguageError):
        require_clean_prose([("CATE", CATE_NOTE), ("CATE", "This proves the forest is right.")])


def test_the_cate_section_offers_its_note_to_the_lint():
    data = cate_section_data(_cate_rows("RICH", [0.8]), _CATE_STATUS, replications=1)

    assert isinstance(data, CateSectionData)
    assert data.prose_fragments() == (CATE_NOTE,)


def test_a_cate_run_adds_a_cate_section_and_clears_not_covered(d_run, cate_run):
    evidence = benchmark_evidence_from_run(d_run, cate_directory=cate_run)

    assert evidence.cate_section is not None
    table = evidence.cate_section.table
    assert set(table["scenario"]) == {"RICH", "NULL"}
    assert set(table["method"]) == {"causal_forest"}
    assert evidence.cate_section.replications == 2
    assert "CATE metrics" not in dict(evidence.not_covered)


def test_without_a_cate_run_there_is_no_section_and_the_entry_stays(d_run):
    evidence = benchmark_evidence_from_run(d_run)

    assert evidence.cate_section is None
    assert "CATE metrics" in dict(evidence.not_covered)


def test_the_cate_run_changes_no_ate_view(d_run, cate_run):
    without = benchmark_evidence_from_run(d_run)
    with_cate = benchmark_evidence_from_run(d_run, cate_directory=cate_run)

    # repr, not ==: Measurement.mcse is NaN for the E8 counts, and NaN != NaN.
    assert [repr(e) for e in with_cate.expectations] == [repr(e) for e in without.expectations]
    assert with_cate.scenarios_run == without.scenarios_run
    assert with_cate.estimators_run == without.estimators_run
    assert with_cate.status == without.status
    assert with_cate.replications == without.replications
    assert [n for n in with_cate.not_covered if n[0] != "CATE metrics"] == [
        n for n in without.not_covered if n[0] != "CATE metrics"
    ]
    assert with_cate.tables.keys() == without.tables.keys()
    for name in without.tables:
        pd.testing.assert_frame_equal(with_cate.tables[name], without.tables[name])
    assert [f.caption for f in with_cate.figures] == [f.caption for f in without.figures]


def test_cate_rows_in_the_main_store_stay_out_of_every_ate_view(d_run, monkeypatch):
    # A future store that carries the arm itself: its CATE rows feed the CATE
    # section and nothing else.
    real = RunStore.open(d_run)
    frame = real.read_frame()
    ate_status = real.status()
    # The mixed store's own ledger: the ATE cells plus two RICH cells.
    status = RunStatus(
        expected=ate_status.expected + 2,
        done=ate_status.done + 2,
        orphans=ate_status.orphans,
        by_scenario={**ate_status.by_scenario, "RICH": (2, 2)},
    )
    combined = pd.concat([frame, _cate_rows("RICH", [0.8, 0.9])], ignore_index=True)

    class _Store:
        replications = real.replications

        def __init__(self, rows):
            self._rows = rows

        def read_frame(self):
            return self._rows

        def status(self):
            return status

    without = benchmark_evidence_from_run(d_run)
    monkeypatch.setattr(
        "cp_report.benchmark.RunStore",
        type("_Open", (), {"open": staticmethod(lambda _d: _Store(combined))}),
    )
    with_rows = benchmark_evidence_from_run(d_run)

    assert with_rows.scenarios_run == without.scenarios_run
    assert with_rows.estimators_run == without.estimators_run
    # E8 checks the whole store's completeness, so it alone sees the two CATE cells.
    assert [repr(e) for e in with_rows.expectations if e.id != "E8"] == [
        repr(e) for e in without.expectations if e.id != "E8"
    ]
    for name in without.tables:
        pd.testing.assert_frame_equal(with_rows.tables[name], without.tables[name])
    assert [f.caption for f in with_rows.figures] == [f.caption for f in without.figures]
    # The overview counts ATE cells only; the section counts its own.
    assert with_rows.status.expected == ate_status.expected
    assert "RICH" not in with_rows.status.by_scenario
    assert with_rows.cate_section is not None
    assert with_rows.cate_section.cells_expected == 2
    assert with_rows.cate_section.cells_done == 2
    assert list(with_rows.cate_section.table["scenario"]) == ["RICH"]


def test_a_cate_run_of_the_wrong_tier_is_refused(d_run):
    with pytest.raises(ValueError, match="'cate'"):
        benchmark_evidence_from_run(d_run, cate_directory=d_run)


def test_two_sources_of_cate_rows_are_refused(d_run, cate_run, monkeypatch):
    from cp_evaluation.store import RunStore as _RealRunStore

    real = _RealRunStore.open(d_run)
    combined = pd.concat([real.read_frame(), _cate_rows("RICH", [0.8])], ignore_index=True)

    class _Store:
        def __init__(self, rows, store):
            self._rows, self._store = rows, store
            self.replications = store.replications
            self.identity = store.identity

        def read_frame(self):
            return self._rows

        def status(self):
            return self._store.status()

    def _open(directory):
        store = _RealRunStore.open(directory)
        rows = combined if Path(directory) == Path(d_run) else store.read_frame()
        return _Store(rows, store)

    monkeypatch.setattr(
        "cp_report.benchmark.RunStore", type("_Open", (), {"open": staticmethod(_open)})
    )
    with pytest.raises(ValueError, match="two protocols"):
        benchmark_evidence_from_run(d_run, cate_directory=cate_run)


def _p7_rows(methods=("twfe", "callaway_santanna"), n=3, ci_basis="analytic"):
    records = []
    for method in methods:
        for seed in range(n):
            row = dict.fromkeys(ROW_COLUMNS, np.nan)
            row.update(shape="long_panel", scenario="P7", n_units=600, n_periods=20,
                       seed=seed, method=method, status="fitted", reason="",
                       estimand="ATT", truth=6.7, point=6.7 + 0.1 * (seed - 1),
                       se=0.2, ci_lower=6.0, ci_upper=7.4, ci_basis=ci_basis,
                       runtime_seconds=1.0)
            records.append(row)
    return pd.DataFrame(records, columns=list(ROW_COLUMNS))


_P7_STATUS = RunStatus(expected=3, done=3, orphans=0, by_scenario={"P7": (3, 3)})


def test_the_p7_section_has_bias_rmse_and_primary_coverage_per_estimator():
    data = p7_section_data(_p7_rows(), _P7_STATUS, replications=3)

    assert isinstance(data, P7SectionData)
    table = data.table.set_index("method")
    assert set(table.index) == {"twfe", "callaway_santanna"}
    assert table.loc["twfe", "bias"] == pytest.approx(0.0)
    assert table.loc["twfe", "coverage"] == pytest.approx(1.0)
    assert table.loc["twfe", "interval"] == "analytic"
    for column in ("bias_mcse", "bias_n_used", "rmse", "rmse_mcse", "coverage_mcse",
                   "coverage_n_used"):
        assert column in table.columns
    assert (data.replications, data.cells_done, data.cells_expected) == (3, 3, 3)


def test_no_fitted_p7_row_means_no_section():
    unfitted = _p7_rows().assign(status="ineligible")

    assert p7_section_data(unfitted, _P7_STATUS, replications=3) is None
    assert p7_section_data(unfitted.iloc[0:0], _P7_STATUS, replications=3) is None


def test_the_p7_note_passes_the_lint_and_the_lint_still_bites():
    require_clean_prose([("P7", P7_NOTE)])
    with pytest.raises(ReportLanguageError):
        require_clean_prose([("P7", P7_NOTE), ("P7", "This proves twfe is right.")])


def test_a_p7_run_adds_its_section_and_clears_not_covered(d_run, p7_run):
    evidence = benchmark_evidence_from_run(d_run, p7_directory=p7_run)

    assert evidence.p7_section is not None
    assert set(evidence.p7_section.table["method"]) == {"twfe", "callaway_santanna"}
    assert "scenario P7" not in dict(evidence.not_covered)
    table = evidence.p7_section.table.set_index("method")
    assert table.loc["callaway_santanna", "interval"] == "bootstrap_se"
    assert table.loc["twfe", "interval"] == "analytic"


def test_the_p7_section_is_the_only_rendered_difference(d_run, cate_run, p7_run, tmp_path):
    without = benchmark_evidence_from_run(d_run, cate_directory=cate_run)
    with_p7 = benchmark_evidence_from_run(d_run, cate_directory=cate_run, p7_directory=p7_run)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    text_without = write_markdown_benchmark_report(without, tmp_path / "a").read_text()
    text_with = write_markdown_benchmark_report(with_p7, tmp_path / "b").read_text()

    start = text_with.index("## Scenario P7")
    end = text_with.find("\n## ", start + 1)
    end = len(text_with) if end == -1 else end + 1
    stripped_with = text_with[:start] + text_with[end:]
    stripped_without = "\n".join(
        line for line in text_without.split("\n") if not line.startswith("- scenario P7:")
    )

    assert "## Scenario P7" in text_with
    assert "- scenario P7:" not in text_with
    assert "- scenario P7:" in text_without
    assert stripped_with == stripped_without


def test_the_p7_run_changes_no_ate_view(d_run, p7_run):
    without = benchmark_evidence_from_run(d_run)
    with_p7 = benchmark_evidence_from_run(d_run, p7_directory=p7_run)

    assert [repr(e) for e in with_p7.expectations] == [repr(e) for e in without.expectations]
    assert with_p7.scenarios_run == without.scenarios_run
    assert with_p7.status == without.status
    for name in without.tables:
        pd.testing.assert_frame_equal(with_p7.tables[name], without.tables[name])
    assert [f.caption for f in with_p7.figures] == [f.caption for f in without.figures]
    assert [n for n in with_p7.not_covered if n[0] != "scenario P7"] == [
        n for n in without.not_covered if n[0] != "scenario P7"
    ]


def test_a_p7_run_of_the_wrong_tier_is_refused(d_run):
    with pytest.raises(ValueError, match="'p7'"):
        benchmark_evidence_from_run(d_run, p7_directory=d_run)


def test_long_panel_rows_in_the_main_store_stay_out_of_every_ate_view(d_run, monkeypatch):
    real = RunStore.open(d_run)
    frame = real.read_frame()
    ate_status = real.status()
    status = RunStatus(
        expected=ate_status.expected + 3,
        done=ate_status.done + 3,
        orphans=ate_status.orphans,
        by_scenario={**ate_status.by_scenario, "P7": (3, 3)},
    )
    combined = pd.concat([frame, _p7_rows()], ignore_index=True)

    class _Store:
        replications = real.replications

        def __init__(self, rows):
            self._rows = rows

        def read_frame(self):
            return self._rows

        def status(self):
            return status

    without = benchmark_evidence_from_run(d_run)
    monkeypatch.setattr(
        "cp_report.benchmark.RunStore",
        type("_Open", (), {"open": staticmethod(lambda _d: _Store(combined))}),
    )
    with_rows = benchmark_evidence_from_run(d_run)

    assert with_rows.scenarios_run == without.scenarios_run
    assert with_rows.estimators_run == without.estimators_run
    # E8 checks the whole store's completeness, so it alone sees the three P7 cells.
    assert [repr(e) for e in with_rows.expectations if e.id != "E8"] == [
        repr(e) for e in without.expectations if e.id != "E8"
    ]
    for name in without.tables:
        pd.testing.assert_frame_equal(with_rows.tables[name], without.tables[name])
    assert [f.caption for f in with_rows.figures] == [f.caption for f in without.figures]
    assert with_rows.status.expected == ate_status.expected
    assert "P7" not in with_rows.status.by_scenario
    assert with_rows.p7_section is not None
    assert with_rows.p7_section.cells_expected == 3
    assert with_rows.p7_section.cells_done == 3
    assert set(with_rows.p7_section.table["method"]) == {"twfe", "callaway_santanna"}


def test_two_sources_of_long_panel_rows_are_refused(d_run, p7_run, monkeypatch):
    from cp_evaluation.store import RunStore as _RealRunStore

    real = _RealRunStore.open(d_run)
    combined = pd.concat([real.read_frame(), _p7_rows()], ignore_index=True)

    class _Store:
        def __init__(self, rows, store):
            self._rows, self._store = rows, store
            self.replications = store.replications
            self.identity = store.identity

        def read_frame(self):
            return self._rows

        def status(self):
            return self._store.status()

    def _open(directory):
        store = _RealRunStore.open(directory)
        rows = combined if Path(directory) == Path(d_run) else store.read_frame()
        return _Store(rows, store)

    monkeypatch.setattr(
        "cp_report.benchmark.RunStore", type("_Open", (), {"open": staticmethod(_open)})
    )
    with pytest.raises(ValueError, match="two protocols"):
        benchmark_evidence_from_run(d_run, p7_directory=p7_run)
