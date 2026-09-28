"""Tests for `cp_report.benchmark`'s data layer: the four section dataclasses and
`benchmark_evidence_from_run` (design doc `2026-09-23-benchmark-study-report-design.md` §5, §7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from cp_core.language import causal_language_lint
from cp_evaluation.store import IncompleteRunError, RunStore
from cp_report.benchmark import (
    BIAS_TABLE,
    COVERAGE_COLUMN_NOTE,
    COVERAGE_TABLE,
    ExpectationRow,
    HeadlineFindingsData,
    OverviewData,
    benchmark_evidence_from_run,
    headline_findings_data,
    limitations_data,
    method_comparisons_data,
    overview_data,
)

from report.benchmark_builders import result, tables


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
