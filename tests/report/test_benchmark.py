"""Tests for `cp_report.benchmark`'s data layer: the four section dataclasses and
`benchmark_evidence_from_run` (design doc `2026-09-23-benchmark-study-report-design.md` §5, §7)."""

from __future__ import annotations

import pytest
from cp_core.language import causal_language_lint
from cp_evaluation.store import IncompleteRunError, RunStore
from cp_report.benchmark import (
    COVERAGE_COLUMN_NOTE,
    REPORT_TABLES,
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


def test_method_comparisons_data_embeds_only_the_curated_tables():
    data = method_comparisons_data(tables(), "an exploratory note")

    assert set(data.tables) == set(REPORT_TABLES)
    assert "runtime__scenario" not in data.tables


def test_method_comparisons_data_skips_a_curated_table_missing_from_the_input():
    only_accuracy = {"accuracy__scenario": tables()["accuracy__scenario"]}

    data = method_comparisons_data(only_accuracy, "note")

    assert set(data.tables) == {"accuracy__scenario"}


def test_the_coverage_note_travels_with_a_coverage_table_and_only_then():
    with_coverage = method_comparisons_data(tables(), "note")
    accuracy_only = {"accuracy__scenario": tables()["accuracy__scenario"]}
    without = method_comparisons_data(accuracy_only, "note")

    assert with_coverage.coverage_note == COVERAGE_COLUMN_NOTE
    assert with_coverage.prose_fragments() == ("note", COVERAGE_COLUMN_NOTE)
    assert without.coverage_note == ""
    assert without.prose_fragments() == ("note",)


def test_the_coverage_note_names_both_non_analytic_estimators_and_passes_the_lint():
    assert "hierarchical_did" in COVERAGE_COLUMN_NOTE
    assert "callaway_santanna" in COVERAGE_COLUMN_NOTE
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
    # One heatmap per REPORT_HEATMAP_METRICS, one profile per REPORT_PROFILE_SELECTIONS.
    assert len(evidence.figures) == 3


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
