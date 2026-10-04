"""The study bundle: the benchmark evidence as one JSON document.

It is the dashboard's input, so its shape is pinned here key for key, and
the properties the explorer relies on are each tested: a missing number is
`null`, two renderings are identical, and the bundle names the report it was
written with.
"""

import dataclasses
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from cp_report.benchmark import write_markdown_benchmark_report
from cp_report.bundle import (
    SCHEMA_VERSION,
    RunProvenance,
    benchmark_bundle,
    render_benchmark_bundle,
    write_benchmark_bundle,
)

REPORT = b"# a report\n"

TOP_LEVEL = {
    "schema_version", "report_sha256", "provenance", "overview", "expectations",
    "estimators", "scenarios", "levels", "family_metrics", "metric_definitions", "families",
    "report_tables", "cate", "p7", "limitations",
}  # fmt: skip


def _no_constants(token):
    raise AssertionError(f"the bundle holds the non-JSON token {token}")


def test_the_top_level_keys_are_exactly_these(benchmark_evidence):
    bundle = benchmark_bundle(benchmark_evidence, REPORT)

    assert set(bundle) == TOP_LEVEL
    assert bundle["schema_version"] == SCHEMA_VERSION == 1


def test_each_section_has_exactly_the_keys_the_explorer_reads(benchmark_evidence):
    evidence = dataclasses.replace(
        benchmark_evidence, scenario_shapes={"A": "cross_section", "P1": "panel"}
    )
    provenance = (RunProvenance("study", "study", 2, 6, 6, "abc123"),)

    bundle = benchmark_bundle(evidence, REPORT, provenance)

    assert set(bundle["overview"]) == {
        "n_estimators", "n_scenarios", "n_cells", "replications", "coverage_mcse",
    }  # fmt: skip
    assert set(bundle["expectations"]) == {"n_met", "n_failed", "n_inconclusive", "rows"}
    assert {frozenset(row) for row in bundle["expectations"]["rows"]} == {
        frozenset({"id", "statement", "verdict", "narrative"})
    }
    assert bundle["provenance"] == [
        {
            "role": "study", "tier": "study", "replications": 2,
            "cells_done": 6, "cells_expected": 6, "git_commit": "abc123",
        }  # fmt: skip
    ]
    assert bundle["estimators"] == [
        {"name": "difference_in_means", "family": "cross_sectional"},
        {"name": "lin_regression", "family": "cross_sectional"},
    ]
    assert bundle["scenarios"] == [
        {"name": "A", "shape": "cross_section"},
        {"name": "P1", "shape": "panel"},
    ]
    assert set(bundle["report_tables"]) == {"tables", "coverage_note", "exploratory_note"}
    assert set(bundle["limitations"]) == {"not_covered", "statements"}
    assert bundle["limitations"]["not_covered"] == [
        {"name": "donor_pool", "reason": "Not yet implemented."}
    ]
    assert bundle["cate"] is None and bundle["p7"] is None


def test_the_headline_tables_are_filed_by_family_then_level(benchmark_evidence):
    bundle = benchmark_bundle(benchmark_evidence, REPORT)

    names = {
        f"{family}__{level}" for family, levels in bundle["families"].items() for level in levels
    }
    assert names == set(benchmark_evidence.tables)
    rows = bundle["families"]["runtime"]["scenario"]
    assert len(rows) == len(benchmark_evidence.tables["runtime__scenario"])
    assert {"method", "scenario", "runtime_seconds"} <= set(rows[0])


def test_a_table_not_named_family_and_level_is_refused(benchmark_evidence):
    tables = {**benchmark_evidence.tables, "per_cell_summary": pd.DataFrame({"a": [1]})}

    with pytest.raises(ValueError, match="per_cell_summary"):
        benchmark_bundle(dataclasses.replace(benchmark_evidence, tables=tables), REPORT)


def test_a_missing_number_is_null_and_never_nan(benchmark_evidence):
    table = benchmark_evidence.tables["runtime__scenario"].copy()
    table["runtime_seconds"] = table["runtime_seconds"].astype(float)
    table.loc[0, "runtime_seconds"] = np.nan
    table["runtime_seconds_n_used"] = np.int64(0)
    tables = {**benchmark_evidence.tables, "runtime__scenario": table}

    text = render_benchmark_bundle(dataclasses.replace(benchmark_evidence, tables=tables), REPORT)

    rows = json.loads(text, parse_constant=_no_constants)["families"]["runtime"]["scenario"]
    assert rows[0]["runtime_seconds"] is None
    assert rows[1]["runtime_seconds"] is not None
    # A numpy integer arrives as a plain JSON integer, not a float and not a string.
    assert rows[0]["runtime_seconds_n_used"] == 0 and type(rows[0]["runtime_seconds_n_used"]) is int


def test_an_infinite_number_is_null_too(benchmark_evidence):
    table = benchmark_evidence.tables["runtime__scenario"].copy()
    table["runtime_seconds"] = table["runtime_seconds"].astype(float)
    table.loc[0, "runtime_seconds"] = np.inf
    tables = {**benchmark_evidence.tables, "runtime__scenario": table}

    text = render_benchmark_bundle(dataclasses.replace(benchmark_evidence, tables=tables), REPORT)

    rows = json.loads(text, parse_constant=_no_constants)["families"]["runtime"]["scenario"]
    assert rows[0]["runtime_seconds"] is None


def test_a_cell_the_bundle_cannot_represent_is_an_error_not_a_string(benchmark_evidence):
    table = benchmark_evidence.tables["runtime__scenario"].copy()
    table["when"] = pd.Timestamp("2026-10-04")
    tables = {**benchmark_evidence.tables, "runtime__scenario": table}

    with pytest.raises(TypeError, match="Timestamp"):
        benchmark_bundle(dataclasses.replace(benchmark_evidence, tables=tables), REPORT)


def test_two_renderings_are_byte_identical_and_end_in_a_newline(benchmark_evidence):
    first = render_benchmark_bundle(benchmark_evidence, REPORT)
    second = render_benchmark_bundle(benchmark_evidence, REPORT)

    assert first == second
    assert first.endswith("}\n")
    assert list(json.loads(first)) == sorted(json.loads(first))


def test_the_bundle_names_the_report_it_was_written_with(benchmark_evidence, tmp_path):
    report = write_markdown_benchmark_report(benchmark_evidence, tmp_path)

    path = write_benchmark_bundle(benchmark_evidence, tmp_path / "bundle.json", report)

    recorded = json.loads(path.read_text())["report_sha256"]
    assert recorded == hashlib.sha256(report.read_bytes()).hexdigest()
    # And it follows the report: a different report gives a different hash.
    assert recorded != benchmark_bundle(benchmark_evidence, REPORT)["report_sha256"]


def test_the_bundle_is_not_written_into_a_directory_that_does_not_exist(
    benchmark_evidence, tmp_path
):
    report = write_markdown_benchmark_report(benchmark_evidence, tmp_path)

    with pytest.raises(FileNotFoundError):
        write_benchmark_bundle(benchmark_evidence, tmp_path / "missing" / "bundle.json", report)


def test_every_headline_metric_has_a_definition_and_no_other_name_does(benchmark_evidence):
    bundle = benchmark_bundle(benchmark_evidence, REPORT)

    metrics = {metric for names in bundle["family_metrics"].values() for metric in names}
    assert set(bundle["metric_definitions"]) == metrics
    assert all(text.endswith(".") for text in bundle["metric_definitions"].values())
