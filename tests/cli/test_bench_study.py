import json

import pandas as pd
import pytest
from cp_cli.__main__ import main
from cp_cli.io import read_benchmark
from cp_evaluation.store import MANIFEST_NAME, RunStore

from cli._tiny_study import use_tiny_study_tier


@pytest.fixture(autouse=True)
def tiny_study_tier(monkeypatch):
    use_tiny_study_tier(monkeypatch)


def _run(out, *extra):
    return main(["bench", "run", "--tier", "study", "--out", str(out), *extra])


def _interrupt(tmp_path):
    """A run with one cell's shard deleted, so it is 5 of 6 done."""
    _run(tmp_path / "run")
    store = RunStore.open(tmp_path / "run")
    victim = sorted(store.expected_cells())[0]
    (tmp_path / "run" / "cells" / f"{victim}.parquet").unlink()


def test_status_counts_cells_per_scenario(tmp_path, capsys):
    _run(tmp_path / "run")
    capsys.readouterr()

    assert main(["bench", "status", str(tmp_path / "run")]) == 0

    out = capsys.readouterr().out
    assert "cells: 6/6 done, 0 remaining, 0 orphan shards" in out
    assert "A: 4/4" in out and "P1: 2/2" in out


def test_status_of_a_directory_that_is_not_a_run_names_the_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match=MANIFEST_NAME):
        main(["bench", "status", str(tmp_path)])


def test_compact_writes_a_single_file_run_that_reads_back_as_the_same_rows(tmp_path):
    _run(tmp_path / "run")

    assert main(["bench", "compact", str(tmp_path / "run"), "--out", str(tmp_path / "one")]) == 0

    frame, sidecar = read_benchmark(tmp_path / "one")
    sharded = RunStore.open(tmp_path / "run").read_frame()
    key = ["shape", "scenario", "n_units", "seed", "method"]
    # `read_benchmark` returns the partition column last, and in no fixed row order.
    frame = frame[list(sharded.columns)]
    pd.testing.assert_frame_equal(
        frame.sort_values(key).reset_index(drop=True).drop(columns="runtime_seconds"),
        sharded.sort_values(key).reset_index(drop=True).drop(columns="runtime_seconds"),
        check_dtype=False,
    )
    assert sidecar["tier"] == "study" and sidecar["sharded"] is True


def test_compact_refuses_a_directory_that_is_not_a_sharded_run(tmp_path):
    with pytest.raises(FileNotFoundError, match=MANIFEST_NAME):
        main(["bench", "compact", str(tmp_path), "--out", str(tmp_path / "x")])


def test_study_writes_tables_expectations_coverage_figures_and_a_record(tmp_path, capsys):
    _run(tmp_path / "run")
    capsys.readouterr()

    assert main(["bench", "study", str(tmp_path / "run"), "--out", str(tmp_path / "out")]) == 0

    out = tmp_path / "out"
    assert (out / "tables" / "coverage__scenario.csv").exists()
    assert (out / "tables" / "coverage__scenario.md").read_text().startswith("| method |")
    assert (out / "tables" / "per_cell_summary.csv").exists()
    assert "# Pre-registered expectations" in (out / "expectations.md").read_text()
    assert "## Not covered" in (out / "coverage-of-study.md").read_text()
    assert (out / "figures" / "heatmap_coverage_analytic.png").stat().st_size > 0
    record = json.loads((out / "study.json").read_text())
    assert record["cells"] == {"expected": 6, "done": 6, "orphans": 0}
    assert {e["id"]: e["verdict"] for e in record["expectations"]}["E8"] == "met"
    assert "E8=met" in capsys.readouterr().out


def test_study_refuses_an_incomplete_run_unless_partial(tmp_path):
    _interrupt(tmp_path)

    with pytest.raises(RuntimeError, match="1 of 6"):
        main(["bench", "study", str(tmp_path / "run"), "--out", str(tmp_path / "out")])
    main(["bench", "study", str(tmp_path / "run"), "--out", str(tmp_path / "out"), "--partial"])
    record = json.loads((tmp_path / "out" / "study.json").read_text())
    assert record["partial"] is True
    assert {e["id"]: e["verdict"] for e in record["expectations"]}["E8"] == "failed"


def test_study_records_which_figures_it_refused_and_why(tmp_path):
    _run(tmp_path / "run")

    main(["bench", "study", str(tmp_path / "run"), "--out", str(tmp_path / "out")])

    refused = (tmp_path / "out" / "figures" / "refusals.md").read_text()
    assert refused.startswith("# Figures not drawn")
    # The tiny grid varies no axis, so every profile is refused with a reason.
    assert "A / effect_scale" in refused
