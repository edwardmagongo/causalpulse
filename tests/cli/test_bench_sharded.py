import pandas as pd
import pytest
from cp_cli.__main__ import main
from cp_evaluation.store import RunStore
from evaluation._run_helpers import mixed_config, reference_frame

from cli._tiny_study import use_tiny_study_tier


@pytest.fixture(autouse=True)
def tiny_study_tier(monkeypatch):
    use_tiny_study_tier(monkeypatch)


def _run(out, *extra):
    return main(["bench", "run", "--tier", "study", "--out", str(out), *extra])


def test_a_sharded_run_writes_a_store_and_says_how_far_it_got(tmp_path, capsys):
    assert _run(tmp_path / "run") == 0

    store = RunStore.open(tmp_path / "run")
    assert store.status().remaining == 0
    assert "6/6 cells done" in capsys.readouterr().out


def test_running_into_a_directory_that_holds_a_run_needs_resume(tmp_path):
    _run(tmp_path / "run")

    with pytest.raises(ValueError, match="--resume"):
        _run(tmp_path / "run")


def test_resume_on_a_directory_with_no_run_is_refused(tmp_path):
    with pytest.raises(ValueError, match="holds no run"):
        _run(tmp_path / "run", "--resume")


def test_resuming_a_finished_run_fits_nothing(tmp_path, capsys):
    _run(tmp_path / "run")
    capsys.readouterr()

    _run(tmp_path / "run", "--resume")

    assert "0 fitted now, 6 already done" in capsys.readouterr().out


def test_resume_with_more_replications_extends_the_run(tmp_path, capsys):
    _run(tmp_path / "run")
    capsys.readouterr()

    _run(tmp_path / "run", "--resume", "--replications", "3")

    store = RunStore.open(tmp_path / "run")
    assert store.replications == 3
    assert store.status().expected == 9 and store.status().remaining == 0
    assert "3 fitted now, 6 already done" in capsys.readouterr().out


def test_resume_cannot_lower_the_replications(tmp_path):
    _run(tmp_path / "run")

    with pytest.raises(ValueError, match="finished shards"):
        _run(tmp_path / "run", "--resume", "--replications", "1")


def test_resuming_under_a_different_root_entropy_is_refused_and_names_it(tmp_path):
    _run(tmp_path / "run")

    with pytest.raises(ValueError, match="root_entropy"):
        _run(tmp_path / "run", "--resume", "--root-entropy", "9")


def test_replications_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        _run(tmp_path / "run", "--replications", "0")


@pytest.mark.parametrize("flag", [["--workers", "2"], ["--resume"], ["--replications", "2"]])
def test_the_sharded_only_flags_are_refused_for_a_single_file_tier(tmp_path, flag):
    with pytest.raises(ValueError, match="apply only to sharded tiers"):
        main(["bench", "run", "--tier", "ci", "--out", str(tmp_path / "x"), *flag])


def test_two_workers_write_the_same_store_as_one(tmp_path):
    _run(tmp_path / "one", "--workers", "1")
    _run(tmp_path / "two", "--workers", "2")

    one = RunStore.open(tmp_path / "one").read_frame()
    two = RunStore.open(tmp_path / "two").read_frame()
    pd.testing.assert_frame_equal(
        one.drop(columns="runtime_seconds"), two.drop(columns="runtime_seconds")
    )


def _interrupt(tmp_path):
    """A run with one cell's shard deleted, so it is 5 of 6 done."""
    _run(tmp_path / "run")
    store = RunStore.open(tmp_path / "run")
    victim = sorted(store.expected_cells())[0]
    (tmp_path / "run" / "cells" / f"{victim}.parquet").unlink()


def test_report_reads_a_finished_sharded_run(tmp_path, capsys):
    _run(tmp_path / "run")
    # Without this, the single-file writer would satisfy every assertion below.
    assert (tmp_path / "run" / "manifest.json").exists()
    capsys.readouterr()

    assert main(["bench", "report", str(tmp_path / "run")]) == 0

    out = capsys.readouterr().out
    assert "tier: study" in out and "PARTIAL" not in out


def test_report_refuses_an_incomplete_run_and_says_how_many_cells_are_missing(tmp_path):
    _interrupt(tmp_path)

    with pytest.raises(RuntimeError, match="1 of 6"):
        main(["bench", "report", str(tmp_path / "run")])


def test_report_partial_labels_itself_and_reports_what_exists(tmp_path, capsys):
    _interrupt(tmp_path)
    capsys.readouterr()

    assert main(["bench", "report", str(tmp_path / "run"), "--partial"]) == 0

    assert "PARTIAL: this run is unfinished" in capsys.readouterr().out


def test_the_sharded_frame_is_what_run_grid_gives(tmp_path):
    _run(tmp_path / "run")

    frame = RunStore.open(tmp_path / "run").read_frame()

    pd.testing.assert_frame_equal(
        frame.drop(columns="runtime_seconds"),
        reference_frame(mixed_config(2)).drop(columns="runtime_seconds"),
    )
