import shutil

import pytest
from cp_cli.__main__ import main


def test_track_logs_a_run_once_and_refuses_a_repeat(tiny_run, tmp_path, capsys):
    uri = f"sqlite:///{tmp_path / 'store' / 'mlflow.db'}"

    assert main(["bench", "track", str(tiny_run), "--tracking-uri", uri]) == 0
    run_id = capsys.readouterr().out.strip()

    from mlflow.tracking import MlflowClient

    run = MlflowClient(tracking_uri=uri).get_run(run_id)
    assert run.info.run_name == tiny_run.name
    assert any(key.startswith("A.") for key in run.data.metrics)

    assert main(["bench", "track", str(tiny_run), "--tracking-uri", uri]) == 1
    assert f"already tracked as MLflow run {run_id}" in capsys.readouterr().err

    assert main(["bench", "track", str(tiny_run), "--tracking-uri", uri, "--force"]) == 0


def test_tracking_uri_defaults_to_the_environment_then_mlruns(tiny_run, tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'env-store' / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    assert main(["bench", "track", str(tiny_run)]) == 0
    assert (tmp_path / "env-store" / "mlflow.db").exists()

    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    monkeypatch.chdir(tmp_path)
    assert main(["bench", "track", str(tiny_run), "--force"]) == 0
    assert (tmp_path / "mlruns" / "mlflow.db").exists()


def test_a_run_without_a_manifest_is_refused(tmp_path, capsys):
    legacy = tmp_path / "legacy"
    legacy.mkdir()

    assert main(["bench", "track", str(legacy)]) == 1
    assert "has no manifest.json" in capsys.readouterr().err


def test_an_incomplete_run_is_refused_by_load_run(tiny_run, tmp_path):
    from cp_evaluation.store import IncompleteRunError, RunStore

    # A private copy: the module-scoped `tiny_run` is shared with every other test.
    broken = tmp_path / "broken"
    shutil.copytree(tiny_run, broken)
    # The same way tests/cli/test_bench_study.py interrupts a run: one cell's shard gone.
    victim = sorted(RunStore.open(broken).expected_cells())[0]
    (broken / "cells" / f"{victim}.parquet").unlink()

    with pytest.raises(IncompleteRunError):
        main(
            [
                "bench",
                "track",
                str(broken),
                "--tracking-uri",
                f"sqlite:///{tmp_path / 's' / 'mlflow.db'}",
            ]
        )
