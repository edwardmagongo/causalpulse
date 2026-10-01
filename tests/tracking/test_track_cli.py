import builtins
import shutil
import sys

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


def test_a_store_error_is_one_line_not_a_traceback(tiny_run, tmp_path, capsys):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    uri = f"sqlite:///{blocker / 'mlflow.db'}"

    assert main(["bench", "track", str(tiny_run), "--tracking-uri", uri]) == 1

    err = capsys.readouterr().err
    assert err.startswith("could not log to ")
    assert "Traceback" not in err


@pytest.mark.parametrize("missing", ["cp_tracking.log", "mlflow"])
def test_without_the_tracking_extra_track_says_how_to_install_it(
    missing, tmp_path, monkeypatch, capsys
):
    # The api/worker image installs cp-cli without its `tracking` extra. A None
    # entry makes that module's import fail as if it were not installed.
    # Dropping cp_tracking.log forces it to re-import mlflow, and dropping
    # mlflow's cached submodules stops `from mlflow.entities import` finding
    # one an earlier test in this worker already loaded.
    for name in [m for m in sys.modules if m == "cp_tracking.log" or m.startswith("mlflow.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, missing, None)

    assert main(["bench", "track", str(tmp_path)]) == 1

    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "cp-cli[tracking]" in err


def test_a_missing_module_other_than_the_extra_keeps_its_traceback(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name.startswith("cp_tracking"):
            raise ModuleNotFoundError("No module named 'cp_tracking_dep'", name="cp_tracking_dep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)

    with pytest.raises(ModuleNotFoundError, match="cp_tracking_dep"):
        main(["bench", "track", str(tmp_path)])
