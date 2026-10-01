"""One real, freshly fitted sharded run over the tiny six-cell study grid per module."""

import copy
import json
import os
from pathlib import Path

import pytest
from cli._tiny_study import use_tiny_study_tier
from cp_cli.__main__ import main
from cp_cli.commands.bench import load_run
from cp_evaluation.store import MANIFEST_NAME


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("tracking") / "tiny-run"
    with pytest.MonkeyPatch.context() as monkeypatch:
        use_tiny_study_tier(monkeypatch)
        assert main(["bench", "run", "--tier", "study", "--out", str(run_dir)]) == 0
    return run_dir


@pytest.fixture(scope="module")
def _loaded_once(tiny_run):
    frame, sidecar = load_run(tiny_run)
    manifest = json.loads((tiny_run / MANIFEST_NAME).read_text())
    return frame, manifest, sidecar, tiny_run


@pytest.fixture
def loaded(_loaded_once):
    """`(frame, manifest, sidecar, run_dir)` as `bench track` reads them; private copies."""
    frame, manifest, sidecar, run_dir = _loaded_once
    return frame.copy(), copy.deepcopy(manifest), copy.deepcopy(sidecar), run_dir


@pytest.fixture(autouse=True)
def _keep_mlflow_out_of_the_working_directory(tmp_path, monkeypatch):
    """MLflow defaults to `./mlruns` and `./mlartifacts`. Run every test in its own
    tmp dir so a test that forgets an explicit location leaks there, and fail if
    the directory the suite was started from gained either one."""
    start = Path(os.getcwd())
    before = {name for name in ("mlruns", "mlartifacts") if (start / name).exists()}
    monkeypatch.chdir(tmp_path)
    yield
    after = {name for name in ("mlruns", "mlartifacts") if (start / name).exists()}
    assert after == before, f"a tracking test wrote {sorted(after - before)} into {start}"
