"""A real, freshly fitted sharded run over the tiny six-cell study grid."""

import json

import pytest
from cli._tiny_study import use_tiny_study_tier
from cp_cli.__main__ import main
from cp_cli.commands.bench import load_run
from cp_evaluation.store import MANIFEST_NAME


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    use_tiny_study_tier(monkeypatch)
    run_dir = tmp_path / "tiny-run"
    assert main(["bench", "run", "--tier", "study", "--out", str(run_dir)]) == 0
    return run_dir


@pytest.fixture
def loaded(tiny_run):
    """`(frame, manifest, sidecar, run_dir)` exactly as `bench track` will read them."""
    frame, sidecar = load_run(tiny_run)
    manifest = json.loads((tiny_run / MANIFEST_NAME).read_text())
    return frame, manifest, sidecar, tiny_run
