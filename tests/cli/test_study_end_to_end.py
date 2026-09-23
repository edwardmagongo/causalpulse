"""The whole study path once, on the estimators the `ci` tier gates on.

A sharded run interrupted by a failing cell, resumed at two workers and extended,
then analysed. Slow because the ci estimators fit real bootstrap intervals; the same
paths run in the fast suite over cheaper estimators.
"""

import json

import cp_cli.commands.bench as bench_module
import pytest
from cp_cli.__main__ import main
from cp_evaluation.execute import CellFailedError, fit_cell_to_shard
from cp_evaluation.grid import CrossSectionAxes, GridConfig, PanelAxes
from cp_evaluation.store import RunStore
from cp_evaluation.tiers import CI_ESTIMATORS, CI_FIT_CONFIG

pytestmark = pytest.mark.slow


def _config(replications: int = 2) -> GridConfig:
    return GridConfig(
        cross_section=CrossSectionAxes(("A", "B"), (500,), (1.0,)),
        panel=PanelAxes(("P1",), (150,), (6,)),
        replications=replications,
    )


def test_a_study_run_survives_an_interruption_extends_and_is_analysed(tmp_path, monkeypatch):
    monkeypatch.setitem(
        bench_module.TIERS, "study", (lambda: _config(2), CI_ESTIMATORS, CI_FIT_CONFIG)
    )
    run, out = tmp_path / "run", tmp_path / "out"
    real_execute = bench_module.execute
    calls = {"n": 0}

    def fails_on_the_third(task):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ZeroDivisionError("injected")
        return fit_cell_to_shard(task)

    # First attempt, at workers=1: serial so the order is fixed (the third cell
    # raises), and because fails_on_the_third is a closure, which execute()'s
    # picklability guard refuses outright at workers>1 -- correctly, since a
    # closure can never reach a worker process, but that refusal is not what
    # this attempt means to exercise.
    with monkeypatch.context() as patched:
        patched.setattr(
            bench_module,
            "execute",
            lambda *args, **kwargs: real_execute(*args, _task=fails_on_the_third, **kwargs),
        )
        with pytest.raises(CellFailedError):
            main(["bench", "run", "--tier", "study", "--out", str(run), "--workers", "1"])
    assert RunStore.open(run).status().done == 2

    # Resume with the real task at two workers, and extend to three replications.
    argv = ["bench", "run", "--tier", "study", "--out", str(run), "--resume"]
    assert main([*argv, "--workers", "2", "--replications", "3"]) == 0
    status = RunStore.open(run).status()
    assert (status.expected, status.done, status.orphans) == (9, 9, 0)

    assert main(["bench", "study", str(run), "--out", str(out)]) == 0
    record = json.loads((out / "study.json").read_text())
    assert record["cells"] == {"expected": 9, "done": 9, "orphans": 0}
    assert {e["id"]: e["verdict"] for e in record["expectations"]}["E8"] == "met"
