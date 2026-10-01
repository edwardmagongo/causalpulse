import os

import pandas as pd
import pytest
from cp_evaluation.execute import (
    SHAPE_COST_RANK,
    CellFailedError,
    execute,
    fit_cell_to_shard,
    longest_first,
    pool_environment,
)
from cp_evaluation.grid import (
    CateCell,
    CrossSectionCell,
    DonorPoolCell,
    LongPanelCell,
    PanelCell,
    cell_key,
    expand_cells,
)
from cp_evaluation.store import RunStore, write_shard

from evaluation import _tasks
from evaluation._run_helpers import (
    ESTIMATORS,
    FIT_CONFIG,
    mixed_config,
    new_store,
    reference_frame,
    tiny_config,
    without_runtime,
)


def _run(store: RunStore, **kwargs):
    return execute(store, estimators=ESTIMATORS, fit_config=FIT_CONFIG, **kwargs)


def _equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(without_runtime(left), without_runtime(right))


# -- serial ----------------------------------------------------------------


def test_a_serial_execution_fills_the_store_and_matches_run_grid(tmp_path):
    config = mixed_config()
    store = new_store(tmp_path / "run", config)

    summary = _run(store)

    assert (summary.expected, summary.already_done, summary.ran) == (6, 0, 6)
    assert store.status().remaining == 0
    _equal(store.read_frame(), reference_frame(config))


def test_running_a_finished_store_does_nothing(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(2))
    _run(store)

    again = _run(store)

    assert (again.already_done, again.ran) == (2, 0)


def test_progress_is_reported_once_per_cell(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(3))
    seen: list[tuple[int, int]] = []

    _run(store, progress=lambda done, total: seen.append((done, total)))

    assert seen == [(1, 3), (2, 3), (3, 3)]


# -- interruption and resume ------------------------------------------------


def test_a_failing_cell_is_not_recorded_and_stops_the_run(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(4))
    calls = {"n": 0}

    def fails_on_the_third(task):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ZeroDivisionError("injected")
        return fit_cell_to_shard(task)

    # workers=1: the assertion below counts on strict serial order ("the third
    # call, the fourth never attempted"), which only a single process gives. It
    # also keeps `fails_on_the_third` a closure never sent to a worker -- see
    # `execute`'s picklability check, which is what a workers>1 version of this
    # test would hit instead of the ordering it means to test.
    with pytest.raises(CellFailedError, match="ZeroDivisionError") as caught:
        _run(store, workers=1, _task=fails_on_the_third)

    assert "--resume" in str(caught.value)
    # Two cells finished and were recorded; the third produced no ledger line
    # and the fourth was never attempted.
    assert len(store.intact_entries()) == 2
    assert store.status().remaining == 2


def test_a_resumed_run_equals_one_that_was_never_interrupted(tmp_path):
    config = tiny_config(4)
    interrupted = new_store(tmp_path / "interrupted", config)
    calls = {"n": 0}

    def fails_on_the_third(task):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ZeroDivisionError("injected")
        return fit_cell_to_shard(task)

    # workers=1: see the comment on the previous test.
    with pytest.raises(CellFailedError):
        _run(interrupted, workers=1, _task=fails_on_the_third)
    resumed = _run(interrupted)

    assert (resumed.already_done, resumed.ran) == (2, 2)
    _equal(interrupted.read_frame(), reference_frame(config))


def test_a_shard_left_without_a_ledger_line_is_redone_not_trusted(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(2))
    (cell, *_) = expand_cells(tiny_config(2))
    # A kill between writing a shard and recording it leaves exactly this: a
    # file, no line. This one is deliberately wrong (no rows at all).
    write_shard(store.directory, cell_key(cell), [])
    assert store.status().orphans == 1

    _run(store)

    assert store.status().remaining == 0
    assert store.status().orphans == 0
    _equal(store.read_frame(), reference_frame(tiny_config(2)))


def test_a_leftover_temporary_shard_is_swept(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(1))
    leftover = store.directory / "cells" / "abc.parquet.tmp"
    leftover.write_bytes(b"half")

    _run(store)

    assert not leftover.exists()


# -- extension --------------------------------------------------------------


def test_extending_a_run_equals_running_the_larger_grid_directly(tmp_path):
    extended = new_store(tmp_path / "extended", tiny_config(2))
    _run(extended)
    extended.set_replications(4)
    added = _run(extended)

    assert (added.already_done, added.ran) == (2, 2)
    _equal(extended.read_frame(), reference_frame(tiny_config(4)))


# -- ordering ---------------------------------------------------------------


def test_the_dearest_cells_are_submitted_first_whatever_order_they_arrive_in():
    cells = list(expand_cells(mixed_config()))

    forward = longest_first(cells)
    backward = longest_first(list(reversed(cells)))

    assert forward == backward
    assert isinstance(forward[0], PanelCell)
    cross = [c for c in forward if isinstance(c, CrossSectionCell)]
    assert [c.n_units for c in cross] == sorted((c.n_units for c in cross), reverse=True)


def test_a_donor_pool_cell_orders_with_the_panel_cells_and_does_not_raise():
    donors = [DonorPoolCell("S0", 0), DonorPoolCell("S3", 1)]
    cells = [*expand_cells(mixed_config()), *donors]

    forward = longest_first(cells)
    backward = longest_first(list(reversed(cells)))

    assert forward == backward
    nuts = [c for c in forward if not isinstance(c, CrossSectionCell)]
    assert forward[: len(nuts)] == nuts
    assert all(d in nuts for d in donors)


def test_cells_are_scheduled_by_measured_cost_class_then_size():
    cells = [
        CrossSectionCell(scenario="A", n_units=2000, confounding_strength=1.0, seed=0),
        CateCell("RICH", 0),
        DonorPoolCell("S1", 0),
        PanelCell(scenario="P1", n_units=150, n_periods=7, seed=0),
        LongPanelCell("P7", 0),
        CrossSectionCell(scenario="A", n_units=500, confounding_strength=1.0, seed=0),
    ]

    ordered = longest_first(cells)

    # Long panel (529.7s) before panel (35-62s a cell) before donor (5.5-32.6s,
    # straddling CATE's ~15s) before CATE before cross-section (8.7-12.3s), as
    # measured in the design doc's section 9.1,
    # whatever their unit counts: a 2,000-unit CATE cell no longer jumps the queue.
    assert [type(c) for c in ordered] == [
        LongPanelCell, PanelCell, DonorPoolCell, CateCell, CrossSectionCell, CrossSectionCell
    ]
    # Within a shape, larger samples first.
    assert [c.n_units for c in ordered if isinstance(c, CrossSectionCell)] == [2000, 500]


def test_a_long_panel_cell_is_scheduled_before_every_other_shape():
    cells = [PanelCell(scenario="P1", n_units=150, n_periods=10, seed=0), LongPanelCell("P7", 0)]

    assert [type(c) for c in longest_first(cells)] == [LongPanelCell, PanelCell]


def test_every_cell_shape_has_a_cost_rank():
    assert set(SHAPE_COST_RANK) == {
        LongPanelCell, PanelCell, DonorPoolCell, CateCell, CrossSectionCell
    }


def test_an_unranked_cell_type_is_refused_not_scheduled_somewhere():
    class _NewShape:
        scenario = "X"
        seed = 0
        n_units = 10

    with pytest.raises(KeyError):
        longest_first([_NewShape()])  # type: ignore[list-item]


# -- the process pool ---------------------------------------------------------


def test_the_pool_environment_caps_threads_and_restores_what_it_found():
    before = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "XLA_FLAGS")}

    with pool_environment(3):
        assert os.environ["OMP_NUM_THREADS"] == "3"
        assert "intra_op_parallelism_threads=3" in os.environ["XLA_FLAGS"]

    assert {k: os.environ.get(k) for k in before} == before


def test_the_pool_environment_keeps_flags_the_caller_already_set(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/nowhere")

    with pool_environment(1):
        assert os.environ["XLA_FLAGS"].startswith("--xla_dump_to=/nowhere ")

    assert os.environ["XLA_FLAGS"] == "--xla_dump_to=/nowhere"


def test_two_workers_produce_exactly_what_one_does(tmp_path):
    config = mixed_config()
    serial = new_store(tmp_path / "serial", config)
    pooled = new_store(tmp_path / "pooled", config)

    _run(serial, workers=1)
    summary = _run(pooled, workers=2)

    assert summary.workers == 2
    _equal(pooled.read_frame(), serial.read_frame())


def test_a_cell_that_raises_in_a_worker_fails_the_run_and_records_nothing_for_it(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(3))

    with pytest.raises(CellFailedError, match="ZeroDivisionError"):
        _run(store, workers=2, _task=_tasks.raise_on_seed_one)

    assert cell_key_for_seed(store, 1) not in store.intact_entries()
    # Resuming with the real task finishes it.
    _run(store, workers=2)
    _equal(store.read_frame(), reference_frame(tiny_config(3)))


def test_a_worker_that_dies_fails_the_run_and_a_resume_completes_it(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(3))

    with pytest.raises(CellFailedError, match="--resume"):
        _run(store, workers=2, _task=_tasks.die_on_seed_one)

    assert cell_key_for_seed(store, 1) not in store.intact_entries()
    _run(store, workers=2)
    _equal(store.read_frame(), reference_frame(tiny_config(3)))


def test_workers_must_be_positive(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(1))

    with pytest.raises(ValueError, match="workers"):
        _run(store, workers=0)


def test_an_unpicklable_task_is_refused_before_a_worker_ever_starts(tmp_path):
    # A closure sent through the pool's queue at workers > 1 does not fail fast
    # with a clear error on this machine -- it hangs. This is the check that
    # turns that into an immediate, named ValueError instead.
    store = new_store(tmp_path / "run", tiny_config(1))
    calls = {"n": 0}

    def a_closure(task):
        calls["n"] += 1
        return task

    with pytest.raises(ValueError, match="cannot be sent to a worker process"):
        _run(store, workers=2, _task=a_closure)


def cell_key_for_seed(store: RunStore, seed: int) -> str:
    (cell,) = [c for c in store.expected_cells().values() if c.seed == seed]
    return cell_key(cell)
