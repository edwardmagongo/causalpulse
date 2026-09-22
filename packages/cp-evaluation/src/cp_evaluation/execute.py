"""Running a store's remaining cells, in parallel, without losing what finished.

One task is one cell. A worker fits the cell and writes its shard
(`store.write_shard`); the parent, and only the parent, appends the ledger line
when the worker reports back. Nothing here decides what a cell contains -- that
is `run.fit_cell`, the same function the serial `run_grid` calls -- so the
serial and parallel paths cannot fit a cell differently.

A cell whose fit raises anything other than the documented refusal fails the
whole run. It is never recorded as a row: a benchmark that turns a crash into a
result reports a number over the survivors and calls it a study. Everything
already in the ledger survives, and running again resumes.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from cp_core.values import EstimatorConfig

from cp_evaluation.grid import Cell, PanelCell, cell_key
from cp_evaluation.run import fit_cell, resolve_candidates
from cp_evaluation.store import LedgerEntry, RunStore, write_shard

# Measured 2026-09-22 on this 16 GB, 10-core machine (`benchmarks/measure_workers.py`,
# 36-cell sample at the study's own bootstrap settings), against a rule fixed
# before the measurement: the largest of 1, 2, 4 workers whose wall clock is at
# least 1.3x faster than the next smaller count's, and whose peak swap rose by
# under 2,000 MB over the swap already in use.
#
#   workers   wall s   speedup   swap before MB   swap peak MB
#         1    725.5      1.00             3431           3431
#         2    390.3      1.86             3431           3431
#         4    298.2      1.31 (vs 2)      3431           4421 (+990)
#
# 2 passes both clauses (1.86x, +0 MB). 4 also passes both, though its speedup
# over 2 (1.31x) clears the 1.3x bar narrowly: rerunning it would not be
# surprising to land the other side of 1.30. The rule takes the largest count
# that passes, so the default is 4. See the design document's section 13.2 for
# the full record.
DEFAULT_WORKERS = 4

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class CellFailedError(RuntimeError):
    """A cell's fit raised something other than a documented refusal."""


@dataclass(frozen=True)
class CellTask:
    """Everything a worker needs to fit one cell and write its shard."""

    cell: Cell
    root_entropy: int
    fit_config: EstimatorConfig
    estimators: tuple[str, ...]
    directory: Path


def fit_cell_to_shard(task: CellTask) -> LedgerEntry:
    """The unit of work: fit one cell, write its shard, describe the file."""
    rows = fit_cell(
        task.cell,
        root_entropy=task.root_entropy,
        fit_config=task.fit_config,
        candidates=resolve_candidates(task.estimators),
    )
    return write_shard(task.directory, cell_key(task.cell), rows)


TaskFunction = Callable[[CellTask], LedgerEntry]


def longest_first(cells: Sequence[Cell]) -> list[Cell]:
    """Cells ordered so the dearest are submitted first.

    Panel cells carry the slow estimators (`hierarchical_did` samples with NUTS)
    and larger samples cost more, so submitting them first stops the run ending
    with one straggler while the other workers idle. The key ends in `cell_key`,
    so the order is total and does not depend on the input's. It changes only
    when cells finish, never what they compute.
    """
    return sorted(
        cells,
        key=lambda cell: (not isinstance(cell, PanelCell), -cell.n_units, cell_key(cell)),
    )


@contextmanager
def pool_environment(threads: int) -> Iterator[None]:
    """Cap each worker's numerical thread pools, restoring the environment after.

    Set in the parent before any worker starts: a spawned worker copies the
    parent's environment when it is created and imports numpy and JAX before any
    code of ours runs, so an initializer would be too late. Without a cap, N
    workers each open a machine's worth of BLAS and XLA threads.
    """
    names = (*_THREAD_VARIABLES, "XLA_FLAGS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in _THREAD_VARIABLES:
            os.environ[name] = str(threads)
        existing = os.environ.get("XLA_FLAGS", "")
        eigen = "true" if threads > 1 else "false"
        cap = f"--xla_cpu_multi_thread_eigen={eigen} intra_op_parallelism_threads={threads}"
        os.environ["XLA_FLAGS"] = f"{existing} {cap}".strip()
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass(frozen=True)
class ExecutionSummary:
    expected: int
    already_done: int
    ran: int
    workers: int


def _failure(task: CellTask, exc: BaseException) -> CellFailedError:
    return CellFailedError(
        f"fitting {task.cell!r} failed with {type(exc).__name__}: {exc}. "
        "Nothing was recorded for it; run again with --resume to continue."
    )


def execute(
    store: RunStore,
    *,
    estimators: Sequence[str],
    fit_config: EstimatorConfig,
    workers: int = DEFAULT_WORKERS,
    threads_per_worker: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    _task: TaskFunction = fit_cell_to_shard,
) -> ExecutionSummary:
    """Fit every cell the store has not finished, recording each as it completes.

    `_task` exists only so a test can make a chosen cell fail or a worker die;
    production callers never pass it. At `workers > 1` it must be picklable by
    reference (a module-level function, not a closure): a closure sent through
    the pool's queue does not fail fast with a clear error on this machine --
    it hangs indefinitely, apparently only when the process also imports this
    project's heavier dependencies (found the hard way, measuring Task 10's
    default). Checked eagerly here so that failure mode becomes a `ValueError`
    naming the reason, not a silent stall a caller has to diagnose from a
    process list.
    """
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    if workers > 1:
        try:
            pickle.dumps(_task)
        except (AttributeError, TypeError) as exc:
            raise ValueError(
                f"_task {_task!r} cannot be sent to a worker process (workers={workers}): "
                f"{exc}. A closure only works at workers=1; pass a module-level function."
            ) from exc
    store.sweep_temporary()

    expected = store.expected_cells()
    done = set(store.intact_entries()) & set(expected)
    todo = longest_first([cell for key, cell in expected.items() if key not in done])
    tasks = [
        CellTask(
            cell=cell,
            root_entropy=store.identity.root_entropy,
            fit_config=fit_config,
            estimators=tuple(estimators),
            directory=store.directory,
        )
        for cell in todo
    ]
    summary = ExecutionSummary(
        expected=len(expected), already_done=len(done), ran=len(tasks), workers=workers
    )
    if not tasks:
        return summary

    if workers == 1:
        for number, task in enumerate(tasks, start=1):
            try:
                entry = _task(task)
            except Exception as exc:
                raise _failure(task, exc) from exc
            store.record(entry)
            if progress is not None:
                progress(number, len(tasks))
        return summary

    threads = threads_per_worker or max(1, (os.cpu_count() or 1) // workers)
    with (
        pool_environment(threads),
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool,
    ):
        futures = {pool.submit(_task, task): task for task in tasks}
        try:
            for number, future in enumerate(as_completed(futures), start=1):
                try:
                    entry = future.result()
                except Exception as exc:
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise _failure(futures[future], exc) from exc
                store.record(entry)
                if progress is not None:
                    progress(number, len(tasks))
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return summary
