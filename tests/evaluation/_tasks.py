"""Fault-injecting cell tasks for the executor's process-pool tests.

A spawned worker imports the task function by module path, so these live in an
importable module rather than inside a test file's local scope.
"""

import os

from cp_evaluation.execute import CellTask, fit_cell_to_shard
from cp_evaluation.store import LedgerEntry


def raise_on_seed_one(task: CellTask) -> LedgerEntry:
    if task.cell.seed == 1:
        raise ZeroDivisionError("injected failure")
    return fit_cell_to_shard(task)


def die_on_seed_one(task: CellTask) -> LedgerEntry:
    if task.cell.seed == 1:
        os._exit(1)
    return fit_cell_to_shard(task)
