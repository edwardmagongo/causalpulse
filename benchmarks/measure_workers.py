"""Time a fixed sample of study cells at 1, 2 and 4 workers, and watch swap.

Run from the repository root:  uv run python benchmarks/measure_workers.py [--quick]

The sample is a small grid of the same shape as the study's, at the study's own fit
settings (`FULL_ESTIMATORS`, `bootstrap_reps=1000`), so a cell here costs what a
study cell costs. Panel cells carry `hierarchical_did`, the dear ones. Each worker
count runs into a fresh store, so nothing is resumed from an earlier count.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from cp_evaluation.execute import execute
from cp_evaluation.grid import CrossSectionAxes, GridConfig, PanelAxes, expand_cells
from cp_evaluation.store import RunIdentity, RunStore
from cp_evaluation.tiers import FULL_ESTIMATORS, FULL_FIT_CONFIG

QUICK = "--quick" in sys.argv


def sample_config() -> GridConfig:
    if QUICK:
        return GridConfig(
            cross_section=CrossSectionAxes(("A",), (200,), (1.0,)),
            panel=PanelAxes(("P1",), (100,), (7,)),
            replications=2,
        )
    return GridConfig(
        cross_section=CrossSectionAxes(("A", "D", "F"), (500,), (1.0,)),
        panel=PanelAxes(("P1", "P2", "P3", "P4", "P5", "P6"), (150,), (7,)),
        replications=4,
    )


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used =")[1].split("M")[0])


class SwapWatcher(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = swap_used_mb()
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.wait(2.0):
            self.peak = max(self.peak, swap_used_mb())


def main() -> None:
    config = sample_config()
    counts = (1, 2) if QUICK else (1, 2, 4)
    print(f"sample: {len(expand_cells(config))} cells")
    header = ("workers", "wall s", "speedup", "swap before MB", "swap peak MB")
    print(f"{header[0]:>8} {header[1]:>8} {header[2]:>8} {header[3]:>15} {header[4]:>13}")
    baseline = None
    for workers in counts:
        directory = Path(tempfile.mkdtemp()) / "run"
        identity = RunIdentity.of("sample", config, FULL_ESTIMATORS, FULL_FIT_CONFIG, 0)
        store = RunStore.create(directory, identity, config.replications)
        before = swap_used_mb()
        watcher = SwapWatcher()
        watcher.start()
        started = time.perf_counter()
        execute(store, estimators=FULL_ESTIMATORS, fit_config=FULL_FIT_CONFIG, workers=workers)
        wall = time.perf_counter() - started
        watcher.stop.set()
        baseline = baseline or wall
        speedup = baseline / wall
        print(
            f"{workers:>8} {wall:>8.1f} {speedup:>8.2f} {before:>15.0f} {watcher.peak:>13.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
