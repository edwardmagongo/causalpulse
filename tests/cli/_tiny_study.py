"""The `study` tier pointed at a six-cell grid, for CLI tests.

The real tier is 4,000 cells and hours of fitting. A test monkeypatches the tier
table to this instead, so the command's own paths run over a grid a test can afford.
"""

import cp_cli.commands.bench as bench_module
from evaluation._run_helpers import ESTIMATORS, FIT_CONFIG, mixed_config


def use_tiny_study_tier(monkeypatch) -> None:
    monkeypatch.setitem(
        bench_module.TIERS, "study", (lambda: mixed_config(2), ESTIMATORS, FIT_CONFIG)
    )
