"""Benchmark runs, logged to MLflow for comparison across runs and commits.

Only benchmark runs are tracked. API analyses are not: their manifest and the
Postgres rows beside it already record everything MLflow would, scoped to
their owner. `cp_tracking.payload` decides what to log and never imports
MLflow; `cp_tracking.log` is the one module that does.
"""

__version__ = "0.1.0"
