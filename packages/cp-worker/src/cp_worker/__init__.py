"""Runs analyses the API queued: stored CSV in, `cp_report.pipeline.analyse`, rows out.

Imports `cp_api.db` and nothing else from `cp_api`
(`.importlinter`'s `worker-sees-only-the-db` contract).
"""

__version__ = "0.1.0"
