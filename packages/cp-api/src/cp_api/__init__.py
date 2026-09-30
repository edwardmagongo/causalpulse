"""The HTTP API over the analysis pipeline.

Imports `cp_core` and its own modules only; `.importlinter`'s `api-cannot-fit`
contract enforces it. Nothing here fits a model. This file imports nothing,
so the worker's one sanctioned import, `cp_api.db`, never drags in the app.
"""

__version__ = "0.1.0"
