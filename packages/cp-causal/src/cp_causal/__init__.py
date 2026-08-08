"""Causal estimators, diagnostics and orchestration.

This package must never import the ground-truth simulation package. Estimators
are required to have no code path to ground truth, and that separation is
enforced mechanically by the import-linter contract in `.importlinter`.

Importing this package registers every estimator, so the router and the
comparison layer see a complete set without callers importing each module.
"""

from cp_causal.estimators import (  # noqa: F401
    doubly_robust,
    experimental,
    regression,
    weighting,
)

__version__ = "0.1.0"
