"""Causal estimators, diagnostics and orchestration.

This package must never import the ground-truth simulation package. Estimators
are required to have no code path to ground truth, and that separation is
enforced mechanically by the import-linter contract in `.importlinter`.

Importing this package registers every estimator, so the router and the
comparison layer see a complete set without callers importing each module.
"""

from cp_causal._support import require_no_missing
from cp_causal.estimators import (  # noqa: F401
    bsts,
    callaway_santanna,
    causal_forest,
    did,
    doubly_robust,
    event_study,
    experimental,
    hierarchical_did,
    regression,
    synthetic_control,
    weighting,
)

__all__ = ["require_no_missing"]
__version__ = "0.1.0"
