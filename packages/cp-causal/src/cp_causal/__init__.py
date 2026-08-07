"""Causal estimators, diagnostics and orchestration.

This package must never import the ground-truth simulation package. Estimators
are required to have no code path to ground truth, and that separation is
enforced mechanically by the import-linter contract in `.importlinter`.
"""

__version__ = "0.1.0"
