"""Probes used to validate the data-generating processes themselves.

These are not estimators for real analysis. The oracle reads ground truth
directly, and the naive comparison exists to demonstrate that confounded
scenarios really are confounded. Both live here rather than in cp-causal, since
cp-causal is forbidden from importing this package.
"""

from __future__ import annotations

import math

from cp_core.dataset import CausalDataset

from cp_simulation.truth import SimulatedDataset


def oracle_ate(simulated: SimulatedDataset) -> float:
    """The average treatment effect read straight off the potential outcomes."""
    cate = simulated.truth.true_cate
    if len(cate) == 0:
        raise ValueError("true_cate is empty or undefined; cannot compute oracle ATE")
    ate = float(cate.mean())
    if math.isnan(ate):
        raise ValueError("true_cate is empty or undefined; cannot compute oracle ATE")
    return ate


def naive_difference_in_means(dataset: CausalDataset) -> float:
    """Raw treated-minus-control difference, ignoring confounding entirely."""
    frame = dataset.frame
    treated = frame[frame[dataset.schema.treatment] == 1][dataset.schema.outcome]
    control = frame[frame[dataset.schema.treatment] == 0][dataset.schema.outcome]
    if treated.empty:
        raise ValueError("no treated units")
    if control.empty:
        raise ValueError("no control units")
    return float(treated.mean() - control.mean())
