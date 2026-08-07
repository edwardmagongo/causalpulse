"""Ground truth carried alongside simulated data.

Held separately from the CausalDataset so that no estimator can reach it. The
import-linter contract makes that separation mechanical rather than advisory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cp_core.dataset import CausalDataset


# eq=False: this holds array-valued fields, so a generated __eq__ would raise
# on the ambiguous truth value of an array rather than compare. Identity
# equality is the honest semantics for a container of measurements.
@dataclass(frozen=True, eq=False)
class GroundTruth:
    """The true causal quantities of a simulated population.

    Every field is measured from realised potential outcomes, never copied from
    an input parameter.
    """

    true_ate: float
    true_att: float
    true_cate: np.ndarray
    true_propensity: np.ndarray
    true_confounders: tuple[str, ...]
    scenario_id: str
    config_hash: str


@dataclass(frozen=True)
class SimulatedDataset:
    """A generated dataset and its ground truth, deliberately kept apart."""

    dataset: CausalDataset
    truth: GroundTruth
