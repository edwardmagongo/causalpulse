"""Ground truth for a simulated panel.

A separate type from `GroundTruth` rather than a superset of it. A single
record with `true_propensity` meaningless in panel mode and `true_att_gt` None
in cross-section is precisely how a wrong number gets read confidently, and on
a project whose central discipline is that truth is measured, that trade is not
worth the cosmetic unification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from cp_core.dataset import CausalDataset


# eq=False: a generated __eq__ would raise on the ambiguous truth value of the
# ndarray fields, and a generated __hash__ would raise on the unhashable
# mapping fields.
@dataclass(frozen=True, eq=False)
class PanelGroundTruth:
    """The true causal quantities of a simulated panel.

    Every field is measured from the realised potential-outcome matrices, never
    copied from an input parameter. `true_att_gt[(g, t)]` is the mean of
    `Y_it(1) - Y_it(0)` over cohort-g units at period t, for t >= g.
    """

    true_att_gt: Mapping[tuple[int, int], float]
    true_att_overall: float
    true_att_by_cohort: Mapping[int, float]
    true_att_by_event_time: Mapping[int, float]
    true_effect_matrix: np.ndarray
    true_cohorts: np.ndarray
    true_pre_trend_gap: float
    scenario_id: str
    config_hash: str


@dataclass(frozen=True)
class SimulatedPanel:
    """A generated panel and its ground truth, deliberately kept apart."""

    dataset: CausalDataset
    truth: PanelGroundTruth
