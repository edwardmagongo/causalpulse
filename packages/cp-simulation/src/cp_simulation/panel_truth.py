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


def pre_trend_gap(
    untreated_path: np.ndarray, cohorts: np.ndarray, periods: np.ndarray
) -> float:
    """Measured Y(0) trend-slope gap, ever-treated minus never-treated.

    Measured from the untreated potential outcomes rather than declared, so
    parallel trends is an observable property of a scenario rather than a claim
    about it — the precedent `confounders()` set for the cross-sectional DGP.
    Returns nan when either arm is empty, since there is then no gap to report.
    """
    ever = cohorts > 0
    if not ever.any() or ever.all():
        return float("nan")
    centred = periods - periods.mean()
    denominator = float((centred**2).sum())
    slopes = (untreated_path * centred[None, :]).sum(axis=1) / denominator
    return float(slopes[ever].mean() - slopes[~ever].mean())


def measure_panel_truth(
    cohorts: np.ndarray,
    effect_matrix: np.ndarray,
    treated: np.ndarray,
    untreated_path: np.ndarray,
    periods: np.ndarray,
    scenario_id: str,
    config_hash: str,
) -> PanelGroundTruth:
    """Aggregate an already-computed per-cell effect matrix into ground truth.

    Shared by `PanelDGP` and `BinaryPanelDGP`, whose `_measure` methods differ
    only in the units of `effect_matrix` (potential-outcome difference `tau`
    for the continuous DGP, `p1 - p0` for the binary one) and in what
    `untreated_path` is (`y0` versus `p0`). Everything downstream of having a
    per-cell effect matrix -- the (g, t) grid, the cohort and event-time
    aggregates, the overall ATT, and the pre-trend gap -- is identical, and
    lives here exactly once so a future fix to the aggregation logic cannot
    apply to one DGP's truth definition and not the other's.
    """
    att_gt: dict[tuple[int, int], float] = {}
    by_cohort: dict[int, float] = {}
    for cohort in sorted({int(c) for c in cohorts if c > 0}):
        members = cohorts == cohort
        for index, period in enumerate(periods.astype(int)):
            if period < cohort:
                continue
            att_gt[(cohort, int(period))] = float(effect_matrix[members, index].mean())
        cells = treated[members]
        by_cohort[cohort] = float(effect_matrix[members][cells].mean())

    exposure = periods.astype(int)[None, :] - cohorts[:, None]
    by_event_time = {
        int(horizon): float(effect_matrix[treated & (exposure == horizon)].mean())
        for horizon in sorted({int(h) for h in exposure[treated]})
    }

    return PanelGroundTruth(
        true_att_gt=att_gt,
        true_att_overall=(
            float(effect_matrix[treated].mean()) if treated.any() else float("nan")
        ),
        true_att_by_cohort=by_cohort,
        true_att_by_event_time=by_event_time,
        true_effect_matrix=np.where(treated, effect_matrix, 0.0),
        true_cohorts=cohorts,
        true_pre_trend_gap=pre_trend_gap(untreated_path, cohorts, periods),
        scenario_id=scenario_id,
        config_hash=config_hash,
    )
