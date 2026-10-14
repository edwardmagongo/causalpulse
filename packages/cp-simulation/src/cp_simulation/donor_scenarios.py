"""Named donor-pool scenario presets.

A third registry, distinct from both the cross-sectional `SCENARIOS` and the
panel `PANEL_SCENARIOS`: donor-pool scenarios are not points on either grid.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cp_simulation.components import UnitGenerator
from cp_simulation.donor_components import (
    ClusteredUnitEffects,
    LevelOffsetUnitEffects,
    SingleUnitAdoption,
)
from cp_simulation.panel_components import ConstantDynamicEffect, LinearTimePath, NormalUnitEffects
from cp_simulation.panel_dgp import PanelDGP
from cp_simulation.panel_truth import SimulatedPanel

# The protocol every donor-pool recovery measurement in this repository runs
# at. Fixed here, not chosen per test, for the same reason PANEL_N_UNITS is
# fixed in `panel_scenarios.py`: a test that quietly uses fewer donors or
# periods is measuring a different, unvalidated regime.
DONOR_N_DONORS = 17
DONOR_N_UNITS = DONOR_N_DONORS + 1
DONOR_N_PERIODS = 24
DONOR_START = 18  # 17 pre-periods, 7 post-periods
DONOR_SEEDS = 400

_DONOR_COVARIATES = ("x1", "x2")


@dataclass(frozen=True, eq=False)
class ContaminatedDonorPanel:
    """Wraps a `PanelDGP`, adding a hidden shock to one donor's outcome.

    Not a new generative model: the shock is layered onto an otherwise
    ordinary `PanelDGP` panel's *realised* outcome after generation, which is
    safe precisely because the contaminated donor is never in the treated
    cohort -- `PanelGroundTruth.true_att_by_cohort` describes the treated
    unit's effect alone and is untouched by a shock to a different unit's
    outcome column (pinned by
    `test_s4_shocks_exactly_one_donor_and_leaves_ground_truth_untouched`).

    `donor_offset` indexes into the donor pool (units other than the treated
    one) in sorted-unit-id order, not into the full unit list.
    """

    base: PanelDGP
    donor_offset: int
    shock: float
    shock_start: int

    def generate(self, n_units: int, n_periods: int, rng: np.random.Generator) -> SimulatedPanel:
        panel = self.base.generate(n_units, n_periods, rng)
        frame = panel.dataset.frame.copy()
        treated_unit = frame.loc[frame["cohort"] > 0, "unit_id"].iloc[0]
        donor_ids = np.sort(frame.loc[frame["unit_id"] != treated_unit, "unit_id"].unique())
        contaminated = donor_ids[self.donor_offset]
        mask = (frame["unit_id"] == contaminated) & (frame["period"] >= self.shock_start)
        frame.loc[mask, "outcome"] += self.shock
        dataset = type(panel.dataset)(frame=frame, schema=panel.dataset.schema)
        return SimulatedPanel(dataset=dataset, truth=panel.truth)


@dataclass(frozen=True)
class DonorScenarioSpec:
    scenario_id: str
    name: str
    declared_effect: float
    description: str
    build: Callable[[], PanelDGP | ContaminatedDonorPanel]
    n_units: int = DONOR_N_UNITS
    n_periods: int = DONOR_N_PERIODS


def build_donor_scenario(scenario_id: str) -> PanelDGP | ContaminatedDonorPanel:
    return DONOR_SCENARIOS[scenario_id].build()


def _representative(scenario_id: str, tau: float, start: int = DONOR_START) -> PanelDGP:
    return PanelDGP(
        unit_generator=UnitGenerator(covariate_names=_DONOR_COVARIATES),
        unit_effects=NormalUnitEffects(level_sd=0.3, slope_sd=0.1),
        time_path=LinearTimePath(slope=0.1),
        adoption=SingleUnitAdoption(start=start),
        effect=ConstantDynamicEffect(tau=tau),
        scenario_id=scenario_id,
        noise_sd=0.5,
    )


DONOR_SCENARIOS: dict[str, DonorScenarioSpec] = {
    "S0": DonorScenarioSpec(
        scenario_id="S0",
        name="Null effect",
        declared_effect=0.0,
        description=(
            "S1's representative single-unit adoption with tau=0. Calibrates "
            "the permutation p-value's false-positive rate, the role P2 "
            "played for the DiD family in Phase 3."
        ),
        build=lambda: _representative("S0", tau=0.0),
    ),
    "S1": DonorScenarioSpec(
        scenario_id="S1",
        name="Clean single-unit adoption",
        declared_effect=1.0,
        description=(
            "The treated unit's own slope is drawn from the same "
            "distribution as the donor pool -- no divergence, no "
            "adversarial construction. The recovery baseline every other "
            "scenario is judged against."
        ),
        build=lambda: _representative("S1", tau=1.0),
    ),
    "S2": DonorScenarioSpec(
        scenario_id="S2",
        name="Divergent pre-trend",
        declared_effect=1.0,
        description=(
            "The treated unit sits in a donor sub-cluster with a "
            "distinctly different slope from the rest of the pool. "
            "Synthetic control must concentrate weight on the matching "
            "cluster to recover truth; a naive equal-weighted donor "
            "average, mixing both clusters, is badly biased. The scenario "
            "this phase exists to demonstrate."
        ),
        build=lambda: PanelDGP(
            unit_generator=UnitGenerator(covariate_names=_DONOR_COVARIATES),
            unit_effects=ClusteredUnitEffects(
                n_similar=9, mean_a=0.3, mean_b=-0.1, within_sd=0.02, level_sd=0.3
            ),
            time_path=LinearTimePath(slope=0.1),
            adoption=SingleUnitAdoption(start=DONOR_START),
            effect=ConstantDynamicEffect(tau=1.0),
            scenario_id="S2",
            noise_sd=0.5,
        ),
    ),
    "S3": DonorScenarioSpec(
        scenario_id="S3",
        name="Convex-hull violation",
        declared_effect=1.0,
        description=(
            "The treated unit's baseline level sits 5.0 above the donor "
            "pool's own maximum by construction (`LevelOffsetUnitEffects`), "
            "so no non-negative simplex-constrained combination of donors "
            "can match its pre-period path. Triggers the convex_hull "
            "assumption check and exercises the ridge-augmented fallback."
        ),
        build=lambda: PanelDGP(
            unit_generator=UnitGenerator(covariate_names=_DONOR_COVARIATES),
            unit_effects=LevelOffsetUnitEffects(
                base=NormalUnitEffects(level_sd=0.3, slope_sd=0.1),
                unit_index=0,
                level_offset=5.0,
            ),
            time_path=LinearTimePath(slope=0.1),
            adoption=SingleUnitAdoption(start=DONOR_START),
            effect=ConstantDynamicEffect(tau=1.0),
            scenario_id="S3",
            noise_sd=0.5,
        ),
    ),
    "S4": DonorScenarioSpec(
        scenario_id="S4",
        name="Donor contamination",
        declared_effect=1.0,
        description=(
            "S1's DGP, with donor 0 receiving its own +5.0 hidden shock "
            "from the treatment period onward -- a shift no adoption "
            "mechanism declares and that never appears in ground truth. "
            "Triggers the donor_contamination assumption check. Corrected "
            "from an original +1.5: the check's real statistic (the max "
            "robust z-score across all donors, not one pre-specified donor) "
            "has much lower power than a single-donor check, so a smaller "
            "shock was not reliably detectable at a well-calibrated "
            "threshold. See donor_contamination_check's own docstring."
        ),
        build=lambda: ContaminatedDonorPanel(
            base=_representative("S4", tau=1.0),
            donor_offset=0,
            shock=5.0,
            shock_start=DONOR_START,
        ),
    ),
    "S5": DonorScenarioSpec(
        scenario_id="S5",
        name="Insufficient pre-periods",
        declared_effect=1.0,
        description=(
            "S1's DGP collapsed to 6 periods with adoption at period 3, "
            "leaving only 2 pre-treatment periods -- well under this "
            "estimator's declared minimum of 10. Triggers "
            "sufficient_pre_periods, reusing the check Phase 3 already "
            "built rather than a new one."
        ),
        build=lambda: _representative("S5", tau=1.0, start=3),
        n_units=DONOR_N_UNITS,
        n_periods=6,
    ),
}
