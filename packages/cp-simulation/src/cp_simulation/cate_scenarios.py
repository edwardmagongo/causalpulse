"""CATE scenario presets: cross-sectional data with genuine multi-covariate
heterogeneity, distinct from both `SCENARIOS` (A-J) and the panel/donor-pool
registries -- these are not points on either existing grid.

See the design doc (`docs/specs/2026-08-17-causal-core-phase6-design.md`)
§3.1 for the rationale per scenario.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cp_simulation.components import (
    LogisticBaseline,
    MultivariateHeterogeneousEffect,
    RandomAssignment,
    UnitGenerator,
)
from cp_simulation.dgp import CrossSectionDGP

# Fixed for every CATE-scenario measurement in this repository, the same
# reason DONOR_N_DONORS is fixed in donor_scenarios.py: a test that quietly
# uses a different unit count or covariate width is measuring a different,
# unvalidated regime.
CATE_N_UNITS = 2000
_CATE_COVARIATES = ("x0", "x1", "x2", "x3", "x4", "x5")

_RICH_EFFECT = MultivariateHeterogeneousEffect(
    base_tau=0.05,
    continuous_index=0,
    continuous_coefficient=0.0,
    binary_interaction_index=1,
    binary_interaction_continuous_index=0,
    binary_interaction_coefficient=0.08,
    secondary_continuous_index=2,
    secondary_coefficient=0.03,
    secondary_clip=1.0,
)


@dataclass(frozen=True)
class CATEScenarioSpec:
    scenario_id: str
    name: str
    description: str
    build: Callable[[], CrossSectionDGP]


def _baseline() -> LogisticBaseline:
    return LogisticBaseline(
        intercept=0.0, coefficients=(0.2, -0.3, 0.0, 0.1, 0.0, 0.0)
    )


CATE_SCENARIOS: dict[str, CATEScenarioSpec] = {
    "RICH": CATEScenarioSpec(
        scenario_id="RICH",
        name="Rich heterogeneity",
        description=(
            "Effect driven by an interaction between a continuous covariate "
            "(x0) and a binary covariate (x1), plus a smaller independent "
            "effect from a second continuous covariate (x2). x3-x5 have no "
            "effect on tau. Not recoverable by a single interaction term, "
            "unlike scenario F's one-covariate step function -- this is "
            "what per-unit CATE and subgroup recovery get scored against."
        ),
        build=lambda: CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_CATE_COVARIATES),
            baseline=_baseline(),
            effect=_RICH_EFFECT,
            assignment=RandomAssignment(share=0.5),
            scenario_id="RICH",
        ),
    ),
}


def build_cate_scenario(scenario_id: str) -> CrossSectionDGP:
    return CATE_SCENARIOS[scenario_id].build()
