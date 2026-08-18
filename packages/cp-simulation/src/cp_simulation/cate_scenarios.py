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
    ConstantEffect,
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
_NOISY_COVARIATES = tuple(f"x{i}" for i in range(12))

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

_NULL_EFFECT = ConstantEffect(tau=0.05)


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


def _noisy_baseline() -> LogisticBaseline:
    coefficients = [0.2, -0.3, 0.0, 0.1] + [0.0] * 8
    return LogisticBaseline(intercept=0.0, coefficients=tuple(coefficients))


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
    "NULL": CATEScenarioSpec(
        scenario_id="NULL",
        name="Null heterogeneity",
        description=(
            "Effect is constant (tau=0.05) across every covariate. This is "
            "the scenario the omnibus calibration test must correctly pass "
            "on -- reporting 'no reliable evidence of heterogeneity' and no "
            "subgroups -- the guard this phase exists to prove works, not "
            "just the forest's ability to find signal when it's there."
        ),
        build=lambda: CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_CATE_COVARIATES),
            baseline=_baseline(),
            effect=_NULL_EFFECT,
            assignment=RandomAssignment(share=0.5),
            scenario_id="NULL",
        ),
    ),
    "NOISY": CATEScenarioSpec(
        scenario_id="NOISY",
        name="Many noise covariates",
        description=(
            "Same interaction structure as RICH (x0/x1/x2 drive tau) but "
            "widened to 12 covariates, 9 of which are pure noise. Large "
            "enough for Benjamini-Hochberg FDR control across subgroup "
            "comparisons to have something real to correct: many candidate "
            "subgroups on noise covariates that must NOT survive correction."
        ),
        build=lambda: CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_NOISY_COVARIATES),
            baseline=_noisy_baseline(),
            effect=_RICH_EFFECT,
            assignment=RandomAssignment(share=0.5),
            scenario_id="NOISY",
        ),
    ),
}


def build_cate_scenario(scenario_id: str) -> CrossSectionDGP:
    return CATE_SCENARIOS[scenario_id].build()
