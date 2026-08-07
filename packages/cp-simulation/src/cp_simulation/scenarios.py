"""Named scenario presets over the component grid.

Presets exist for legibility; the benchmark grid varies confounding strength
continuously through `build_scenario`, which is why each spec is a builder
rather than a fixed instance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cp_simulation.components import (
    ConstantEffect,
    LogisticBaseline,
    PropensityAssignment,
    RandomAssignment,
    UnitGenerator,
)
from cp_simulation.dgp import CrossSectionDGP

_COVARIATES = ("x1", "x2")
_BASELINE = LogisticBaseline(intercept=-1.0, coefficients=(0.8, 0.5))


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    name: str
    declared_effect: float
    description: str
    build: Callable[[float], CrossSectionDGP]


def _randomised(scenario_id: str, tau: float) -> Callable[[float], CrossSectionDGP]:
    def build(_confounding_strength: float) -> CrossSectionDGP:
        return CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_COVARIATES),
            baseline=_BASELINE,
            effect=ConstantEffect(tau=tau),
            assignment=RandomAssignment(share=0.5),
            scenario_id=scenario_id,
        )

    return build


def _confounded(scenario_id: str, tau: float) -> Callable[[float], CrossSectionDGP]:
    def build(confounding_strength: float) -> CrossSectionDGP:
        return CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_COVARIATES),
            baseline=_BASELINE,
            effect=ConstantEffect(tau=tau),
            assignment=PropensityAssignment(
                coefficients=(0.9, 0.7), strength=confounding_strength
            ),
            scenario_id=scenario_id,
        )

    return build


SCENARIOS: dict[str, ScenarioSpec] = {
    "A": ScenarioSpec(
        scenario_id="A",
        name="Simple randomised experiment",
        declared_effect=0.05,
        description="Random assignment, homogeneous +5pp effect.",
        build=_randomised("A", 0.05),
    ),
    "B": ScenarioSpec(
        scenario_id="B",
        name="No treatment effect",
        declared_effect=0.0,
        description="Random assignment, exactly zero effect. Any systematic "
        "positive finding here is a false positive.",
        build=_randomised("B", 0.0),
    ),
    "C": ScenarioSpec(
        scenario_id="C",
        name="Harmful intervention",
        declared_effect=-0.03,
        description="Random assignment, homogeneous -3pp effect.",
        build=_randomised("C", -0.03),
    ),
    "D": ScenarioSpec(
        scenario_id="D",
        name="Confounding",
        declared_effect=0.05,
        description="Covariates drive both assignment and outcome, so naive "
        "comparisons are biased upward.",
        build=_confounded("D", 0.05),
    ),
}


def build_scenario(scenario_id: str, confounding_strength: float = 1.0) -> CrossSectionDGP:
    """Instantiate a scenario's DGP at a given confounding strength."""
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"unknown scenario {scenario_id!r}; available: {sorted(SCENARIOS)}"
        )
    return SCENARIOS[scenario_id].build(confounding_strength)
