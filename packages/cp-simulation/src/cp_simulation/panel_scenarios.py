"""Named panel scenario presets.

A registry distinct from the cross-sectional `SCENARIOS`: the parent spec's A-J
letters describe one cross-sectional component grid, and panel scenarios are
not points on it.

The `[0.05, 0.85]` baseline bound and the honest window `tau in [-0.05, +0.15]`
are artefacts of the `p1 = clip(p0 + tau, 0, 1)` construction on the
probability scale. They do not apply to P1-P5, which are continuous and
unclipped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cp_simulation.components import UnitGenerator
from cp_simulation.panel_components import (
    Adoption,
    ConstantDynamicEffect,
    DynamicEffect,
    LinearTimePath,
    NormalUnitEffects,
    RampDynamicEffect,
    StaggeredAdoption,
    TrendSelectedAdoption,
    UniformAdoption,
)
from cp_simulation.panel_dgp import PanelDGP

# The protocol every panel recovery measurement in this repository runs at.
# Fixed here rather than chosen per test: design section 4.4 measured every
# tolerance at these values, and a test that quietly uses fewer units or seeds
# is measuring something else. At 200 units and 30 seeds the P1 bias reads
# +0.029, which looks like a real bias and is not.
PANEL_N_UNITS = 400
PANEL_N_PERIODS = 8
PANEL_SEEDS = 60

_COVARIATES = ("x1", "x2")
_UNIFORM_START = 5
_COHORT_PERIODS = (3, 5, 7)
_NEVER_SHARE = 0.4


@dataclass(frozen=True)
class PanelScenarioSpec:
    scenario_id: str
    name: str
    declared_effect: float
    description: str
    build: Callable[[], PanelDGP]


def _panel(
    scenario_id: str,
    adoption: Adoption,
    effect: DynamicEffect,
    *,
    slope_sd: float = 0.0,
    time_slope: float = 0.1,
    error_rho: float = 0.0,
) -> Callable[[], PanelDGP]:
    def build() -> PanelDGP:
        return PanelDGP(
            unit_generator=UnitGenerator(covariate_names=_COVARIATES),
            unit_effects=NormalUnitEffects(level_sd=1.0, slope_sd=slope_sd),
            time_path=LinearTimePath(slope=time_slope),
            adoption=adoption,
            effect=effect,
            scenario_id=scenario_id,
            noise_sd=1.0,
            error_rho=error_rho,
        )

    return build


PANEL_SCENARIOS: dict[str, PanelScenarioSpec] = {
    "P1": PanelScenarioSpec(
        scenario_id="P1",
        name="Clean 2x2",
        declared_effect=1.0,
        description="Uniform adoption mid-panel, zero slope dispersion, constant "
        "effect. Parallel trends holds exactly by construction, so every DiD "
        "estimator must recover truth to Monte Carlo error. The panel analogue "
        "of scenario A.",
        build=_panel(
            "P1",
            UniformAdoption(start=_UNIFORM_START, share=0.5),
            ConstantDynamicEffect(tau=1.0),
        ),
    ),
    "P2": PanelScenarioSpec(
        scenario_id="P2",
        name="Null effect",
        declared_effect=0.0,
        description="As P1 with tau = 0. Any systematic finding is a false "
        "positive. The role scenario B plays.",
        build=_panel(
            "P2",
            UniformAdoption(start=_UNIFORM_START, share=0.5),
            ConstantDynamicEffect(tau=0.0),
        ),
    ),
    "P3": PanelScenarioSpec(
        scenario_id="P3",
        name="Staggered, homogeneous effect",
        declared_effect=1.0,
        description="Staggered adoption with the same effect in every cohort. "
        "TWFE and Callaway-Sant'Anna must agree; this is the control condition "
        "that makes P4 interpretable.",
        build=_panel(
            "P3",
            StaggeredAdoption(periods=_COHORT_PERIODS, never_share=_NEVER_SHARE),
            ConstantDynamicEffect(tau=1.0),
        ),
    ),
    "P4": PanelScenarioSpec(
        scenario_id="P4",
        name="Staggered, effects growing with exposure",
        declared_effect=2.8361,
        description="Staggered adoption with an effect that ramps with exposure, "
        "so effects differ across cohorts at any calendar period and "
        "already-treated units serve as controls with negative weights. TWFE is "
        "badly biased while Callaway-Sant'Anna recovers truth. The declared "
        "effect here is a measured cohort-size-weighted average, not an input.",
        build=_panel(
            "P4",
            StaggeredAdoption(periods=_COHORT_PERIODS, never_share=_NEVER_SHARE),
            RampDynamicEffect(tau=1.0),
        ),
    ),
    "P5": PanelScenarioSpec(
        scenario_id="P5",
        name="Parallel trends violated",
        declared_effect=1.0,
        description="Steeper-trending units adopt earlier. Every DiD estimator "
        "is biased here, Callaway-Sant'Anna included, because it assumes "
        "parallel trends too. The pre-trend check must FAIL.",
        build=_panel(
            "P5",
            TrendSelectedAdoption(periods=_COHORT_PERIODS, never_share=_NEVER_SHARE),
            ConstantDynamicEffect(tau=1.0),
            slope_sd=0.3,
        ),
    ),
}


def build_panel_scenario(scenario_id: str) -> PanelDGP:
    """Instantiate a panel scenario's DGP."""
    if scenario_id not in PANEL_SCENARIOS:
        raise KeyError(
            f"unknown panel scenario {scenario_id!r}; available: "
            f"{sorted(PANEL_SCENARIOS)}"
        )
    return PANEL_SCENARIOS[scenario_id].build()
