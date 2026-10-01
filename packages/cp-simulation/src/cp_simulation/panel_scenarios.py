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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from cp_simulation.components import UnitGenerator
from cp_simulation.panel_binary import BinaryPanelDGP
from cp_simulation.panel_components import (
    Adoption,
    ConstantDynamicEffect,
    DynamicEffect,
    LevelSelectedAdoption,
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

# This scenario's own protocol, deliberately separate from PANEL_N_* above:
# a hierarchy over cohorts needs more cohorts than eight periods can hold,
# and PANEL_N_PERIODS is the protocol every Phase 3 measurement was taken
# at. Following CATE_N_UNITS' precedent from Phase 6 rather than widening a
# constant other measurements depend on.
HIER_N_UNITS = 600
HIER_N_PERIODS = 20

HIER_SEEDS = 20
_HIER_COHORT_PERIODS = (4, 6, 8, 10, 12, 14, 16)

# Panel scenarios that fix their own size, so the benchmark grid runs them as
# fixed-protocol cells rather than crossing them with PanelAxes. P7 alone: its
# cross-cohort variance was shown identifiable at this protocol, and it needs at
# least 16 periods, which the panel arm's (7, 10) cannot give it.
PANEL_PROTOCOLS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"P7": (HIER_N_UNITS, HIER_N_PERIODS)}
)


@dataclass(frozen=True)
class PanelScenarioSpec:
    scenario_id: str
    name: str
    declared_effect: float
    description: str
    build: Callable[[], PanelDGP | BinaryPanelDGP]


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
    # P6 uses the SAME time_slope every other panel scenario uses -- no special
    # parameter. (Corrected 2026-08-14: an earlier version pinned this at 0.3,
    # on a since-falsified claim that the shared slope of 0.1 made the bias
    # vanish. It does not -- measured at 800 units over 120 seeds against this
    # scenario's actual unit-level parameters, the bias is +0.0360 (18.3 MCSE) at
    # the shared slope of 0.1, larger than the +0.0139 (7.0 MCSE) the discarded
    # 0.3 parameterisation produced. The bias is non-monotonic in time_slope and
    # even changes sign across the range; it is not a straight line to zero.)
    # What IS load-bearing: LevelSelectedAdoption. Link-function bias needs the
    # two arms to sit at different points on the logistic curve, which is what
    # selecting on level does and random adoption does not. Switching this to
    # UniformAdoption would leave a preset that appears to demonstrate the bias
    # and does not -- see test_random_adoption_produces_no_link_bias.
    "P6": PanelScenarioSpec(
        scenario_id="P6",
        name="Binary panel",
        declared_effect=0.1061,
        description="Binary outcomes with the effect applied on the logit "
        "scale and level-selected adoption. Parallel trends holds exactly on "
        "the logit scale but fails on the probability scale, so a 2x2 DiD "
        "reporting percentage points is biased with no confounding -- the "
        "bias IS a differential pre-trend, just one that only shows up on "
        "the probability scale the estimator reports on. Measures and "
        "documents that link-function bias rather than avoiding it.",
        build=lambda: BinaryPanelDGP(
            unit_generator=UnitGenerator(covariate_names=_COVARIATES),
            unit_effects=NormalUnitEffects(level_sd=0.8, slope_sd=0.0),
            time_path=LinearTimePath(slope=0.1),
            adoption=LevelSelectedAdoption(start=_UNIFORM_START),
            scenario_id="P6",
            intercept=-1.5,
            tau_logit=0.45,
        ),
    ),
    "P7": PanelScenarioSpec(
        scenario_id="P7",
        name="Many cohorts, dispersed effects",
        declared_effect=6.7358,
        description="Seven adoption cohorts with effects that differ across "
        "cohorts, generated at this scenario's own longer protocol: "
        "HIER_N_UNITS=600, HIER_N_PERIODS=20. Exists so a hierarchical "
        "model's cross-cohort variance is identifiable: the three cohorts of "
        "P3/P4 leave sigma_tau largely prior-driven. The declared effect is "
        "the measured mean of true_att_overall across HIER_SEEDS=20 seeds -- "
        "6.735807, sd 0.005855, MCSE 0.001309 -- the same quantity P4's "
        "2.8361 was measured as, not a plain per-cohort-size weighted "
        "average of true_att_by_cohort (which comes out to 6.01 here and "
        "would similarly diverge from P4's declared label if used there).",
        build=_panel(
            "P7",
            StaggeredAdoption(periods=_HIER_COHORT_PERIODS, never_share=_NEVER_SHARE),
            RampDynamicEffect(tau=1.0),
        ),
    ),
}


def build_panel_scenario(scenario_id: str) -> PanelDGP | BinaryPanelDGP:
    """Instantiate a panel scenario's DGP."""
    if scenario_id not in PANEL_SCENARIOS:
        raise KeyError(
            f"unknown panel scenario {scenario_id!r}; available: "
            f"{sorted(PANEL_SCENARIOS)}"
        )
    return PANEL_SCENARIOS[scenario_id].build()
