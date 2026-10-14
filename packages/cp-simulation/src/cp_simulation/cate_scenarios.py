"""CATE scenario presets: cross-sectional data with genuine multi-covariate
heterogeneity, distinct from both `SCENARIOS` (A-J) and the panel/donor-pool
registries -- these are not points on either existing grid.
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

# The coefficients are set from a measured detection floor, not chosen by eye.
# These scenarios score a binary (logistic) outcome, so the per-unit noise is
# sigma ~ 0.50, and a differential-calibration test at CATE_N_UNITS = 2000
# resolves an effect spread only in proportion to sd(tau) * sqrt(n) / sigma.
# An earlier draft used 0.08 / 0.03, giving sd(tau) ~ 0.038: measured, that is
# below the floor for ANY such test -- substituting the TRUE per-unit tau for
# an estimate (an oracle no estimator can beat) reached only t ~ +1.68 and
# detected heterogeneity on 7 of 20 seeds, and the real forest detected it on 0
# of 12. A scenario whose stated job is "what per-unit CATE and subgroup
# recovery get scored against" cannot carry heterogeneity that is undetectable
# in principle.
#
# At 0.25 / 0.10 the same measurement gives sd(tau) ~ 0.123, an oracle t of
# +5.54 detecting on 20 of 20 seeds, and the real forest detecting on 12 of 12
# with its out-of-bag CATE correlating +0.82 with the truth (it was +0.38 at
# 0.08 / 0.03 -- the forest recovers the structure better when there is more of
# it to find, so the two effects compound). The oracle figures use plain OLS:
# the true tau is not a function of the outcome, so the out-of-bag bias and
# variance corrections in `heterogeneity_checks.calibration_test` do not apply
# to it. NULL is deliberately left untouched: the false-positive guard this
# phase exists to prove keeps exactly its old meaning.
#
# Two consequences worth stating, since Tasks 12-16 inherit them. The true CATE
# range widens from [+0.020, +0.160] to [-0.050, +0.400], so 24% of units now
# have genuinely NEGATIVE individual effects where none did before -- a
# treatment that helps most units and harms some, which is the realistic case
# subgroup discovery exists for, but a change in character rather than only in
# scale. And the true ATE rises from 0.065 to 0.096 (seed 0, n=2000).
_RICH_EFFECT = MultivariateHeterogeneousEffect(
    base_tau=0.05,
    continuous_index=0,
    continuous_coefficient=0.0,
    binary_interaction_index=1,
    binary_interaction_continuous_index=0,
    binary_interaction_coefficient=0.25,
    secondary_continuous_index=2,
    secondary_coefficient=0.10,
    secondary_clip=1.0,
)

_NULL_EFFECT = ConstantEffect(tau=0.05)


@dataclass(frozen=True)
class CATEScenarioSpec:
    scenario_id: str
    name: str
    description: str
    build: Callable[[], CrossSectionDGP]
    # Whether the scenario's true CATE is meant to vary across units. Declared
    # rather than measured: NULL's realised truth has sd ~3e-17 (rounding of a
    # constant), so a variance test cannot tell it apart without a tolerance
    # chosen to fit. `tests/simulation/test_cate_scenarios.py` checks the
    # declaration against realised data. The benchmark's CATE arm reads it to
    # decide whether correlation and calibration slope are defined.
    heterogeneous: bool


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
        heterogeneous=True,
    ),
    "NULL": CATEScenarioSpec(
        scenario_id="NULL",
        name="Null heterogeneity",
        description=(
            "Effect is constant (tau=0.05) across every covariate. This is "
            "the scenario the omnibus calibration test must correctly pass "
            "on -- reporting 'no reliable evidence of heterogeneity' and no "
            "subgroups -- the guard this phase exists to demonstrate, not "
            "just the forest's ability to find signal when it's there."
        ),
        build=lambda: CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_CATE_COVARIATES),
            baseline=_baseline(),
            effect=_NULL_EFFECT,
            assignment=RandomAssignment(share=0.5),
            scenario_id="NULL",
        ),
        heterogeneous=False,
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
        heterogeneous=True,
    ),
}


def build_cate_scenario(scenario_id: str) -> CrossSectionDGP:
    return CATE_SCENARIOS[scenario_id].build()
