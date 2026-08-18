"""CATE scenario components and the CATE_SCENARIOS registry."""

from __future__ import annotations

import numpy as np
from cp_simulation.cate_scenarios import CATE_N_UNITS, CATE_SCENARIOS, build_cate_scenario
from cp_simulation.components import MultivariateHeterogeneousEffect


def test_multivariate_heterogeneous_effect_combines_continuous_and_binary_drivers():
    # x0 continuous, x1 binary, x2 continuous (clipped), x3-x5 no effect on tau.
    effect = MultivariateHeterogeneousEffect(
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
    covariates = np.array(
        [
            [1.0, 1.0, 0.5, 0.0, 0.0, 0.0],   # x0>0 & x1=1 -> interaction fires
            [-1.0, 1.0, 0.5, 0.0, 0.0, 0.0],  # x0<0 & x1=1 -> interaction does not fire
            [1.0, 0.0, 0.5, 0.0, 0.0, 0.0],   # x1=0 -> interaction does not fire
        ]
    )
    tau = effect.tau_values(covariates)
    assert tau[0] == 0.05 + 0.08 * 1.0 + 0.03 * 0.5
    assert tau[1] == 0.05 + 0.03 * 0.5
    assert tau[2] == 0.05 + 0.03 * 0.5


def test_multivariate_heterogeneous_effect_clips_the_secondary_driver():
    effect = MultivariateHeterogeneousEffect(
        base_tau=0.0,
        continuous_index=0,
        continuous_coefficient=0.0,
        binary_interaction_index=1,
        binary_interaction_continuous_index=0,
        binary_interaction_coefficient=0.0,
        secondary_continuous_index=2,
        secondary_coefficient=1.0,
        secondary_clip=1.0,
    )
    covariates = np.array([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])  # secondary driver way above the clip
    tau = effect.tau_values(covariates)
    assert tau[0] == 1.0  # clipped to secondary_clip, not 5.0


def test_rich_heterogeneity_scenario_measured_truth_has_real_spread():
    # A sanity check, not the calibrated recovery harness (that's Task 14).
    # The framework validates itself per parent spec §4.5: this scenario's
    # true_cate must actually vary meaningfully, or it cannot test per-unit
    # CATE recovery at all.
    dgp = build_cate_scenario("RICH")
    simulated = dgp.generate(CATE_N_UNITS, np.random.default_rng(0))
    true_cate = simulated.truth.true_cate
    std = true_cate.std()
    assert std > 0.02, f"RICH scenario's true_cate is nearly flat: std={std}"


def test_null_heterogeneity_scenario_measured_truth_is_flat():
    dgp = build_cate_scenario("NULL")
    simulated = dgp.generate(CATE_N_UNITS, np.random.default_rng(0))
    true_cate = simulated.truth.true_cate
    assert true_cate.std() < 0.005, f"NULL scenario's true_cate is not flat: std={true_cate.std()}"


def test_every_cate_scenario_is_covered_by_this_registry():
    assert set(CATE_SCENARIOS) == {"RICH", "NULL", "NOISY"}
