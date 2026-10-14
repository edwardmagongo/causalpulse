"""CATE scenario components and the CATE_SCENARIOS registry."""

from __future__ import annotations

import numpy as np
import pytest
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
    # The framework validates itself: this scenario's
    # true_cate must actually vary meaningfully, or it cannot test per-unit
    # CATE recovery at all.
    #
    # The floor is a measured detection threshold, not a round number. These
    # scenarios use a binary outcome (sigma ~ 0.50) at n = 2000, where an
    # effect spread below ~0.05 is undetectable even by an oracle holding the
    # TRUE per-unit tau -- an earlier draft of this scenario sat at 0.038 and
    # the real forest detected its heterogeneity on 0 of 12 seeds. See
    # `cate_scenarios._RICH_EFFECT`'s comment for the full measurement.
    # Measured here: 0.1225, range [0.1187, 0.1263] over 20 seeds.
    dgp = build_cate_scenario("RICH")
    simulated = dgp.generate(CATE_N_UNITS, np.random.default_rng(0))
    true_cate = simulated.truth.true_cate
    std = true_cate.std()
    assert std > 0.08, (
        f"RICH scenario's true_cate spread is {std:.4f}, below the ~0.05 floor at "
        f"which a differential-calibration test can detect heterogeneity at all "
        f"on a binary outcome at n={CATE_N_UNITS}. Subgroup discovery is gated on "
        f"that test, so a spread this small makes the whole subgroup path dead code."
    )


def test_null_heterogeneity_scenario_measured_truth_is_flat():
    dgp = build_cate_scenario("NULL")
    simulated = dgp.generate(CATE_N_UNITS, np.random.default_rng(0))
    true_cate = simulated.truth.true_cate
    assert true_cate.std() < 0.005, f"NULL scenario's true_cate is not flat: std={true_cate.std()}"


def test_noisy_scenario_has_more_covariates_than_true_drivers():
    dgp = build_cate_scenario("NOISY")
    simulated = dgp.generate(CATE_N_UNITS, np.random.default_rng(0))
    assert simulated.dataset.schema.covariates == (
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"
    )
    # Only x0/x1 (interaction) and x2 (secondary) drive tau; x3-x11 do not.
    # Same measured detection floor as RICH, whose effect this scenario reuses.
    true_cate = simulated.truth.true_cate
    assert true_cate.std() > 0.08


def test_every_cate_scenario_is_covered_by_this_registry():
    assert set(CATE_SCENARIOS) == {"RICH", "NULL", "NOISY"}


def test_only_null_is_declared_constant():
    constant = {sid for sid, spec in CATE_SCENARIOS.items() if not spec.heterogeneous}

    assert constant == {"NULL"}


@pytest.mark.parametrize("scenario_id", sorted(CATE_SCENARIOS))
def test_the_heterogeneity_declaration_matches_the_realised_truth(scenario_id):
    # The benchmark turns CATE correlation and calibration slope off on the
    # declaration alone, so the declaration must agree with the data. Measured
    # 2026-09-27: RICH/NOISY true-CATE sd 0.119-0.121, NULL ~3e-17. The two
    # bounds sit orders of magnitude from both, so neither is tuned to a seed.
    spec = CATE_SCENARIOS[scenario_id]
    for seed in range(3):
        simulated = spec.build().generate(
            n_units=CATE_N_UNITS, rng=np.random.default_rng(seed)
        )
        spread = float(np.std(simulated.truth.true_cate))
        if spec.heterogeneous:
            assert spread > 0.05, (scenario_id, seed, spread)
        else:
            assert spread < 1e-12, (scenario_id, seed, spread)
