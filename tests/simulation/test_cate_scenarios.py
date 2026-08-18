"""CATE scenario components and the CATE_SCENARIOS registry."""

from __future__ import annotations

import numpy as np
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
