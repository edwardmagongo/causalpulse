"""BSTS: model, MCMC runner, posterior summary, the registered estimator.

Every test here uses FAST_MCMC_CONFIG -- tiny chains that exercise code paths
and array shapes, not trustworthy posteriors. Recovery accuracy is
test_bsts_recovery.py's job (slow, PRODUCTION_MCMC_CONFIG).
"""

from __future__ import annotations

import numpy as np
from cp_causal._panel import donor_matrix, panel_arrays
from cp_causal.estimators.bsts import FAST_MCMC_CONFIG, MCMCConfig, _run_mcmc
from cp_simulation.donor_scenarios import DONOR_N_PERIODS, DONOR_N_UNITS, build_donor_scenario


def _generate(scenario_id: str, seed: int = 0):
    return build_donor_scenario(scenario_id).generate(
        DONOR_N_UNITS, DONOR_N_PERIODS, np.random.default_rng(seed)
    )


def test_run_mcmc_returns_samples_shaped_by_the_fast_config():
    panel = _generate("S1")
    parr = panel_arrays(panel.dataset)
    treated, donors, pre_mask = donor_matrix(parr)
    treated_pre, donor_pre = treated[pre_mask], donors[:, pre_mask]

    donor_mean = donor_pre.mean(axis=1, keepdims=True)
    donor_sd = donor_pre.std(axis=1, keepdims=True) + 1e-9
    donor_pre_std = (donor_pre - donor_mean) / donor_sd
    treated_mean, treated_sd = treated_pre.mean(), treated_pre.std() + 1e-9
    treated_pre_std = (treated_pre - treated_mean) / treated_sd

    mcmc = _run_mcmc(donor_pre_std, treated_pre_std, treated_pre.shape[0], FAST_MCMC_CONFIG, 0)
    samples = mcmc.get_samples()

    assert samples["donor_coef"].shape == (FAST_MCMC_CONFIG.num_samples, donor_pre.shape[0])
    assert samples["level0"].shape == (FAST_MCMC_CONFIG.num_samples,)
    assert samples["eta"].shape == (FAST_MCMC_CONFIG.num_samples, treated_pre.shape[0])


def test_mcmc_config_is_a_frozen_dataclass():
    config = MCMCConfig(num_warmup=10, num_samples=10, num_chains=1)
    assert config.target_accept_prob == 0.95
    try:
        config.num_warmup = 99  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised
