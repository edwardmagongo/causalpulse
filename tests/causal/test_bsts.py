"""BSTS: model, MCMC runner, posterior summary, the registered estimator.

Every test here uses FAST_MCMC_CONFIG -- tiny chains that exercise code paths
and array shapes, not trustworthy posteriors. Recovery accuracy is
test_bsts_recovery.py's job (slow, PRODUCTION_MCMC_CONFIG).
"""

from __future__ import annotations

import numpy as np
from cp_causal._panel import donor_matrix, panel_arrays
from cp_causal.estimators.bsts import FAST_MCMC_CONFIG, MCMCConfig, _run_mcmc, summarize_posterior
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


def _synthetic_posterior(n_draws: int, t_pre: int, n_donors: int, n_post: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "eta": rng.normal(scale=0.1, size=(n_draws, t_pre)),
        "zeta": rng.normal(scale=0.05, size=(n_draws, t_pre)),
        "level0": rng.normal(size=n_draws),
        "slope0": rng.normal(scale=0.1, size=n_draws),
        "sigma_level": np.full(n_draws, 0.1),
        "sigma_slope": np.full(n_draws, 0.05),
        "donor_coef": rng.normal(scale=0.1, size=(n_draws, n_donors)),
    }


def test_summarize_posterior_shapes_match_pre_plus_post_periods():
    n_draws, t_pre, n_donors, n_post = 200, 17, 17, 7
    samples = _synthetic_posterior(n_draws, t_pre, n_donors, n_post)
    donor_pre_std = np.zeros((n_donors, t_pre))
    donor_post_std = np.zeros((n_donors, n_post))
    treated_pre = np.zeros(t_pre)
    treated_post = np.zeros(n_post)

    summary = summarize_posterior(
        samples, donor_pre_std, donor_post_std, treated_pre, treated_post,
        treated_mean=0.0, treated_sd=1.0, ci_level=0.95,
    )

    assert summary.mu_path.shape == (t_pre + n_post,)
    assert summary.lower.shape == (t_pre + n_post,)
    assert summary.upper.shape == (t_pre + n_post,)
    assert summary.att_draws.shape == (n_draws,)
    assert summary.donor_coef_mean.shape == (n_donors,)
    assert np.all(summary.lower <= summary.mu_path)
    assert np.all(summary.mu_path <= summary.upper)


def test_summarize_posterior_recovers_a_known_constant_effect():
    # donor_coef = 0 and every trend parameter fixed at 0 except level0 = 0:
    # the fitted path is exactly 0 in original units. A treated series that
    # is exactly +2.0 above that in the post-period should read back an ATT
    # of +2.0 with no draw-to-draw spread.
    n_draws, t_pre, n_donors, n_post = 50, 10, 5, 3
    samples = {
        "eta": np.zeros((n_draws, t_pre)),
        "zeta": np.zeros((n_draws, t_pre)),
        "level0": np.zeros(n_draws),
        "slope0": np.zeros(n_draws),
        "sigma_level": np.zeros(n_draws),
        "sigma_slope": np.zeros(n_draws),
        "donor_coef": np.zeros((n_draws, n_donors)),
    }
    donor_pre_std = np.zeros((n_donors, t_pre))
    donor_post_std = np.zeros((n_donors, n_post))
    treated_pre = np.zeros(t_pre)
    treated_post = np.full(n_post, 2.0)

    summary = summarize_posterior(
        samples, donor_pre_std, donor_post_std, treated_pre, treated_post,
        treated_mean=0.0, treated_sd=1.0, ci_level=0.95,
    )

    assert np.allclose(summary.att_draws, 2.0)
