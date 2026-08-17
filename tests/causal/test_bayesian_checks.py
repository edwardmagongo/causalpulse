"""MCMC diagnostics: r_hat_check, effective_sample_size_check, divergent_transitions_check."""

from __future__ import annotations

import numpy as np
from cp_causal.bayesian_checks import effective_sample_size_check, r_hat_check


def _converged_chains(
    n_chains: int = 4, n_draws: int = 200, seed: int = 0
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Same distribution, independent draws per chain: R-hat should sit near 1.0.
    return {"mu": rng.normal(size=(n_chains, n_draws))}


def _diverged_chains(
    n_chains: int = 4, n_draws: int = 200, seed: int = 0
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Each chain centred on a different value: chains disagree, R-hat should be large.
    offsets = np.arange(n_chains).reshape(n_chains, 1) * 10.0
    return {"mu": offsets + rng.normal(size=(n_chains, n_draws))}


def test_r_hat_check_passes_on_well_mixed_chains():
    result = r_hat_check(_converged_chains())
    assert result.status == "PASS"
    assert result.statistic is not None
    assert result.statistic < 1.01


def test_r_hat_check_fails_on_chains_that_have_not_converged():
    result = r_hat_check(_diverged_chains())
    assert result.status == "FAIL"
    assert result.statistic is not None
    assert result.statistic > 1.01


def test_r_hat_check_reports_the_worst_parameter_not_an_arbitrary_one():
    samples = _converged_chains()
    samples["donor_coef"] = _diverged_chains(seed=1)["mu"][:, :, None]  # extra event dim
    result = r_hat_check(samples)
    assert result.status == "FAIL"


def test_effective_sample_size_check_passes_on_well_mixed_chains():
    result = effective_sample_size_check(_converged_chains(n_draws=1000))
    assert result.status == "PASS"
    assert result.statistic is not None
    assert result.statistic >= 200.0


def test_effective_sample_size_check_fails_on_a_stuck_chain():
    # A random walk with tiny steps: strongly autocorrelated between
    # consecutive draws (unlike i.i.d. noise around a constant, which
    # has zero autocorrelation regardless of its scale and would NOT
    # produce a low ESS) -- this genuinely mimics a poorly-mixing chain.
    rng = np.random.default_rng(0)
    stuck = np.cumsum(rng.normal(scale=0.01, size=(4, 200)), axis=1)
    result = effective_sample_size_check({"mu": stuck})
    assert result.status == "FAIL"
