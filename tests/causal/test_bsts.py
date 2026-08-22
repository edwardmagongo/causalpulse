"""BSTS: model, MCMC runner, posterior summary, the registered estimator.

Every test here uses FAST_MCMC_CONFIG -- tiny chains that exercise code paths
and array shapes, not trustworthy posteriors. Recovery accuracy is
test_bsts_recovery.py's job (slow, PRODUCTION_MCMC_CONFIG).

The tests that assert on a fitted S1 result share one module-scoped fit
(`s1_fit`) instead of each running their own. What a FAST_MCMC_CONFIG fit
actually costs is compilation, not sampling: measured warm, a run of this
config spends ~1.6s compiling the NUTS kernel against ~0.8s sampling it, and
XLA compiles once per `MCMC` instance, so N separate fits of one scenario pay
that ~1.6s N times to re-derive an identical result. Shrinking the config
buys back only part of the 0.8s and costs fidelity (at half the draws the
r_hat/ESS diagnostics degrade into inf/nan); dropping the duplicate fits
buys the whole compile. `EffectEstimate` is frozen and each of these tests
reads a different field of it, so there is no state to leak between them.
"""

from __future__ import annotations

import numpy as np
import pytest
from cp_causal._panel import donor_matrix, panel_arrays
from cp_causal.estimators.bsts import (
    BSTS,
    FAST_MCMC_CONFIG,
    MCMCConfig,
    _run_mcmc,
    summarize_posterior,
)
from cp_causal.registry import all_estimators
from cp_causal.router import eligible_estimators
from cp_core.values import EstimatorConfig
from cp_simulation.donor_scenarios import DONOR_N_PERIODS, DONOR_N_UNITS, build_donor_scenario
from cp_simulation.scenarios import build_scenario


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
    # get_samples() (no group_by_chain) concatenates draws across chains, so
    # the leading dimension is num_chains * num_samples, not num_samples alone
    # -- found during Task 8, when FAST_MCMC_CONFIG moved from 1 to 2 chains.
    total_draws = FAST_MCMC_CONFIG.num_chains * FAST_MCMC_CONFIG.num_samples

    assert samples["donor_coef"].shape == (total_draws, donor_pre.shape[0])
    assert samples["level0"].shape == (total_draws,)
    assert samples["eta"].shape == (total_draws, treated_pre.shape[0])


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
    treated_post = np.zeros(n_post)

    summary = summarize_posterior(
        samples, donor_pre_std, donor_post_std, treated_post,
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
    treated_post = np.full(n_post, 2.0)

    summary = summarize_posterior(
        samples, donor_pre_std, donor_post_std, treated_post,
        treated_mean=0.0, treated_sd=1.0, ci_level=0.95,
    )

    assert np.allclose(summary.att_draws, 2.0)


CONFIG = EstimatorConfig(random_seed=0, options={"mcmc_config": FAST_MCMC_CONFIG})


@pytest.fixture(scope="module")
def s1_fit():
    """One FAST_MCMC_CONFIG fit of S1, shared by every test that only reads
    fields off the result -- see the module docstring for why it is shared."""
    return BSTS().fit(_generate("S1").dataset, CONFIG)


def test_bsts_is_registered_under_its_own_name():
    from cp_causal.registry import get_estimator

    assert get_estimator("bsts") is BSTS


def test_bsts_requirements_match_synthetic_controls_donor_pool_shape():
    from cp_causal.estimators.synthetic_control import SyntheticControl

    assert BSTS.requirements.needs_panel == SyntheticControl.requirements.needs_panel
    assert BSTS.requirements.max_treated_units == SyntheticControl.requirements.max_treated_units
    assert BSTS.requirements.min_control_units == SyntheticControl.requirements.min_control_units


def test_bsts_fit_returns_a_posterior_based_effect_estimate(s1_fit):
    assert s1_fit.ci_basis == "posterior"
    assert s1_fit.p_value is None
    assert s1_fit.ci.lower <= s1_fit.point <= s1_fit.ci.upper
    assert "p_effect_positive" in s1_fit.diagnostics
    assert "p_effect_above_threshold" in s1_fit.diagnostics
    assert "donor_coef" in s1_fit.diagnostics


def test_bsts_fit_reports_four_assumption_checks(s1_fit):
    names = {c.name for c in s1_fit.assumptions}
    assert names == {
        "r_hat",
        "effective_sample_size",
        "divergent_transitions",
        "sufficient_pre_periods",
    }


def test_bsts_fit_populates_counterfactual_path_for_every_period(s1_fit):
    path = s1_fit.counterfactual

    assert path is not None
    assert path.periods.shape == path.observed.shape
    assert path.predicted_absent_treatment.shape == path.periods.shape


def test_bsts_fit_fails_sufficient_pre_periods_on_a_short_pre_period_scenario():
    panel = _generate("S5")
    result = BSTS().fit(panel.dataset, CONFIG)

    status = {c.name: c.status for c in result.assumptions}
    assert status["sufficient_pre_periods"] == "FAIL"


def test_importing_cp_causal_registers_bsts():
    # test_bsts_is_registered_under_its_own_name above is self-fulfilling --
    # it passes only because this file already imports
    # cp_causal.estimators.bsts directly above, not because `import cp_causal`
    # alone triggers registration. This test pins the real contract: that
    # `cp_causal/__init__.py` imports the bsts module (for its @register side
    # effect) so real callers who only do `import cp_causal` still see it in
    # the registry. Mirrors test_importing_cp_causal_registers_every_panel_
    # estimator (test_panel_recovery.py) and test_importing_cp_causal_
    # registers_every_cross_sectional_estimator (test_recovery.py).
    import cp_causal  # noqa: F401

    assert "bsts" in set(all_estimators())


def test_bsts_is_eligible_on_a_donor_pool_panel_dataset():
    panel = _generate("S1")

    assert "bsts" in set(eligible_estimators(panel.dataset))


def test_bsts_is_excluded_on_a_cross_sectional_dataset():
    simulated = build_scenario("A").generate(n_units=200, rng=np.random.default_rng(0))

    assert "bsts" not in set(eligible_estimators(simulated.dataset))
