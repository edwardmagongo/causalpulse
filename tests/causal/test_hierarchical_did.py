"""Hierarchical Bayesian DiD: model, posterior arithmetic, and the estimator."""

from __future__ import annotations

import numpy as np
from cp_causal._mcmc import run_nuts
from cp_causal.estimators.hierarchical_did import (
    HIER_FAST_MCMC_CONFIG,
    _model,
)


def _toy_panel(n_units: int = 40, n_periods: int = 6, seed: int = 0):
    """A tiny two-cohort panel with a known effect, for shape tests only."""
    rng = np.random.default_rng(seed)
    cohort_of_unit = np.where(
        np.arange(n_units) % 4 == 0, 0, np.where(np.arange(n_units) % 4 == 1, 3, 4)
    )
    unit_index = np.repeat(np.arange(n_units), n_periods)
    time_index = np.tile(np.arange(n_periods), n_units)
    adopt = cohort_of_unit[unit_index]
    treated_mask = (adopt > 0) & (time_index >= adopt)
    outcome = rng.normal(size=n_units * n_periods) + 2.0 * treated_mask
    codes = {g: i for i, g in enumerate(sorted(set(cohort_of_unit.tolist()) - {0}))}
    cohort_index = np.array([codes.get(g, -1) for g in adopt])
    return outcome, unit_index, time_index, cohort_index, treated_mask, len(codes)


def test_model_samples_and_exposes_the_expected_sites():
    outcome, unit_index, time_index, cohort_index, treated_mask, n_cohorts = _toy_panel()

    mcmc = run_nuts(
        _model,
        HIER_FAST_MCMC_CONFIG,
        0,
        outcome_std=outcome,
        unit_index=unit_index,
        time_index=time_index,
        cohort_index=cohort_index,
        treated_mask=treated_mask,
        n_units=int(unit_index.max()) + 1,
        n_periods=int(time_index.max()) + 1,
        n_cohorts=n_cohorts,
    )
    samples = mcmc.get_samples()

    assert "mu_tau" in samples
    assert "sigma_tau" in samples
    assert samples["tau_g"].shape[1] == n_cohorts
    assert np.all(samples["sigma_tau"] >= 0.0)


def test_fast_config_has_at_least_two_chains():
    # gelman_rubin (r_hat_check's dependency) hard-asserts >= 2 chains, so a
    # 1-chain config crashes the moment fit() reaches its assumption checks.
    # Phase 5 found this only when the integration task first wired the two
    # together.
    assert HIER_FAST_MCMC_CONFIG.num_chains >= 2


def test_never_treated_rows_do_not_contribute_a_treatment_effect():
    """cohort_index is -1 for never-treated units. A gather at a negative
    index does not raise in JAX -- it silently wraps (plain negative
    indexing reads the last element; here the model additionally clips to
    0, which is cohort 0, not an error either way). So if the model read
    `tau_g[cohort_index]` without also gating on `treated_mask`, a
    never-treated row would silently pick up *some* cohort's effect instead
    of zero. This test catches exactly that: it makes cohort 0 real (so the
    clipped/-wrapped index is valid and non-trivial), draws one forward
    trace, and requires every never-treated row's implied mean to equal
    `alpha[unit] + gamma[time]` with *no* added effect term, regardless of
    what tau_g holds.
    """
    import jax
    from numpyro.handlers import seed, trace

    n_units, n_periods = 12, 4
    unit_index = np.repeat(np.arange(n_units), n_periods)
    time_index = np.tile(np.arange(n_periods), n_units)
    # Half the units are never-treated (cohort_index == -1, treated_mask
    # always False); the rest belong to a single real cohort 0.
    cohort_of_unit = np.where(np.arange(n_units) < 6, -1, 0)
    cohort_index = cohort_of_unit[unit_index]
    adopt_time = np.where(cohort_of_unit == -1, 10_000, 2)
    treated_mask = time_index >= adopt_time[unit_index]
    assert not treated_mask[cohort_index == -1].any()

    outcome = np.zeros(n_units * n_periods)
    n_cohorts = 1

    traced = trace(seed(_model, jax.random.PRNGKey(0))).get_trace(
        outcome_std=outcome,
        unit_index=unit_index,
        time_index=time_index,
        cohort_index=cohort_index,
        treated_mask=treated_mask,
        n_units=n_units,
        n_periods=n_periods,
        n_cohorts=n_cohorts,
    )
    tau_g = np.asarray(traced["tau_g"]["value"])
    assert not np.allclose(tau_g, 0.0), "test is vacuous if the sampled cohort effect is ~0"

    alpha = np.asarray(traced["sigma_alpha"]["value"]) * np.asarray(traced["z_alpha"]["value"])
    gamma = np.asarray(traced["sigma_gamma"]["value"]) * np.asarray(traced["z_gamma"]["value"])
    baseline_mean = alpha[unit_index] + gamma[time_index]
    actual_mean = np.asarray(traced["obs"]["fn"].loc)

    never_treated_rows = np.isin(unit_index, np.where(cohort_of_unit == -1)[0])
    assert np.allclose(actual_mean[never_treated_rows], baseline_mean[never_treated_rows])
    # Sanity check the harness itself: the treated cohort's rows *should*
    # differ from the no-effect baseline (else the assertion above would
    # pass trivially even with a broken model).
    assert not np.allclose(actual_mean[~never_treated_rows], baseline_mean[~never_treated_rows])
