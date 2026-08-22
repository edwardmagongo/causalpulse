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


def test_cohort_cell_counts_count_treated_cells_not_treated_units():
    """Weights must match `true_att_overall`, which averages over every
    treated (unit, period) CELL -- so a cohort's weight is units x
    post-periods, not units. Measured on P4: cell weighting reproduces its
    registered 2.8356; unit weighting gives 2.5028.
    """
    from cp_causal._panel import panel_arrays
    from cp_causal.estimators.hierarchical_did import build_model_inputs
    from cp_simulation.panel_scenarios import build_panel_scenario

    dgp = build_panel_scenario("P3")
    simulated = dgp.generate(200, 8, np.random.default_rng(0))
    inputs = build_model_inputs(panel_arrays(simulated.dataset))

    assert inputs.cohort_cell_counts.shape == (inputs.n_cohorts,)
    # Cells, not units: each count is that cohort's treated-ROW total.
    for code in range(inputs.n_cohorts):
        expected = int(((inputs.cohort_index == code) & inputs.treated_mask).sum())
        assert int(inputs.cohort_cell_counts[code]) == expected
    # The total must equal the treated-row count exactly. A unit-based count
    # would be smaller by roughly the post-period count -- the 2.8356 vs
    # 2.5028 gap on P4.
    assert int(inputs.cohort_cell_counts.sum()) == int(inputs.treated_mask.sum())
    assert inputs.cohort_index[~inputs.treated_mask].size > 0
    # Standardisation is what the model's priors assume.
    assert abs(float(inputs.outcome_std.mean())) < 1e-6
    assert abs(float(inputs.outcome_std.std()) - 1.0) < 1e-6


def test_cohort_codes_are_ordered_by_adoption_period():
    """Cohort codes must ascend with adoption period.

    `cohort_cell_counts[i]` is the weight applied to the model's `tau_g[i]`,
    so the two must refer to the same cohort. That alignment rests entirely
    on codes being assigned in ascending-period order, and nothing else in
    the suite pins it: the cell-count test above compares each count against
    `cohort_index == code` self-consistently, so it passes identically under
    a reversed or shuffled mapping.

    A misordering does not raise. It silently misweights the effect --
    reconstructing P4's ground truth with correct codes reproduces
    `true_att_overall` exactly at 2.8376, while reversing the code-to-period
    mapping yields 2.1624.
    """
    from cp_causal._panel import panel_arrays
    from cp_causal.estimators.hierarchical_did import build_model_inputs
    from cp_simulation.panel_scenarios import build_panel_scenario

    dgp = build_panel_scenario("P3")
    simulated = dgp.generate(200, 8, np.random.default_rng(0))
    panel = panel_arrays(simulated.dataset)
    inputs = build_model_inputs(panel)

    # Every treated unit's code must rank the same way its adoption period
    # does. Built from the panel's own per-unit cohort column rather than
    # from `adoption_periods`, so this fails if either the ordering or the
    # period-to-code mapping is disturbed.
    period_of_code: dict[int, int] = {}
    for unit, period in enumerate(panel.cohort):
        if period <= 0:
            continue
        code = int(inputs.cohort_index[panel.unit_index == unit][0])
        period_of_code.setdefault(code, int(period))
        assert period_of_code[code] == int(period), (
            f"code {code} maps to both period {period_of_code[code]} and {period}"
        )

    assert len(period_of_code) == inputs.n_cohorts
    codes_in_order = [period_of_code[c] for c in range(inputs.n_cohorts)]
    assert codes_in_order == sorted(codes_in_order), (
        f"cohort codes are not ascending in adoption period: {codes_in_order}"
    )


def test_never_treated_units_carry_no_cohort():
    from cp_causal._panel import panel_arrays
    from cp_causal.estimators.hierarchical_did import build_model_inputs
    from cp_simulation.panel_scenarios import build_panel_scenario

    dgp = build_panel_scenario("P3")
    simulated = dgp.generate(200, 8, np.random.default_rng(0))
    inputs = build_model_inputs(panel_arrays(simulated.dataset))

    # Never-treated rows must never be marked treated, whatever their index.
    assert not inputs.treated_mask[inputs.cohort_index < 0].any()
