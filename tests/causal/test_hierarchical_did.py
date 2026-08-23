"""Hierarchical Bayesian DiD: model, posterior arithmetic, and the estimator."""

from __future__ import annotations

import numpy as np
import pytest
from cp_causal._mcmc import run_nuts
from cp_causal.estimators.hierarchical_did import (
    HIER_FAST_MCMC_CONFIG,
    HierarchicalDiD,
    _model,
)
from cp_core.values import EstimatorConfig
from cp_simulation.panel_scenarios import build_panel_scenario

CONFIG = EstimatorConfig(random_seed=0, options={"mcmc_config": HIER_FAST_MCMC_CONFIG})


def _panel_fixture(scenario_id: str = "P3", seed: int = 0):
    dgp = build_panel_scenario(scenario_id)
    return dgp.generate(200, 8, np.random.default_rng(seed))


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
    exposure = np.where(treated_mask, time_index - adopt, 0).astype(np.int64)
    return (
        outcome,
        unit_index,
        time_index,
        cohort_index,
        treated_mask,
        exposure,
        len(codes),
    )


def test_model_samples_and_exposes_the_expected_sites():
    (
        outcome,
        unit_index,
        time_index,
        cohort_index,
        treated_mask,
        exposure,
        n_cohorts,
    ) = _toy_panel()

    mcmc = run_nuts(
        _model,
        HIER_FAST_MCMC_CONFIG,
        0,
        outcome_std=outcome,
        unit_index=unit_index,
        time_index=time_index,
        cohort_index=cohort_index,
        treated_mask=treated_mask,
        exposure=exposure,
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

    # Deliberately nonzero on EVERY row, never-treated ones included. The
    # effect term is `tau_g + delta * exposure`, so if the mask were dropped
    # a never-treated row would pick up both pieces; forcing a large
    # exposure here means the delta half cannot cancel to zero by accident
    # and the mask is provably what does the work.
    exposure = np.full(n_units * n_periods, 5, dtype=np.int64)

    traced = trace(seed(_model, jax.random.PRNGKey(0))).get_trace(
        outcome_std=outcome,
        unit_index=unit_index,
        time_index=time_index,
        cohort_index=cohort_index,
        treated_mask=treated_mask,
        exposure=exposure,
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


def test_weighted_att_weights_by_treated_cells_not_uniformly():
    from cp_causal.estimators.hierarchical_did import weighted_att_draws

    # Two cohorts, wildly unequal treated-cell counts and very different
    # effects. Cell counts, not unit counts -- see the docstring.
    tau_g = np.array([[1.0, 5.0], [1.0, 5.0]])  # 2 draws, 2 cohorts
    cells = np.array([90, 10])

    att = weighted_att_draws(tau_g, cells, outcome_sd=1.0)

    # Cell-weighted: 0.9*1 + 0.1*5 = 1.4.  Unweighted mean would be 3.0.
    assert att == pytest.approx(np.array([1.4, 1.4]))
    assert not np.allclose(att, 3.0), "weighting by treated cells is not happening"


def test_weighted_att_rescales_off_the_standardised_scale():
    from cp_causal.estimators.hierarchical_did import weighted_att_draws

    tau_g = np.array([[2.0, 2.0]])
    cells = np.array([1, 1])
    att = weighted_att_draws(tau_g, cells, outcome_sd=3.0)
    # The model fits standardised outcomes; effects come back multiplied by sd.
    assert att == pytest.approx(np.array([6.0]))


def test_weighted_att_includes_the_exposure_contribution():
    """The ATT is the average effect over treated cells, and under the
    model that effect is `tau_g + delta * exposure`. Both terms belong in
    it.

    Dropping the second reports the effect at zero exposure instead of the
    average effect. That is not a rounding difference: a model with no
    exposure term at all measured a -17% ATT bias and 0/15 interval
    coverage on P4, against +0.3%/7-of-8 once the term was added.
    """
    from cp_causal.estimators.hierarchical_did import weighted_att_draws

    tau_g = np.array([[1.0, 1.0], [1.0, 1.0]])
    cells = np.array([50, 50])
    delta = np.array([0.2, 0.4])

    att = weighted_att_draws(
        tau_g, cells, outcome_sd=1.0, delta_draws=delta, mean_treated_exposure=3.0
    )
    # 1.0 + 0.2*3 = 1.6, and 1.0 + 0.4*3 = 2.2. Per draw, so the two differ.
    assert att == pytest.approx(np.array([1.6, 2.2]))

    # Omitting the term collapses both draws to the exposure-0 effect.
    without = weighted_att_draws(tau_g, cells, outcome_sd=1.0)
    assert without == pytest.approx(np.array([1.0, 1.0]))
    assert not np.allclose(att, without), (
        "the exposure contribution is being dropped from the ATT"
    )

    # It scales with the outcome like every other effect-scaled quantity.
    scaled = weighted_att_draws(
        tau_g, cells, outcome_sd=2.0, delta_draws=delta, mean_treated_exposure=3.0
    )
    assert scaled == pytest.approx(np.array([3.2, 4.4]))


def test_weighted_att_is_computed_per_draw_not_from_summaries():
    """Averaging the draws first and weighting the summaries afterwards gives
    the same POINT estimate and a wrong credible interval, because it discards
    the joint posterior's cross-cohort correlation. Only the spread separates
    the two, so the fixture has to be built to make them diverge.

    This fixture deliberately combines UNEQUAL weights with draws that are not
    a symmetric swap. An earlier version used `tau_g=[[0,4],[4,0]]` with equal
    weights `[50,50]`, which yields `[2.0, 2.0]` under BOTH the correct
    per-draw computation and an average-first implementation -- it could only
    ever have failed on the shape assertion, never on the property its own
    name claims to guard. Verified numerically before replacing it.
    """
    from cp_causal.estimators.hierarchical_did import weighted_att_draws

    tau_g = np.array([[0.0, 4.0], [8.0, 0.0]])
    cells = np.array([90, 10])
    att = weighted_att_draws(tau_g, cells, outcome_sd=1.0)

    assert att.shape == (2,)
    # Per draw: 0*0.9 + 4*0.1 = 0.4, then 8*0.9 + 0*0.1 = 7.2.
    assert att == pytest.approx(np.array([0.4, 7.2]))
    # Average-first would collapse both draws to 4*0.9 + 2*0.1 = 3.8. Asserted
    # explicitly so the failure names the actual defect rather than just
    # reporting two numbers that differ.
    assert not np.allclose(att, 3.8), (
        "weighted_att_draws averaged the draws before weighting -- the point "
        "estimate survives that, the credible interval does not"
    )


def test_hierarchical_did_is_registered_under_its_own_name():
    from cp_causal.registry import get_estimator

    assert get_estimator("hierarchical_did") is HierarchicalDiD


def test_hierarchical_did_requires_a_panel():
    assert HierarchicalDiD.requirements.needs_panel is True


def test_fit_reports_a_posterior_att_with_cohort_diagnostics():
    result = HierarchicalDiD().fit(_panel_fixture().dataset, CONFIG)

    assert result.estimand == "ATT"
    assert result.ci_basis == "posterior"
    assert result.p_value is None
    assert result.ci.lower <= result.point <= result.ci.upper
    for key in ("mu_tau", "sigma_tau", "tau_g", "p_effect_positive"):
        assert key in result.diagnostics


def test_fit_reports_the_mcmc_diagnostics_as_assumption_checks():
    result = HierarchicalDiD().fit(_panel_fixture().dataset, CONFIG)
    names = {c.name for c in result.assumptions}
    assert {"r_hat", "effective_sample_size", "divergent_transitions"} <= names


def test_fit_rejects_a_non_mcmcconfig_option():
    with pytest.raises(TypeError, match="mcmc_config"):
        HierarchicalDiD().fit(
            _panel_fixture().dataset,
            EstimatorConfig(random_seed=0, options={"mcmc_config": "production"}),
        )
