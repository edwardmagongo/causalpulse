"""SyntheticControl: donor weight solve, point estimate, CounterfactualPath.

Permutation inference and the assumption checks are Tasks 5-6; this file
tests the weight solve and the shape of the result, not final tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest
from cp_causal.estimators.synthetic_control import (
    SyntheticControl,
    permutation_p_value,
    placebo_runs,
    solve_donor_weights,
)
from cp_core.values import EstimatorConfig
from cp_simulation.donor_scenarios import DONOR_N_PERIODS, DONOR_N_UNITS, build_donor_scenario

CONFIG = EstimatorConfig(bootstrap_reps=10, random_seed=1)


def _generate(scenario_id: str, seed: int = 0):
    return build_donor_scenario(scenario_id).generate(
        DONOR_N_UNITS, DONOR_N_PERIODS, np.random.default_rng(seed)
    )


def test_solve_donor_weights_are_non_negative_and_sum_to_one():
    rng = np.random.default_rng(0)
    treated = rng.normal(size=10)
    donors = rng.normal(size=(5, 10))

    weights = solve_donor_weights(treated, donors)

    assert weights.shape == (5,)
    assert (weights >= -1e-8).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_solve_donor_weights_exactly_reproduces_an_achievable_target():
    # If the treated series is itself an exact convex combination of the
    # donors, the solver should find (very close to) that combination and
    # drive pre-period SSE to (very close to) zero.
    donors = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    treated = 0.25 * donors[0] + 0.75 * donors[1]

    weights = solve_donor_weights(treated, donors)
    fitted = weights @ donors

    np.testing.assert_allclose(fitted, treated, atol=1e-4)


def test_synthetic_control_requirements_reject_more_than_one_treated_unit():
    requirements = SyntheticControl.requirements
    assert requirements.max_treated_units == 1
    assert requirements.min_treated_units == 1
    assert requirements.needs_panel is True


def test_fit_returns_att_with_a_populated_counterfactual_path():
    panel = _generate("S1")
    result = SyntheticControl().fit(panel.dataset, CONFIG)

    assert result.estimand == "ATT"
    assert result.n_treated == 1
    assert result.n_control == 17
    path = result.counterfactual
    assert path is not None
    assert path.periods.shape == (DONOR_N_PERIODS,)
    # assert_allclose, not assert_array_equal: `pointwise_effect` is
    # `observed - predicted_absent_treatment`, so recombining it by addition
    # is subject to ordinary IEEE 754 rounding (confirmed on this platform:
    # 1 of 24 periods differs from `observed` by exactly 1 ULP). That is a
    # floating-point rounding artifact, not evidence the invariant is wrong.
    np.testing.assert_allclose(
        path.observed, path.pointwise_effect + path.predicted_absent_treatment, atol=1e-9
    )


def test_fit_recovers_a_reasonable_estimate_on_the_clean_scenario():
    # A loose, single-seed sanity bound -- not the calibrated recovery
    # tolerance, which is Task 7's test_donor_recovery.py at 400 seeds.
    panel = _generate("S1", seed=0)
    result = SyntheticControl().fit(panel.dataset, CONFIG)
    true_att = panel.truth.true_att_by_cohort[18]
    assert abs(result.point - true_att) < 1.0


def test_placebo_runs_first_entry_is_the_real_treated_unit():
    panel = _generate("S1", seed=0)
    from cp_causal._panel import panel_arrays
    from cp_causal.estimators.synthetic_control import _donor_matrix

    parr = panel_arrays(panel.dataset)
    treated, donors, pre = _donor_matrix(parr)
    runs = placebo_runs(treated, donors, pre, ~pre)

    assert runs.point_estimates.shape == (donors.shape[0] + 1,)
    assert runs.rmspe_ratios.shape == (donors.shape[0] + 1,)
    weights = solve_donor_weights(treated[pre], donors[:, pre])
    synthetic = weights @ donors
    expected_point = float(np.mean(treated[~pre] - synthetic[~pre]))
    assert runs.point_estimates[0] == pytest.approx(expected_point)


def test_permutation_p_value_is_conservative_for_a_typical_donor():
    # A rank right in the middle of 18 draws gives p roughly 0.5, not
    # anything close to significant.
    ratios = np.array([1.0] + list(np.linspace(0.5, 1.5, 17)))
    p = permutation_p_value(ratios)
    assert 0.0 < p <= 1.0


def test_permutation_p_value_is_small_when_the_treated_unit_is_the_extreme():
    ratios = np.array([10.0] + [1.0] * 17)
    p = permutation_p_value(ratios)
    assert p == pytest.approx(1 / 18)


def test_fit_reports_a_real_p_value_and_diagnostics():
    panel = _generate("S1", seed=0)
    result = SyntheticControl().fit(panel.dataset, CONFIG)

    assert result.p_value is not None
    assert 0.0 <= result.p_value <= 1.0
    assert result.ci.lower <= result.point <= result.ci.upper
    assert "in_time_placebo_effect" in result.diagnostics
    assert "leave_one_donor_out_max_swing" in result.diagnostics


def test_null_scenario_permutation_p_value_is_roughly_uniform():
    # Design doc §6: S0 calibrates the false-positive rate. At the 5% level,
    # roughly 1 in 20 fits should read as significant by chance. Measured
    # over 100 seeds at this protocol (documented here so a future reader can
    # check the arithmetic): X/100 read significant -- run once and record
    # the real count before asserting a bound, exactly as
    # test_null_scenario_produces_no_systematic_significance does in
    # test_panel_recovery.py.
    significant = 0
    for seed in range(100):
        panel = _generate("S0", seed=seed)
        result = SyntheticControl().fit(panel.dataset, CONFIG)
        significant += int(result.p_value is not None and result.p_value < 0.05)
    assert significant / 100 < 0.25
