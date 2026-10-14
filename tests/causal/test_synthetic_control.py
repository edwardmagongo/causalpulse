"""SyntheticControl: donor weight solve, point estimate, CounterfactualPath.

Permutation inference and the assumption checks are Tasks 5-6; this file
tests the weight solve and the shape of the result, not final tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest
from cp_causal.estimators.synthetic_control import (
    SyntheticControl,
    _leave_one_donor_out_max_swing,
    fit_donor_pool_point,
    hull_aware_placebo_runs,
    leave_one_donor_out_points,
    permutation_p_value,
    placebo_runs,
    ridge_augmented_weights,
    solve_donor_weights,
)
from cp_core.values import EstimatorConfig
from cp_simulation.donor_scenarios import (
    DONOR_N_PERIODS,
    DONOR_N_UNITS,
    DONOR_SCENARIOS,
    build_donor_scenario,
)

CONFIG = EstimatorConfig(bootstrap_reps=10, random_seed=1)


def _generate(scenario_id: str, seed: int = 0):
    # Reads n_units/n_periods off the scenario's own spec, not the shared
    # constants directly -- S5 deliberately overrides both to a shorter
    # panel, and generating it at the default protocol
    # instead would silently test a scenario other than the registered
    # one. Found at final whole-branch review: an earlier version of this
    # helper always used DONOR_N_UNITS/DONOR_N_PERIODS, so
    # test_sufficient_pre_periods_fails_on_s5 below happened to still pass
    # (adoption at period 3 still gives 2 pre-periods either way) without
    # actually exercising S5 as registered.
    spec = DONOR_SCENARIOS[scenario_id]
    return build_donor_scenario(scenario_id).generate(
        spec.n_units, spec.n_periods, np.random.default_rng(seed)
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
    from cp_causal._panel import donor_matrix, panel_arrays

    parr = panel_arrays(panel.dataset)
    treated, donors, pre = donor_matrix(parr)
    runs = placebo_runs(treated, donors, pre, ~pre)

    assert runs.point_estimates.shape == (donors.shape[0] + 1,)
    assert runs.rmspe_ratios.shape == (donors.shape[0] + 1,)
    weights = solve_donor_weights(treated[pre], donors[:, pre])
    synthetic = weights @ donors
    expected_point = float(np.mean(treated[~pre] - synthetic[~pre]))
    assert runs.point_estimates[0] == pytest.approx(expected_point)


def test_hull_aware_placebo_runs_share_placebo_runs_unit_swap():
    # Both iterate `_unit_swaps`, so they must agree unit-for-unit wherever
    # the hull check passes and the two forms coincide. If the swap were
    # written out twice and one copy drifted (a wrong index, a donor left in
    # its own pool), the hull-passing units would stop matching and this
    # would catch it -- the two prior instances of exactly that drift on this
    # branch are why the swap has one definition.
    panel = _generate("S3", seed=0)
    from cp_causal._panel import donor_matrix, panel_arrays

    treated, donors, pre = donor_matrix(panel_arrays(panel.dataset))
    post = ~pre
    simplex = placebo_runs(treated, donors, pre, post)
    hull_aware = hull_aware_placebo_runs(treated, donors, pre, post)

    assert hull_aware.point_estimates.shape == simplex.point_estimates.shape
    assert hull_aware.used_ridge.shape == simplex.point_estimates.shape
    for index in range(simplex.point_estimates.size):
        if not hull_aware.used_ridge[index]:
            assert hull_aware.point_estimates[index] == simplex.point_estimates[index]

    # Index 0 is the real treated unit under both, and on S3 it is the unit
    # whose hull check fails -- so the two arrays must differ there.
    assert hull_aware.used_ridge[0]
    expected = fit_donor_pool_point(treated, donors, pre, post)
    assert hull_aware.point_estimates[0] == expected.point
    assert hull_aware.point_estimates[0] != simplex.point_estimates[0]


def test_hull_aware_placebo_runs_rmspe_ratios_match_simplex_where_nothing_uses_ridge():
    # Mirrors test_hull_aware_placebo_runs_share_placebo_runs_unit_swap's own assertion for
    # point_estimates: wherever a unit's own hull check passes, its ratio must come from the
    # identical simplex fit `placebo_runs` uses, and the two must agree exactly, not
    # approximately. Even on S1 ("clean"), some individual placebo donor swaps still fail their
    # own hull check (each unit's check is against its own donor pool) -- confirmed directly
    # rather than assumed, so this test does not require the whole array to agree.
    panel = _generate("S1", seed=0)
    from cp_causal._panel import donor_matrix, panel_arrays

    treated, donors, pre = donor_matrix(panel_arrays(panel.dataset))
    post = ~pre
    simplex = placebo_runs(treated, donors, pre, post)
    hull_aware = hull_aware_placebo_runs(treated, donors, pre, post)

    assert hull_aware.rmspe_ratios.shape == simplex.rmspe_ratios.shape
    assert (~hull_aware.used_ridge).any(), "expected at least one unit to pass its own hull check"
    for index in range(simplex.rmspe_ratios.size):
        if not hull_aware.used_ridge[index]:
            assert hull_aware.rmspe_ratios[index] == simplex.rmspe_ratios[index]


def test_hull_aware_placebo_runs_rmspe_ratios_differ_from_simplex_under_ridge():
    # S3 triggers the ridge fallback for the real treated unit (index 0, see
    # the shared-unit-swap test above). rmspe_ratios is post/pre RMSPE, which
    # is sensitive to which form built the post-period counterfactual --
    # unlike fit()'s own pre_rmspe, it is not a pre-period-only statistic -- so
    # the hull-aware ratio at index 0 must differ from the plain-simplex one,
    # and must match the ratio computed directly from fit_donor_pool_point's
    # own post-fallback `synthetic`, not from a second, independent formula.
    panel = _generate("S3", seed=0)
    from cp_causal._panel import donor_matrix, panel_arrays

    treated, donors, pre = donor_matrix(panel_arrays(panel.dataset))
    post = ~pre
    simplex = placebo_runs(treated, donors, pre, post)
    hull_aware = hull_aware_placebo_runs(treated, donors, pre, post)

    assert hull_aware.used_ridge[0]
    assert hull_aware.rmspe_ratios[0] != simplex.rmspe_ratios[0]

    expected = fit_donor_pool_point(treated, donors, pre, post)
    pre_rmspe = float(np.sqrt(np.mean((treated[pre] - expected.synthetic[pre]) ** 2)))
    post_rmspe = float(np.sqrt(np.mean((treated[post] - expected.synthetic[post]) ** 2)))
    assert hull_aware.rmspe_ratios[0] == pytest.approx(post_rmspe / pre_rmspe)


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


def test_ridge_augmented_weights_have_an_intercept_and_one_coefficient_per_donor():
    rng = np.random.default_rng(0)
    treated = rng.normal(size=10)
    donors = rng.normal(size=(5, 10))

    intercept, coefficients = ridge_augmented_weights(treated, donors, penalty=1.0)

    assert isinstance(intercept, float)
    assert coefficients.shape == (5,)


def test_ridge_augmented_weights_recover_truth_on_the_convex_hull_scenario():
    # Measured over 300 seeds, plain simplex-SC bias
    # +1.21 (badly biased) vs ridge-augmented (penalty=1.0) bias +0.0223
    # (MCSE 0.0270). Tolerance 0.15 gives (0.15 - 0.0223) / 0.0270 = 4.7
    # MCSE headroom. Run at fewer seeds here (this is a unit test, not the
    # calibrated recovery harness -- that is Task 7) with a looser bound.
    from cp_causal._panel import donor_matrix, panel_arrays

    errors = []
    for seed in range(40):
        panel = build_donor_scenario("S3").generate(
            DONOR_N_UNITS, DONOR_N_PERIODS, np.random.default_rng(seed)
        )
        parr = panel_arrays(panel.dataset)
        treated, donors, pre = donor_matrix(parr)
        intercept, coefficients = ridge_augmented_weights(treated[pre], donors[:, pre], penalty=1.0)
        synthetic = intercept + coefficients @ donors
        point = float(np.mean(treated[~pre] - synthetic[~pre]))
        errors.append(point - panel.truth.true_att_by_cohort[18])
    assert abs(float(np.mean(errors))) < 0.2


def test_fit_reports_all_four_assumption_checks():
    panel = _generate("S1")
    result = SyntheticControl().fit(panel.dataset, CONFIG)
    names = {check.name for check in result.assumptions}
    assert names == {
        "pre_treatment_fit",
        "convex_hull",
        "donor_contamination",
        "sufficient_pre_periods",
    }


def test_convex_hull_check_fails_on_s3_and_passes_on_s1():
    s1 = SyntheticControl().fit(_generate("S1").dataset, CONFIG)
    s3 = SyntheticControl().fit(_generate("S3").dataset, CONFIG)
    s1_status = {c.name: c.status for c in s1.assumptions}
    s3_status = {c.name: c.status for c in s3.assumptions}
    assert s1_status["convex_hull"] == "PASS"
    assert s3_status["convex_hull"] == "FAIL"


def test_pre_treatment_fit_and_loo_swing_are_measured_on_the_right_branch_through_fit():
    # Found at final whole-branch review: pre_rmspe and the leave-one-out
    # diagnostic were both computed from whichever branch produced the
    # reported point (simplex, or the ridge fallback), which made
    # pre_treatment_fit_check's calibrated threshold (1.75, calibrated
    # against the *simplex* residual per its own docstring) unable to ever
    # FAIL on S3 -- the ridge fallback always fits well -- and made
    # leave_one_donor_out_max_swing mostly measure the gap between two
    # different estimator forms rather than genuine donor influence.
    # Both are now measured on the simplex residual / the branch that
    # actually produced `point`, matching their calibration. Measured over
    # 10 seeds post-fix: pre_treatment_fit statistic 3.6-7.0 (FAIL on
    # 10/10, well above the 1.75 threshold and consistent with the
    # calibration's S3-pre-ridge range of 2.04-4.76), loo_swing 0.07-0.21
    # (small and stable, not the 1.1-4.4 the pre-fix version reported).
    for seed in range(10):
        panel = _generate("S3", seed=seed)
        result = SyntheticControl().fit(panel.dataset, CONFIG)
        checks = {c.name: c for c in result.assumptions}
        assert checks["pre_treatment_fit"].status == "FAIL", seed
        assert checks["pre_treatment_fit"].statistic > 1.75, seed
        loo_swing = result.diagnostics["leave_one_donor_out_max_swing"].value
        assert loo_swing < 0.5, f"seed={seed} loo_swing={loo_swing}"


def test_sufficient_pre_periods_fails_on_s5():
    result = SyntheticControl().fit(_generate("S5").dataset, CONFIG)
    status = {c.name: c.status for c in result.assumptions}
    assert status["sufficient_pre_periods"] == "FAIL"


def test_the_max_swing_is_the_widest_gap_among_the_per_donor_estimates():
    """The extracted per-donor loop and the max-swing summary agree.

    The guard on the extraction: if these two ever disagree, one of them is
    computing a different leave-one-out than the other.
    """
    rng = np.random.default_rng(0)
    n_periods = 20
    donor_series = rng.normal(size=(8, n_periods)).cumsum(axis=1)
    treated_series = donor_series.mean(axis=0) + rng.normal(scale=0.1, size=n_periods)
    pre_mask = np.arange(n_periods) < 14
    post_mask = ~pre_mask

    weights = solve_donor_weights(treated_series[pre_mask], donor_series[:, pre_mask])
    synthetic = weights @ donor_series
    full_point = float(np.mean(treated_series[post_mask] - synthetic[post_mask]))

    donors, points = leave_one_donor_out_points(
        treated_series,
        donor_series,
        pre_mask,
        post_mask,
        reference_weights=weights,
        use_ridge=False,
    )
    swing = _leave_one_donor_out_max_swing(
        treated_series,
        donor_series,
        pre_mask,
        post_mask,
        full_point,
        reference_weights=weights,
        use_ridge=False,
    )
    expected = max((abs(p - full_point) for p in points), default=0.0)
    assert swing == pytest.approx(expected)
    # One estimate per influential donor, and each names the donor it dropped.
    assert len(donors) == len(points)
    assert all(0 <= d < donor_series.shape[0] for d in donors)


def test_donor_weights_do_not_depend_on_the_outcome_s_units():
    """Recording an outcome in packs or in cigarettes must not change the fit.

    `solve_donor_weights` minimises a sum of squares whose magnitude is set by
    the outcome's own scale, and SLSQP's finite-difference gradient is not
    scale-free: on the Proposition 99 panel (outcomes ~130, SSE ~4,891) it
    reported `success=True` after five iterations without moving off the
    uniform starting weights at all, yielding an unweighted average of 38
    states in place of a synthetic control.

    Dividing both series by one common positive scalar leaves the minimiser
    exactly unchanged -- the objective scales by 1/s**2 -- so this invariance
    is a property the solver owes, not an approximation.
    """
    rng = np.random.default_rng(0)
    n_pre = 20
    donor_pre = rng.normal(100.0, 20.0, size=(8, n_pre))
    treated_pre = 0.5 * donor_pre[0] + 0.3 * donor_pre[1] + 0.2 * donor_pre[2]

    in_packs = solve_donor_weights(treated_pre, donor_pre)
    in_units = solve_donor_weights(treated_pre * 1000.0, donor_pre * 1000.0)

    assert np.abs(in_packs - in_units).max() < 1e-6


def test_donor_weights_beat_the_uniform_starting_point_on_a_large_scale_outcome():
    # The failure above was silent: SLSQP returned its own x0 and reported
    # success. Pin that the solve actually improves on where it started.
    rng = np.random.default_rng(1)
    donor_pre = rng.normal(130.0, 30.0, size=(38, 19))
    treated_pre = 0.4 * donor_pre[0] + 0.35 * donor_pre[1] + 0.25 * donor_pre[2]

    weights = solve_donor_weights(treated_pre, donor_pre)

    def sse(w: np.ndarray) -> float:
        return float(np.sum((treated_pre - w @ donor_pre) ** 2))

    uniform = np.full(donor_pre.shape[0], 1.0 / donor_pre.shape[0])
    assert sse(weights) < 0.01 * sse(uniform)
