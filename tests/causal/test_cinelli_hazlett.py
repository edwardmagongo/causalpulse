"""Cinelli-Hazlett omitted-variable-bias bounds.

Two things are under test and they are different. The formula itself is
checked against a published worked example, reproducible from three numbers
by hand. The *plumbing* -- that the design matrix reconstructed here is the
one the estimator actually fitted -- is checked by pinning the reconstructed
coefficient against the estimator's own reported point. The second is the
one likeliest to rot: a design that drifts still returns a plausible
robustness value, for a regression nobody ran.
"""

import numpy as np
import pytest
from cp_causal.registry import all_estimators
from cp_causal.sensitivity import (
    OLS_SENSITIVITY_ESTIMATORS,
    _ols_treatment_coefficient,
    _robustness_value,
    cinelli_hazlett,
)
from cp_core.values import EstimatorConfig
from cp_simulation.panel_scenarios import build_panel_scenario
from cp_simulation.scenarios import build_scenario

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)


def cross_section():
    return build_scenario("A").generate(n_units=800, rng=np.random.default_rng(0)).dataset


def uniform_panel():
    return (
        build_panel_scenario("P1")
        .generate(n_units=60, n_periods=6, rng=np.random.default_rng(0))
        .dataset
    )


def test_the_published_worked_example_reproduces():
    # Darfur / `directlyharmed`: coefficient 0.09731582, se 0.02325654, df 783.
    # Re-derived here from those three numbers rather than copied from the
    # plan: t = 4.184450, partial_r2 = 0.021873, RV = 0.138776,
    # RV_alpha = 0.076299. sensemakr publishes 2.2%, 13.9% and 7.6% for this
    # example, which these match to every digit it prints.
    t_statistic = 0.09731582 / 0.02325654
    rv, rv_alpha, partial_r2 = _robustness_value(t_statistic, df=783)
    assert partial_r2 == pytest.approx(0.0219, abs=0.0005)
    assert rv == pytest.approx(0.1388, abs=0.0005)
    assert rv_alpha == pytest.approx(0.0763, abs=0.0005)


def test_a_stronger_result_is_harder_to_explain_away():
    weak, _, _ = _robustness_value(2.0, df=500)
    strong, _, _ = _robustness_value(8.0, df=500)
    assert strong > weak


def test_the_robustness_value_stays_inside_the_unit_interval():
    # RV -> 1 as the t-statistic grows without bound; it never exceeds it.
    for t_statistic in (0.01, 1.0, 10.0, 1000.0, 1e6):
        rv, rv_alpha, partial_r2 = _robustness_value(t_statistic, df=100)
        assert 0.0 <= rv < 1.0
        assert 0.0 <= rv_alpha < 1.0
        assert 0.0 <= partial_r2 < 1.0


def test_the_robustness_value_still_rises_toward_one_in_the_far_tail():
    # The textbook way to write this formula, `0.5 * (sqrt(f**4 + 4 * f**2) -
    # f**2)`, loses every significant digit once `f**4` dominates `4 * f**2`.
    # Measured at df=100: it returns exactly 1.0 by t=1e6 -- which the strict
    # bound asserted above would already reject -- and collapses to 0.0 by
    # t=1e11, inverting the quantity rather than merely blurring it. This
    # module rationalises the difference to `2 / (sqrt(1 + 4 / f**2) + 1)`,
    # which agrees bit-for-bit through f=100 and stays correct here. Measured
    # saturation of the rationalised form is f ~= 7.17e7, i.e. t ~= 7.2e8 at
    # this df -- 7x beyond the largest value this test asserts.
    previous = 0.0
    for t_statistic in (1e2, 1e4, 1e6, 1e8):
        rv, _, _ = _robustness_value(t_statistic, df=100)
        assert rv > previous, f"RV fell at t={t_statistic}"
        assert rv < 1.0
        previous = rv
    assert previous == pytest.approx(1.0, abs=1e-6)


def test_an_insignificant_estimate_has_a_zero_significance_robustness_value():
    # Below the critical value there is nothing for a confounder to undo:
    # the estimate is already indistinguishable from zero.
    _, rv_alpha, _ = _robustness_value(0.5, df=500)
    assert rv_alpha == 0.0


@pytest.mark.parametrize(
    ("estimator_name", "build"),
    [
        ("lin_regression", cross_section),
        ("difference_in_means", cross_section),
        ("did_2x2", uniform_panel),
        ("twfe", uniform_panel),
    ],
)
def test_the_reconstructed_design_reproduces_the_estimator_s_own_point(
    estimator_name, build
):
    # The guard against a drifted design. If this fails, the robustness value
    # describes a regression the estimator never ran.
    dataset = build()
    reported = all_estimators()[estimator_name]().fit(dataset, CONFIG).point

    fit, column = _ols_treatment_coefficient(dataset, estimator_name)
    assert float(fit.beta[column]) == pytest.approx(reported, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize(
    ("estimator_name", "build"),
    [
        ("lin_regression", cross_section),
        ("difference_in_means", cross_section),
        ("did_2x2", uniform_panel),
        ("twfe", uniform_panel),
    ],
)
def test_every_supported_estimator_gets_a_result(estimator_name, build):
    result = cinelli_hazlett(build(), estimator_name, CONFIG)
    assert result is not None
    assert 0.0 <= result.robustness_value < 1.0
    assert result.degrees_of_freedom > 0


def test_event_study_is_refused_because_its_point_is_not_one_coefficient():
    assert "event_study" not in OLS_SENSITIVITY_ESTIMATORS
    assert cinelli_hazlett(uniform_panel(), "event_study", CONFIG) is None


@pytest.mark.parametrize(
    "estimator_name", ["aipw", "ipw", "causal_forest", "synthetic_control", "bsts"]
)
def test_non_ols_estimators_return_none(estimator_name):
    assert cinelli_hazlett(cross_section(), estimator_name, CONFIG) is None


def test_an_unknown_estimator_name_is_refused_rather_than_silently_none():
    # A typo'd name must not read as "this technique does not apply here".
    with pytest.raises(KeyError):
        cinelli_hazlett(cross_section(), "lin_regresion", CONFIG)


def test_the_supported_set_is_exactly_pinned():
    # Exact, not a spot check: an estimator added to this set without a
    # design-reconstruction test above is the failure mode this catches.
    supported = frozenset({"lin_regression", "difference_in_means", "did_2x2", "twfe"})
    assert supported == OLS_SENSITIVITY_ESTIMATORS
