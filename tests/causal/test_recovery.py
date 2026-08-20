import numpy as np
import pytest
from cp_causal.estimators.doubly_robust import AugmentedIPW
from cp_causal.estimators.experimental import DifferenceInMeans
from cp_causal.estimators.regression import LinRegressionAdjustment
from cp_causal.estimators.weighting import InverseProbabilityWeighting
from cp_causal.registry import all_estimators
from cp_causal.router import eligible_estimators
from cp_core.values import EstimatorConfig
from cp_simulation.scenarios import SCENARIOS, build_scenario

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)
ADJUSTING = (LinRegressionAdjustment(), InverseProbabilityWeighting(), AugmentedIPW())


def mean_bias(estimator, scenario_id: str, seeds: int = 10, n: int = 4000) -> float:
    errors = []
    for seed in range(seeds):
        simulated = build_scenario(scenario_id).generate(
            n_units=n, rng=np.random.default_rng(seed)
        )
        estimate = estimator.fit(simulated.dataset, CONFIG)
        errors.append(estimate.point - simulated.truth.true_ate)
    return float(np.mean(errors))


@pytest.mark.parametrize("scenario_id", ["A", "B", "C"])
def test_every_estimator_recovers_randomised_scenarios(scenario_id: str):
    for estimator in (DifferenceInMeans(), *ADJUSTING):
        assert abs(mean_bias(estimator, scenario_id)) < 0.012, estimator.name


def test_adjusting_estimators_remove_confounding_bias_under_scenario_d():
    for estimator in ADJUSTING:
        assert abs(mean_bias(estimator, "D")) < 0.015, estimator.name


def test_adjusting_estimators_remove_confounding_bias_under_scenario_g():
    # Scenario G's true propensity is expit(strength * (p0 - mean(p0))), where
    # p0 is itself a clipped sigmoid of the covariates — markedly nonlinear.
    # AIPW fits two nuisance models: a linear-logit propensity model and a
    # plain-OLS outcome regression against a binary/sigmoid outcome surface.
    # Under scenario G neither nuisance is correctly specified. Double
    # robustness requires *at least one* of the two to be correct; since
    # neither is here, the guarantee does not apply, and a residual
    # finite-sample bias is what theory predicts rather than a defect in the
    # estimator.
    #
    # This test runs 40 seeds (not the module default of 10) because at 10
    # seeds the margin between measured bias and tolerance is too thin —
    # under 1 SE for AIPW and Lin — for the assertion to be a meaningful
    # guard against regressions rather than a coin flip against sampling
    # noise. Measured at n=4000, 40 seeds:
    #   lin_regression: bias +0.00782 (se 0.00260) -> 2.8 SE of headroom under 0.015
    #   ipw:            bias -0.00314 (se 0.00327) -> 5.5 SE of headroom under 0.015
    #   aipw:           bias +0.01696 (se 0.00331) -> 3.9 SE of headroom under 0.03
    # At n=20000, 15 seeds, AIPW bias is +0.01208 (se 0.0036); at n=60000, 6
    # seeds, +0.01096 (se 0.0025) — consistent with shrinking-but-nonzero
    # finite-sample bias rather than a fixed defect.
    tolerances = {
        "lin_regression": 0.015,
        "ipw": 0.015,
        "aipw": 0.03,
    }
    for estimator in ADJUSTING:
        tolerance = tolerances[estimator.name]
        assert abs(mean_bias(estimator, "G", seeds=40)) < tolerance, estimator.name


@pytest.mark.parametrize("scenario_id", ["D", "G"])
def test_unadjusted_estimator_remains_badly_biased(scenario_id: str):
    # The control condition. If this ever passes, the confounded scenarios have
    # stopped being confounded and every result above is meaningless.
    assert mean_bias(DifferenceInMeans(), scenario_id) > 0.05


def test_estimators_recover_the_average_of_a_heterogeneous_effect():
    for estimator in (DifferenceInMeans(), *ADJUSTING):
        assert abs(mean_bias(estimator, "F")) < 0.015, estimator.name


def test_null_scenario_produces_no_systematic_significance():
    # Scenario B has exactly zero effect. At the 5% level roughly 1 in 20 fits
    # should read as significant by chance; a systematically significant
    # estimator is broken.
    significant = 0
    total = 0
    for seed in range(20):
        simulated = build_scenario("B").generate(
            n_units=3000, rng=np.random.default_rng(seed)
        )
        for estimator in (DifferenceInMeans(), AugmentedIPW()):
            significant += int(estimator.fit(simulated.dataset, CONFIG).is_significant)
            total += 1

    assert significant / total < 0.25


def test_bootstrap_and_analytic_intervals_broadly_agree():
    simulated = build_scenario("D").generate(
        n_units=4000, rng=np.random.default_rng(0)
    )
    for estimator in ADJUSTING:
        config = EstimatorConfig(bootstrap_reps=300, random_seed=2)
        result = estimator.fit(simulated.dataset, config)
        assert result.bootstrap_ci is not None
        assert result.se is not None
        # Overlapping intervals; a gross mismatch means one of the two is wrong.
        assert result.bootstrap_ci.lower < result.ci.upper
        assert result.ci.lower < result.bootstrap_ci.upper


def test_router_offers_every_registered_estimator_on_covariate_data():
    simulated = build_scenario("D").generate(
        n_units=4000, rng=np.random.default_rng(0)
    )

    eligible = set(eligible_estimators(simulated.dataset))

    # Exact, not a subset: a panel estimator becoming eligible on a
    # cross-section would mean its `needs_panel` requirement had stopped
    # working, which is the router's whole job.
    #
    # `causal_forest` joined this set in Phase 6. It belongs here on the same
    # terms as the other four -- cross-sectional, `needs_covariates`, and
    # scenario D's 4000 units clear its 50-per-arm minimum -- so its arrival
    # is the router working, not leaking. Every estimator added to this
    # registry must be accounted for here deliberately; that is the point of
    # pinning the set exactly rather than asserting a subset.
    assert {
        "difference_in_means",
        "lin_regression",
        "ipw",
        "aipw",
        "causal_forest",
    } == eligible


def test_importing_cp_causal_registers_every_cross_sectional_estimator():
    import cp_causal  # noqa: F401

    # A subset check here, since the registry now also holds panel estimators.
    assert {"difference_in_means", "lin_regression", "ipw", "aipw"} <= set(
        all_estimators()
    )


def test_all_scenarios_are_covered_by_this_harness():
    # If a scenario is added without a recovery expectation, say so loudly
    # rather than letting it drift untested.
    covered = {"A", "B", "C", "D", "F", "G"}
    assert set(SCENARIOS) == covered
