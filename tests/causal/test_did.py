import numpy as np
import pytest
from cp_causal.estimators.did import DifferenceInDifferences
from cp_core.values import EstimatorConfig
from cp_simulation.panel_scenarios import build_panel_scenario

CONFIG = EstimatorConfig(bootstrap_reps=50, random_seed=1)


def generate(scenario_id, seed=0, n_units=400):
    return build_panel_scenario(scenario_id).generate(
        n_units=n_units, n_periods=8, rng=np.random.default_rng(seed)
    )


def mean_deviation(scenario_id: str, seeds: int = 10) -> float:
    errors = []
    for seed in range(seeds):
        panel = generate(scenario_id, seed=seed)
        estimate = DifferenceInDifferences().fit(panel.dataset, CONFIG)
        errors.append(estimate.point - panel.truth.true_att_overall)
    return float(np.mean(errors))


def test_recovers_the_effect_on_a_clean_two_by_two():
    # A single fixed seed is the wrong design here: across the project's full
    # 60-seed P1 protocol at n_units=400 (see `PANEL_SEEDS`/`PANEL_N_UNITS` in
    # panel_scenarios.py), seed=0 -- the default this test used to run alone
    # -- is deterministically the single worst draw of all 60, deviating by
    # 0.1842 (cross-checked directly against the four-group-means formula,
    # which reproduces this same 0.8158 to 10 decimal places). No tolerance
    # tight enough to be meaningful clears that draw; the earlier abs=0.2 fix
    # only reverse-engineered a number that barely cleared it. The correct
    # fix, per this project's established discipline (see Phase 2's
    # `mean_bias` in test_recovery.py), is to average over several seeds
    # instead of widening the tolerance or picking a luckier one.
    #
    # Measured here at seeds=10, n_units=400 (this module's `generate`
    # default): mean deviation -0.0579, SEM 0.0226 (std 0.0716 / sqrt(10)).
    # abs=0.15 -- the originally planned tolerance -- leaves
    # (0.15 - 0.0579) / 0.0226 = 4.07 SEM of headroom, comfortably above this
    # project's four-SEM bar, while still catching a genuinely broken
    # estimator (wrong sign, wrong scale, or one that ignores treatment).
    assert abs(mean_deviation("P1")) < 0.15

    panel = generate("P1")
    result = DifferenceInDifferences().fit(panel.dataset, CONFIG)
    assert result.estimand == "ATT"
    assert result.method == "did_2x2"


def test_reports_both_an_analytic_standard_error_and_a_bootstrap_interval():
    # Dual uncertainty is a settled project decision: an estimator reporting
    # only one of the two is incomplete.
    result = DifferenceInDifferences().fit(generate("P1").dataset, CONFIG)

    assert result.se is not None and result.se > 0
    assert result.bootstrap_ci is not None
    assert result.ci_basis == "analytic"


def test_sample_sizes_count_units_not_unit_periods():
    # A unit-period count would inflate both by the panel length, making every
    # reported sample size wrong by the same plausible-looking multiple.
    panel = generate("P1", n_units=400)

    result = DifferenceInDifferences().fit(panel.dataset, CONFIG)

    assert result.n_treated + result.n_control == 400
    assert result.n_treated == int((panel.truth.true_cohorts > 0).sum())


def test_scale_follows_the_schema_outcome_type():
    result = DifferenceInDifferences().fit(generate("P1").dataset, CONFIG)

    assert result.scale == "absolute"


def test_clustered_standard_error_is_reported_and_diagnosed():
    result = DifferenceInDifferences().fit(generate("P1").dataset, CONFIG)

    assert "clustered_se_ratio" in result.diagnostics
    assert "cluster_count" in result.diagnostics
    assert result.diagnostics["cluster_count"].value == 400.0


def test_carries_the_counterfactual_control_mean():
    result = DifferenceInDifferences().fit(generate("P1").dataset, CONFIG)

    assert result.counterfactual is not None
    assert result.counterfactual.predicted_absent_treatment < (
        result.counterfactual.observed_mean
    )


def test_no_concurrent_intervention_is_reported_as_not_testable():
    result = DifferenceInDifferences().fit(generate("P1").dataset, CONFIG)

    statuses = {a.name: a.status for a in result.assumptions}
    assert statuses["no_concurrent_intervention"] == "NOT_TESTABLE"
    assert statuses["absorbing_treatment"] == "PASS"


def test_the_router_excludes_it_from_staggered_data_with_a_reason():
    # This is what `needs_uniform_adoption` exists for: an explicit exclusion
    # beats a silently wrong 2x2 on staggered data.
    from cp_causal.router import route

    decisions = {d.estimator_name: d for d in route(generate("P3").dataset)}

    decision = decisions["did_2x2"]
    assert not decision.eligible
    assert any("uniform adoption" in reason for reason in decision.reasons)


def test_the_router_admits_it_on_uniform_adoption():
    from cp_causal.router import eligible_estimators

    assert "did_2x2" in eligible_estimators(generate("P1").dataset)


def test_twfe_and_the_two_by_two_agree_exactly_under_uniform_adoption():
    # Two independent implementations of the same estimand on identical data,
    # which is a stronger cross-check than either has alone. Measured max
    # discrepancy 2.66e-14 over the 5 seeds below (n_units=200), so the
    # tolerance is 1e-9 rather than the 1e-14 an earlier draft of the design
    # proposed -- 1e-14 is below the precision the two paths actually achieve.
    from cp_causal.estimators.did import TwoWayFixedEffects

    for seed in range(5):
        panel = generate("P1", seed=seed, n_units=200)
        a = DifferenceInDifferences().fit(panel.dataset, CONFIG).point
        b = TwoWayFixedEffects().fit(panel.dataset, CONFIG).point
        assert a == pytest.approx(b, abs=1e-9), seed


def test_twfe_accepts_staggered_adoption():
    from cp_causal.estimators.did import TwoWayFixedEffects
    from cp_causal.router import eligible_estimators

    assert "twfe" in eligible_estimators(generate("P3").dataset)

    result = TwoWayFixedEffects().fit(generate("P3").dataset, CONFIG)

    assert result.estimand == "ATT"


def test_twfe_recovers_truth_when_effects_are_homogeneous():
    # The brief's literal test used a single fixed seed (P3's default seed=0).
    # Per this project's just-relearned discipline (see
    # test_recovers_the_effect_on_a_clean_two_by_two above), a single seed is
    # checked for being a deterministic worst case before it is trusted, and
    # the honest fix is to average rather than to pick or widen around one
    # draw.
    #
    # Measured here at seeds=10, n_units=400 (this module's `generate`
    # default): deviations were [0.0761, -0.0195, 0.0430, 0.1032, 0.0296,
    # -0.0126, -0.1242, 0.0012, 0.0246, 0.0421], mean 0.0164, SEM 0.0197 (std
    # 0.0622 / sqrt(10)). seed=0's own deviation is 0.0761 -- not the worst
    # draw (seed=6 is, at -0.1242) -- and every one of the 10 sampled seeds
    # stays under abs=0.15. abs=0.15 leaves (0.15 - 0.0164) / 0.0197 = 6.78 SEM
    # of headroom on the mean, comfortably above this project's four-SEM bar,
    # while still catching a genuinely broken estimator.
    from cp_causal.estimators.did import TwoWayFixedEffects

    errors = []
    for seed in range(10):
        panel = generate("P3", seed=seed)
        result = TwoWayFixedEffects().fit(panel.dataset, CONFIG)
        errors.append(result.point - panel.truth.true_att_overall)

    assert abs(float(np.mean(errors))) < 0.15


def test_twfe_standard_error_survives_a_near_singular_fixed_effect_block():
    # On a full-rank TWFE design the unit- and period-dummy coefficients have
    # near-zero true variances and the sandwich returns some of them as small
    # negatives (around -1.7e-27, roughly one seed in thirty-six on the
    # cluster-robust path). A blanket sqrt over the diagonal turns that into
    # `RuntimeWarning: invalid value encountered in sqrt`, which
    # filterwarnings=["error"] promotes to an intermittent failure with no
    # obvious cause. Reading only the treatment coefficient's entry is the fix.
    from cp_causal.estimators.did import TwoWayFixedEffects

    for seed in range(12):
        result = TwoWayFixedEffects().fit(generate("P1", seed=seed).dataset, CONFIG)
        assert result.se is not None and np.isfinite(result.se)


def test_twfe_clusters_by_default_even_though_the_config_says_iid():
    # EstimatorConfig defaults bootstrap_kind="iid". A panel estimator must not
    # inherit that: an iid bootstrap over panel rows destroys within-unit
    # dependence.
    from cp_causal.estimators.did import TwoWayFixedEffects

    config = EstimatorConfig(bootstrap_reps=50, random_seed=1, bootstrap_kind="iid")
    result = TwoWayFixedEffects().fit(generate("P1").dataset, config)

    assert result.diagnostics["clustered_se_ratio"].value > 1.0
