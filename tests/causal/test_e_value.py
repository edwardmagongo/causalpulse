"""The E-value, checked against arithmetic that can be done by hand.

Internal consistency is not enough for a published formula: a sign error in
`rr + sqrt(rr*(rr-1))` still round-trips through its own inverse. Every case
below is either hand-computable or a value from the source paper.
"""

import math

import numpy as np
import pytest
from cp_causal.registry import all_estimators
from cp_causal.sensitivity import e_value
from cp_core.results import (
    CounterfactualPoint,
    EffectEstimate,
    FitMetadata,
)
from cp_core.values import EstimatorConfig, Interval
from cp_simulation.scenarios import build_scenario


def an_estimate(
    *, point, lower, upper, scale, counterfactual=None
) -> EffectEstimate:
    return EffectEstimate(
        estimand="ATE",
        method="stub",
        point=point,
        se=None,
        ci=Interval(lower=lower, upper=upper),
        ci_basis="analytic",
        bootstrap_ci=None,
        p_value=None,
        n_treated=100,
        n_control=100,
        scale=scale,
        diagnostics={},
        assumptions=(),
        counterfactual=counterfactual,
        fit_metadata=FitMetadata.capture(
            estimator="stub", config_hash="0" * 16, runtime_seconds=0.0,
            random_seed=None,
        ),
    )


def test_a_risk_ratio_of_two_gives_the_hand_computed_e_value():
    # RR = 2  ->  2 + sqrt(2*1) = 3.41421356...
    estimate = an_estimate(
        point=math.log(2.0), lower=math.log(1.5), upper=math.log(2.7),
        scale="log_ratio",
    )
    result = e_value(estimate)
    assert result.e_value == pytest.approx(2 + math.sqrt(2.0), rel=1e-12)
    assert result.approximated is False


def test_the_ci_bound_uses_the_bound_nearer_the_null():
    # Lower bound RR = 1.5  ->  1.5 + sqrt(1.5*0.5) = 2.36602540...
    estimate = an_estimate(
        point=math.log(2.0), lower=math.log(1.5), upper=math.log(2.7),
        scale="log_ratio",
    )
    result = e_value(estimate)
    assert result.e_value_ci_bound == pytest.approx(
        1.5 + math.sqrt(1.5 * 0.5), rel=1e-12
    )


def test_an_interval_covering_the_null_needs_no_confounding_to_explain_away():
    estimate = an_estimate(
        point=math.log(1.2), lower=math.log(0.9), upper=math.log(1.6),
        scale="log_ratio",
    )
    assert e_value(estimate).e_value_ci_bound == 1.0


def test_a_protective_effect_gives_the_same_e_value_as_its_reciprocal():
    # E-values are symmetric about the null: RR = 0.5 and RR = 2 are equally
    # hard to explain away.
    harmful = an_estimate(
        point=math.log(2.0), lower=math.log(1.5), upper=math.log(2.7),
        scale="log_ratio",
    )
    protective = an_estimate(
        point=math.log(0.5), lower=math.log(1 / 2.7), upper=math.log(1 / 1.5),
        scale="log_ratio",
    )
    assert e_value(protective).e_value == pytest.approx(e_value(harmful).e_value)


def test_a_binary_outcome_uses_the_exact_ratio_the_counterfactual_carries():
    # p0 = 0.20, effect = 0.10  ->  RR = 0.30/0.20 = 1.5 exactly, no conversion.
    # Here observed_mean happens to equal baseline + point, which is what makes
    # this fixture self-consistent -- see the real-fit tests below for the
    # estimators where it does not.
    estimate = an_estimate(
        point=0.10, lower=0.04, upper=0.16, scale="percentage_point",
        counterfactual=CounterfactualPoint(
            observed_mean=0.30, predicted_absent_treatment=0.20,
            lower=0.17, upper=0.23,
        ),
    )
    result = e_value(estimate)
    assert result.e_value == pytest.approx(1.5 + math.sqrt(1.5 * 0.5), rel=1e-12)
    assert result.approximated is False
    # The note is still populated: the derivation is not the trivial exp(point),
    # and a reader should be able to see which ratio was used.
    assert "reported effect" in (result.approximation_note or "")


def test_an_effect_more_negative_than_the_baseline_falls_back_to_the_conversion():
    # A ratio must be positive, and baseline + point here is -0.05. Inverting
    # or clipping it would report an E-value for a quantity that does not
    # exist; the honest response is the conversion path with its note.
    estimate = an_estimate(
        point=-0.25, lower=-0.35, upper=-0.15, scale="percentage_point",
        counterfactual=CounterfactualPoint(
            observed_mean=0.05, predicted_absent_treatment=0.20,
            lower=0.17, upper=0.23,
        ),
    )
    result = e_value(estimate, outcome_sd=0.5)
    assert result.approximated is True
    assert np.isfinite(result.e_value)


def test_a_continuous_effect_uses_the_standardised_conversion_and_says_so():
    # d = 0.4 / 2.0 = 0.2  ->  RR = exp(0.91 * 0.2) = 1.19989...
    estimate = an_estimate(point=0.4, lower=0.1, upper=0.7, scale="absolute")
    result = e_value(estimate, outcome_sd=2.0)
    expected_rr = math.exp(0.91 * 0.2)
    assert result.e_value == pytest.approx(
        expected_rr + math.sqrt(expected_rr * (expected_rr - 1)), rel=1e-12
    )
    assert result.approximated is True
    assert "0.91" in (result.approximation_note or "")


def test_a_continuous_effect_without_an_outcome_sd_is_refused():
    estimate = an_estimate(point=0.4, lower=0.1, upper=0.7, scale="absolute")
    with pytest.raises(ValueError, match="outcome_sd"):
        e_value(estimate)


def test_a_non_positive_outcome_sd_is_refused():
    estimate = an_estimate(point=0.4, lower=0.1, upper=0.7, scale="absolute")
    with pytest.raises(ValueError, match="outcome_sd"):
        e_value(estimate, outcome_sd=0.0)


def test_a_zero_control_mean_falls_back_rather_than_dividing_by_zero():
    # A risk ratio is undefined against a zero baseline risk. The honest
    # response is the conversion path with its note, not inf.
    estimate = an_estimate(
        point=0.10, lower=0.04, upper=0.16, scale="percentage_point",
        counterfactual=CounterfactualPoint(
            observed_mean=0.10, predicted_absent_treatment=0.0, lower=0.0, upper=0.0,
        ),
    )
    result = e_value(estimate, outcome_sd=0.5)
    assert result.approximated is True
    assert np.isfinite(result.e_value)


def test_the_published_worked_example_reproduces():
    # VanderWeele & Ding (2017)'s own leukemia example, reproduced verbatim in
    # the EValue R package's documentation (the `evalue` help page, maintained
    # by VanderWeele's group): `evalue(RR(0.80), 0.71, 0.91)`. RR = 0.80
    # (protective) with a 95% CI of (0.71, 0.91).
    #
    # The original version of this test asserted E == 1.98 for RR =
    # 1.61, attributed to this same paper. Neither `1.61 + sqrt(1.61*0.61)`
    # (2.601) nor the CI-bound reading `1.30 + sqrt(1.30*0.30)` (1.925) lands
    # on 1.98, so that recollection was wrong and has been replaced with this
    # pair, which is hand-computable from the source's own documented example.
    estimate = an_estimate(
        point=math.log(0.80), lower=math.log(0.71), upper=math.log(0.91),
        scale="log_ratio",
    )
    result = e_value(estimate)
    # RR = 0.80 < 1 -> invert: 1/0.80 = 1.25.
    # E = 1.25 + sqrt(1.25 * 0.25) = 1.25 + sqrt(0.3125) = 1.80901699...
    assert result.e_value == pytest.approx(1.25 + math.sqrt(0.3125), rel=1e-9)
    # The bound nearer the null (1) for a protective effect is the upper
    # limit, 0.91 -> invert: 1/0.91 = 1.09890109...
    # E = 1/0.91 + sqrt((1/0.91) * (1/0.91 - 1)) = 1.42856...
    inv_hi = 1 / 0.91
    assert result.e_value_ci_bound == pytest.approx(
        inv_hi + math.sqrt(inv_hi * (inv_hi - 1)), rel=1e-9
    )


# The fixtures above are self-consistent by construction: `observed_mean`
# equals `predicted_absent_treatment + point`, so the arm-mean form and the
# effect-derived form agree and neither can fail. Two shipped estimators break
# that identity on real data, and only a real fit shows it.
#
# - `lin_regression` reports raw arm means alongside a Lin-ADJUSTED point.
# - `ipw` reports an unweighted treated mean alongside a Hajek-WEIGHTED
#   baseline, and an adjusted point.
#
# On scenario G at n_units=600, seed 1, the arm-mean form reported E = 2.338
# for an ipw point of 0.00077 -- an estimate flat at the null presented as
# needing a confounder association of RR >= 2.34 to explain away. The error
# always runs toward overstating robustness.
REAL_FIT_ESTIMATORS = ("lin_regression", "ipw")


def a_confounded_dataset():
    return (
        build_scenario("G")
        .generate(n_units=600, rng=np.random.default_rng(1))
        .dataset
    )


@pytest.mark.parametrize("estimator_name", REAL_FIT_ESTIMATORS)
def test_a_real_fit_takes_its_ratio_from_the_estimate_not_the_raw_arm_means(
    estimator_name,
):
    config = EstimatorConfig(bootstrap_reps=25, random_seed=1)
    estimate = all_estimators()[estimator_name]().fit(a_confounded_dataset(), config)
    counterfactual = estimate.counterfactual
    assert counterfactual is not None
    baseline = counterfactual.predicted_absent_treatment

    ratio = (baseline + estimate.point) / baseline
    expected = ratio + math.sqrt(ratio * (ratio - 1.0)) if ratio >= 1.0 else None
    if expected is None:  # pragma: no cover - both estimators are positive here
        inverted = 1.0 / ratio
        expected = inverted + math.sqrt(inverted * (inverted - 1.0))

    result = e_value(estimate)
    assert result.e_value == pytest.approx(expected, rel=1e-12)
    assert result.approximated is False

    # And the arm-mean form is genuinely a different number on this data, so
    # the assertion above is not passing vacuously.
    arm_ratio = counterfactual.observed_mean / baseline
    assert arm_ratio != pytest.approx(ratio, rel=1e-6)


@pytest.mark.parametrize("estimator_name", ("aipw", "difference_in_means"))
def test_an_estimator_whose_baseline_comes_from_its_point_is_unchanged(
    estimator_name,
):
    # The other half of the claim: where the counterfactual IS derived from the
    # reported point, the two forms are the same number, so this fix moved
    # nothing for these estimators.
    config = EstimatorConfig(bootstrap_reps=25, random_seed=1)
    estimate = all_estimators()[estimator_name]().fit(a_confounded_dataset(), config)
    counterfactual = estimate.counterfactual
    assert counterfactual is not None
    baseline = counterfactual.predicted_absent_treatment
    assert counterfactual.observed_mean == pytest.approx(
        baseline + estimate.point, rel=1e-12
    )
