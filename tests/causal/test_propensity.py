import numpy as np
import pytest
from cp_causal.propensity import (
    effective_sample_size,
    fit_propensity,
    overlap_checks,
    standardised_mean_differences,
)


def confounded(n: int = 4000, strength: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    covariates = np.column_stack([rng.normal(size=n), rng.binomial(1, 0.4, n)])
    logit = strength * (covariates @ np.array([0.9, 0.7]))
    propensity = 1.0 / (1.0 + np.exp(-logit))
    treatment = rng.binomial(1, propensity).astype(float)
    return covariates, treatment, propensity


def test_fitted_scores_track_the_true_propensity():
    covariates, treatment, true_ps = confounded()

    fitted = fit_propensity(covariates, treatment)

    assert np.corrcoef(fitted.scores, true_ps)[0, 1] > 0.95


def test_scores_stay_strictly_inside_the_unit_interval():
    covariates, treatment, _ = confounded(strength=3.0)

    fitted = fit_propensity(covariates, treatment)

    assert fitted.scores.min() > 0.0
    assert fitted.scores.max() < 1.0


def test_cross_fitting_produces_out_of_fold_scores():
    covariates, treatment, true_ps = confounded()

    fitted = fit_propensity(
        covariates, treatment, cross_fit_folds=5, rng=np.random.default_rng(3)
    )

    assert fitted.scores.shape == treatment.shape
    assert np.corrcoef(fitted.scores, true_ps)[0, 1] > 0.9


def test_cross_fitting_is_deterministic_for_a_fixed_seed():
    covariates, treatment, _ = confounded()

    first = fit_propensity(
        covariates, treatment, cross_fit_folds=5, rng=np.random.default_rng(7)
    )
    second = fit_propensity(
        covariates, treatment, cross_fit_folds=5, rng=np.random.default_rng(7)
    )

    np.testing.assert_array_equal(first.scores, second.scores)


def test_cross_fitting_requires_an_explicit_rng():
    # Fold assignment is random, so it must derive from the caller's seed chain
    # rather than a generator the function seeds for itself.
    covariates, treatment, _ = confounded()

    with pytest.raises(ValueError, match="rng"):
        fit_propensity(covariates, treatment, cross_fit_folds=5)


def test_weighting_reduces_covariate_imbalance():
    covariates, treatment, _ = confounded()
    fitted = fit_propensity(covariates, treatment)
    weights = np.where(treatment == 1, 1.0 / fitted.scores, 1.0 / (1.0 - fitted.scores))

    before = np.abs(standardised_mean_differences(covariates, treatment))
    after = np.abs(standardised_mean_differences(covariates, treatment, weights))

    assert before.max() > 0.3
    assert after.max() < 0.1


def test_effective_sample_size_is_at_most_the_actual_size():
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    assert effective_sample_size(weights) == pytest.approx(4.0)

    skewed = np.array([100.0, 1.0, 1.0, 1.0])
    assert effective_sample_size(skewed) < 2.0


def test_good_overlap_passes_the_checks():
    covariates, treatment, _ = confounded(strength=0.5)
    fitted = fit_propensity(covariates, treatment)
    smd = standardised_mean_differences(covariates, treatment)

    checks = {c.name: c for c in overlap_checks(fitted.scores, treatment, smd * 0.05)}

    assert checks["positivity"].status == "PASS"


def test_severe_non_overlap_fails_the_checks():
    # Extreme separation: propensity scores pile up at both boundaries, so
    # some units have almost no counterpart in the other arm.
    covariates, treatment, _ = confounded(strength=12.0, seed=5)
    fitted = fit_propensity(covariates, treatment)
    smd = standardised_mean_differences(covariates, treatment)

    checks = {c.name: c for c in overlap_checks(fitted.scores, treatment, smd)}

    assert checks["positivity"].status in {"WARN", "FAIL"}


def test_unconfoundedness_is_always_reported_as_not_testable():
    covariates, treatment, _ = confounded()
    fitted = fit_propensity(covariates, treatment)
    smd = standardised_mean_differences(covariates, treatment)

    checks = {c.name: c for c in overlap_checks(fitted.scores, treatment, smd)}

    assert checks["unconfoundedness"].status == "NOT_TESTABLE"
