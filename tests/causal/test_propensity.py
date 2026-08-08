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


def test_smd_denominator_is_the_unweighted_pooled_sd():
    # Pins the module's load-bearing design choice: the SMD denominator is the
    # *unweighted* pooled SD, so a before/after comparison stays on one scale
    # even though the numerator (the mean) does use the weights. Nothing here
    # calls standardised_mean_differences twice and compares relative
    # movement (that would pass under a weighted-SD denominator too) — the
    # expected number is computed independently by hand below, so a change to
    # a weighted denominator would produce a visibly different number and
    # fail this test.
    #
    # One covariate, two units per arm:
    #   treated x = [10, 20], weights = [1, 3]
    #   control x = [0, 10],  weights = [3, 1]
    covariates = np.array([[10.0], [20.0], [0.0], [10.0]])
    treatment = np.array([1, 1, 0, 0])
    weights = np.array([1.0, 3.0, 3.0, 1.0])

    # Numerator: weighted means.
    #   treated: (10*1 + 20*3) / (1+3) = 70/4 = 17.5
    #   control: (0*3 + 10*1) / (3+1)  = 10/4 = 2.5
    #   weighted mean difference = 17.5 - 2.5 = 15.0
    #
    # Denominator: UNWEIGHTED pooled SD (ddof=1), ignoring the weights
    # entirely:
    #   treated var = ((10-15)^2 + (20-15)^2) / (2-1) = (25+25)/1 = 50
    #   control var = ((0-5)^2   + (10-5)^2)  / (2-1) = (25+25)/1 = 50
    #   pooled sd = sqrt((50+50)/2) = sqrt(50) = 7.0710678118654755
    #
    # Expected SMD = 15.0 / sqrt(50) = 2.1213203435596424
    expected_unweighted_denominator_smd = 15.0 / np.sqrt(50.0)

    # For contrast (not asserted): a WEIGHTED pooled SD would use each arm's
    # own weights in the variance instead of ddof=1 sample variance, e.g.
    #   treated weighted var = (1*(10-17.5)^2 + 3*(20-17.5)^2) / 4
    #                         = (1*56.25 + 3*6.25) / 4 = 75/4 = 18.75
    #   control weighted var = (3*(0-2.5)^2 + 1*(10-2.5)^2) / 4
    #                         = (3*6.25 + 1*56.25) / 4 = 75/4 = 18.75
    #   weighted pooled sd = sqrt((18.75+18.75)/2) = sqrt(18.75) = 4.330127...
    #   weighted-denominator SMD = 15.0 / 4.330127... = 3.4641016...
    # That is materially different from 2.1213203... (about 63% larger), so
    # this fixture would catch a switch to a weighted denominator.

    smd = standardised_mean_differences(covariates, treatment, weights)

    assert smd == pytest.approx([expected_unweighted_denominator_smd], rel=1e-9)


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


def _overlap_scores(
    n_treated_mid, n_treated_low, n_treated_high, n_control_mid, n_control_low, n_control_high
):
    """Build synthetic propensity scores with exact control over the
    positivity diagnostic's two rates.

    Every arm gets at least one score at exactly 0.01 (the low extreme) and
    one at exactly 0.99 (the high extreme), so `shared_low`/`shared_high` in
    `overlap_checks` land on those exact values for *both* arms. Every score
    used here is one of {0.01, 0.5, 0.99}, all inside [shared_low,
    shared_high], so the "outside common support" rate is always exactly 0 —
    this isolates the "extreme" rate as the only thing driving the status.
    Comparisons in the source are `<` / `>` (strict), not `<=` / `>=`, so a
    value exactly at a bound is never counted as outside.
    """
    treated = np.concatenate(
        [np.full(n_treated_low, 0.01), np.full(n_treated_high, 0.99), np.full(n_treated_mid, 0.5)]
    )
    control = np.concatenate(
        [np.full(n_control_low, 0.01), np.full(n_control_high, 0.99), np.full(n_control_mid, 0.5)]
    )
    scores = np.concatenate([treated, control])
    treatment = np.concatenate([np.ones_like(treated), np.zeros_like(control)])
    return scores, treatment


def test_positivity_lands_in_warn_when_neither_cutoff_is_crossed():
    # 500 per arm, 25 extreme per arm (15 low + 10 high) => 50/1000 = 0.05
    # extreme overall: above the WARN threshold (0.01) but at or below the
    # FAIL threshold (0.10). Outside-common-support is 0 by construction (see
    # _overlap_scores). So this pins the middle band exactly, with strict
    # equality rather than the `in {"WARN", "FAIL"}` membership check that
    # let the existing severe-non-overlap test pass trivially under FAIL.
    scores, treatment = _overlap_scores(475, 15, 10, 475, 15, 10)
    smd = np.array([0.01])

    checks = {c.name: c for c in overlap_checks(scores, treatment, smd)}

    assert checks["positivity"].statistic == pytest.approx(0.05)
    assert checks["positivity"].status == "WARN"


def test_positivity_stays_pass_exactly_at_the_warn_boundary():
    # 500 per arm, 5 extreme per arm (3 low + 2 high) => 10/1000 = 0.01
    # extreme overall: exactly equal to the WARN threshold. The source
    # compares with strict `>`, so a rate equal to the threshold must not
    # trigger WARN — this pins that the boundary itself still reads PASS.
    scores, treatment = _overlap_scores(495, 3, 2, 495, 3, 2)
    smd = np.array([0.01])

    checks = {c.name: c for c in overlap_checks(scores, treatment, smd)}

    assert checks["positivity"].statistic == pytest.approx(0.01)
    assert checks["positivity"].status == "PASS"


def test_positivity_fails_just_past_the_fail_boundary():
    # 500 per arm, 55 extreme per arm (30 low + 25 high) => 110/1000 = 0.11
    # extreme overall: just above the FAIL threshold (0.10). Pins the FAIL
    # band with strict equality, independent of the WARN band above it.
    scores, treatment = _overlap_scores(445, 30, 25, 445, 30, 25)
    smd = np.array([0.01])

    checks = {c.name: c for c in overlap_checks(scores, treatment, smd)}

    assert checks["positivity"].statistic == pytest.approx(0.11)
    assert checks["positivity"].status == "FAIL"


def test_unconfoundedness_is_always_reported_as_not_testable():
    covariates, treatment, _ = confounded()
    fitted = fit_propensity(covariates, treatment)
    smd = standardised_mean_differences(covariates, treatment)

    checks = {c.name: c for c in overlap_checks(fitted.scores, treatment, smd)}

    assert checks["unconfoundedness"].status == "NOT_TESTABLE"
