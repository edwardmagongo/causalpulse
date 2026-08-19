"""Guards against fabricated subgroup findings: omnibus calibration test,
best-linear-projection, minimum cell size, Benjamini-Hochberg FDR control.
"""

from __future__ import annotations

import numpy as np
from cp_causal.estimators.causal_forest import ForestFit, fit_forest
from cp_causal.heterogeneity_checks import (
    best_linear_projection,
    calibration_terms,
    calibration_test,
)
from cp_simulation.cate_scenarios import CATE_N_UNITS, build_cate_scenario

# Measured on 2026-08-19 in this worktree at min_leaf_structure=20, max_depth=4,
# subsample_frac=0.5, cross_fit_folds=5, CATE_N_UNITS=2000, data seed == forest
# seed, over 40 data seeds. "naive" is the first draft's statistic (no
# out-of-bag bias correction, OLS standard error, two-sided); "fixed" is what
# `calibration_test` now computes (numerator + sigma^2, standard error times
# sqrt(2), one-sided).
#
# The naive statistic, on the weaker first draft of the RICH scenario:
#
#                     t mean   t sd   |t|>1.96
#     NULL  naive     -1.274   1.556   12/40
#     RICH  naive     -0.195   1.670    9/40
#
# It rejected on 30% of NULL seeds at a nominal 5%, and every one of those
# rejections -- and every one of RICH's -- was NEGATIVE, i.e. the two-sided
# rule would have opened the subgroup gate on evidence that the forest's
# predicted ordering is inverted. That is what the +sigma^2, sqrt(2) and
# one-sided corrections fix; they are pinned directly, on a synthetic forest
# with a closed-form kernel, by the two `_leaking_forest_fit` tests below.
#
# The shipped statistic, on the scenarios as they now stand:
#
#              t mean   t sd    t min   PASS    corr(oob_cate, true_cate)
#     NULL     -0.303   0.973   -2.46   0/40    -0.040 [-0.467, +0.438]
#     RICH     +3.224   0.701   +1.75  40/40    +0.808 [+0.591, +0.935]
#
# Clean separation in both directions, so the two scenario tests below assert
# the real gate behaviour rather than only the checks' shape.
#
# This separation needs the scenario to carry heterogeneity above the detection
# floor. It did not always: an earlier draft of RICH had sd(tau) ~ 0.038
# against a binary outcome's sigma ~ 0.50, where an ORACLE holding the true
# per-unit tau detected on only 50% of seeds and this test on 0/12.
# `cate_scenarios._RICH_EFFECT` was raised on that measurement to sd(tau) ~
# 0.123; see its comment for the sweep. The lesson is worth keeping: a
# non-detection here is as likely to be the fixture's signal-to-noise as the
# statistic's correctness, and the two are told apart by the oracle
# substitution, not by inspection.
#
# The n_trees sweep at data seed 0 (on the earlier, weaker RICH -- the point it
# establishes is about tree count, not effect size):
#
#     n_trees   RICH naive  RICH fixed   NULL naive  NULL fixed
#         500      +1.268      +1.232       +1.409      +1.371
#        1000      +1.323      +1.292       +1.476      +1.426
#        2000      +1.234      +1.247       +1.461      +1.424
#
# Essentially invariant in n_trees -- a 4x increase moves it by <0.1 at a fixed
# seed -- so 500 is kept as the cheapest. The spread is across data seeds, not
# tree counts.
_FOREST_CONFIG: dict[str, object] = dict(
    n_trees=500, min_leaf_structure=20, max_depth=4, subsample_frac=0.5, cross_fit_folds=5,
)


def _fit_on_scenario(scenario_id: str, seed: int):
    dgp = build_cate_scenario(scenario_id)
    simulated = dgp.generate(CATE_N_UNITS, np.random.default_rng(seed))
    covariates = simulated.dataset.frame[list(simulated.dataset.schema.covariates)].to_numpy()
    treatment = simulated.dataset.frame[simulated.dataset.schema.treatment].to_numpy()
    outcome = simulated.dataset.frame[simulated.dataset.schema.outcome].to_numpy(dtype=float)
    fit = fit_forest(
        covariates, treatment, outcome, rng=np.random.default_rng(seed), **_FOREST_CONFIG
    )
    return fit, covariates


def test_calibration_test_detects_the_rich_scenarios_real_heterogeneity():
    """End-to-end, on the scenario per-unit CATE recovery is scored against.

    Measured over 40 seeds: t mean +3.224, sd 0.701, minimum +1.75, PASS on
    40/40. Seed 0 is asserted rather than the whole sweep because each seed
    costs a 500-tree forest fit; the sweep lives in this file's header.
    """
    fit, _ = _fit_on_scenario("RICH", seed=0)
    result = calibration_test(fit, alpha=0.05)

    assert result.name == "calibration_test"
    assert result.statistic is not None
    assert result.status == "PASS", (
        f"the calibration test did not detect RICH's heterogeneity "
        f"(t={result.statistic:.3f}). Subgroup discovery is gated on this check, "
        f"so a miss here shuts the whole subgroup path on the one scenario built "
        f"to exercise it."
    )


def test_calibration_test_reports_no_heterogeneity_on_the_null_scenario():
    """The guard this phase exists to prove: a scenario with a constant tau
    must not license subgroup reporting.

    Measured over 40 seeds: t mean -0.303, sd 0.973, PASS on 0/40. The naive
    statistic this replaced fired on 12/40, every one of them negative.
    """
    fit, _ = _fit_on_scenario("NULL", seed=0)
    result = calibration_test(fit, alpha=0.05)

    assert result.status == "FAIL", (
        f"the calibration test claimed detectable heterogeneity on a scenario "
        f"whose true tau is exactly constant (t={result.statistic:.3f}) -- this is "
        f"fabricated heterogeneity reaching a report."
    )
    assert result.remedy is not None


def test_best_linear_projection_returns_one_coefficient_per_covariate():
    fit, covariates = _fit_on_scenario("RICH", seed=0)
    names = ("x0", "x1", "x2", "x3", "x4", "x5")
    projection = best_linear_projection(fit, covariates, names)
    assert set(projection) == set(names)


_LEAK_N = 600
_LEAK_LEAVES = 12
_LEAK_SIGMA = 0.5
_LEAK_BASE_TAU = 0.05
_LEAK_REPS = 2000


def _leaking_forest_fit(seed: int, true_tau: np.ndarray) -> ForestFit:
    """A ForestFit whose out-of-bag CATE is an exact, closed-form honest leaf
    average of the very `y_res` the calibration regression regresses.

    This is the property that breaks the textbook statistic, isolated from
    everything else a real forest does. Units are partitioned into fixed
    leaves; each unit's prediction is its leaf's R-learner tau
    `sum(w_res*y_res) / sum(w_res**2)` computed with its OWN row removed, which
    is precisely the structure `fit_forest` + `oob_cate` produce -- a leaf
    kernel whose diagonal is zero and whose rows put total mass one on the
    other units, in the `w_res` inner product.

    No adaptive splitting, no subsampling, no residualization: the leaves are
    drawn at random and carry no covariate information, so the forest's
    predictions are *pure noise* and the true differential-calibration
    coefficient is exactly zero whenever `true_tau` is constant. Any rejection
    under a constant `true_tau` is a false positive, by construction rather
    than by measurement.

    Setting every tree's prediction row to the same vector and marking every
    unit out-of-bag in every tree makes `oob_cate` return that vector exactly,
    so the fixture drives the real `calibration_test` code path.
    """
    rng = np.random.default_rng(seed)
    w_res = rng.binomial(1, 0.5, _LEAK_N) - 0.5
    y_res = true_tau * w_res + rng.normal(scale=_LEAK_SIGMA, size=_LEAK_N)
    leaf = rng.integers(0, _LEAK_LEAVES, _LEAK_N)

    leaf_wy = np.bincount(leaf, weights=w_res * y_res, minlength=_LEAK_LEAVES)
    leaf_ww = np.bincount(leaf, weights=w_res**2, minlength=_LEAK_LEAVES)
    cate = (leaf_wy[leaf] - w_res * y_res) / (leaf_ww[leaf] - w_res**2)

    return ForestFit(
        cate=cate,
        per_tree_predictions=np.tile(cate, (2, 1)),
        membership=np.zeros((2, _LEAK_N), dtype=bool),
        y_res=y_res,
        w_res=w_res,
    )


def _leaf_driven_tau(seed: int, spread: float) -> np.ndarray:
    """The same leaf partition `_leaking_forest_fit` will draw for `seed`, used
    to give the units a true tau the forest's predictions genuinely track."""
    rng = np.random.default_rng(seed)
    rng.binomial(1, 0.5, _LEAK_N)
    rng.normal(scale=_LEAK_SIGMA, size=_LEAK_N)
    leaf = rng.integers(0, _LEAK_LEAVES, _LEAK_N)
    centred = (leaf - (_LEAK_LEAVES - 1) / 2) / _LEAK_LEAVES
    return _LEAK_BASE_TAU + spread * _LEAK_SIGMA * centred


def test_calibration_terms_numerator_carries_exactly_the_minus_sigma_squared_bias():
    """The identity the bias correction rests on, measured directly.

    `oob_cate` is a linear functional of `y_res`, so the calibration
    numerator is a quadratic form `y_res' B y_res` and `E[numerator] =
    sigma**2 * trace(B)`. Out-of-bag averaging zeroes that form's diagonal
    while each kernel row keeps total mass one, which fixes `trace(B) = -1`
    exactly -- not approximately, and not as a function of leaf count, noise
    level or sample size.
    """
    true_tau = np.full(_LEAK_N, _LEAK_BASE_TAU)
    numerators, residual_variances, denominators = [], [], []
    for rep in range(_LEAK_REPS):
        numerator, denominator, residual_variance = calibration_terms(
            _leaking_forest_fit(2000 + rep, true_tau)
        )
        numerators.append(numerator)
        denominators.append(denominator)
        residual_variances.append(residual_variance)
    numerators = np.array(numerators)
    sigma_squared = float(np.mean(residual_variances))

    ratio = float(np.mean(numerators)) / -sigma_squared
    # Measured 1.0091, with a Monte-Carlo standard error of 0.1045 at this
    # replication count. The seeds are fixed, so this is reproducible exactly.
    assert 0.80 < ratio < 1.25, (
        f"mean(numerator) / -sigma^2 is {ratio:.4f}, not ~1.0. The out-of-bag "
        f"calibration numerator has an exact -sigma^2 mean, and `calibration_test` "
        f"adds it back; if this identity no longer holds the correction it applies "
        f"is the wrong size."
    )
    inflation = float(np.var(numerators)) / (sigma_squared * float(np.mean(denominators)))
    # Measured 1.9376. OLS estimates only half of Var[numerator], because it
    # treats the numerator as linear in y_res when it is quadratic.
    assert 1.6 < inflation < 2.4, (
        f"var(numerator) / (sigma^2 * mean(denominator)) is {inflation:.4f}, not ~2.0. "
        f"This ratio is what the sqrt(2) standard-error inflation in "
        f"`calibration_test` exists to absorb."
    )


def test_calibration_test_corrects_the_out_of_bag_numerators_bias_and_variance():
    """Regression guard on BOTH correction constants at once.

    Under `_leaking_forest_fit` with a constant true tau there is no
    heterogeneity whatsoever, so the reported statistic must look like a
    standard normal: mean ~0, sd ~1. Measured over these 2000 fixed seeds, the
    shipped statistic gives mean -0.216 (Monte-Carlo se 0.024) and sd 1.054
    (se 0.017). The three ways of getting the corrections wrong land outside:

        variant                            mean t   sd t   one-sided rate
        shipped: +sigma^2 and sqrt(2)      -0.216   1.054     0.0255
        +sigma^2 but no sqrt(2)            -0.306   1.491     0.0805
        sqrt(2) but no +sigma^2            -0.441   1.103     0.0165
        neither (the original draft)       -0.623   1.559     0.0535

    Dropping the sqrt(2) is caught by the sd band and dropping the +sigma^2 by
    the mean band, so both constants are pinned and neither assertion is
    redundant. The mean sits slightly below zero because the statistic is a
    ratio whose numerator and denominator are positively correlated; that
    residual skew is conservative -- it pushes AWAY from declaring
    heterogeneity -- and is left in rather than tuned out.
    """
    true_tau = np.full(_LEAK_N, _LEAK_BASE_TAU)
    statistics = np.array(
        [
            calibration_test(_leaking_forest_fit(2000 + rep, true_tau)).statistic
            for rep in range(_LEAK_REPS)
        ]
    )

    mean_statistic = float(statistics.mean())
    standard_deviation = float(statistics.std())
    assert -0.35 < mean_statistic < 0.35, (
        f"under a forest with no heterogeneity at all the statistic averages "
        f"{mean_statistic:+.3f}, not ~0. A mean near -0.44 or -0.62 means the "
        f"numerator's -sigma^2 out-of-bag bias is no longer being added back."
    )
    assert 0.90 < standard_deviation < 1.25, (
        f"under a forest with no heterogeneity at all the statistic has sd "
        f"{standard_deviation:.3f}, not ~1. An sd near 1.5 means the standard "
        f"error is the plain OLS one again: the numerator is a quadratic form in "
        f"y_res, and its true variance is twice what OLS reports."
    )
    false_positive_rate = float((statistics > 1.645).mean())
    assert false_positive_rate < 0.05, (
        f"false-positive rate {false_positive_rate:.4f} at a nominal 0.05 "
        f"(measured 0.0255). This check gates subgroup reporting; over-rejection "
        f"here is fabricated heterogeneity reaching a report."
    )


def test_calibration_test_detects_heterogeneity_the_forest_genuinely_tracks():
    """The corrections must not have bought their size by killing all power.

    Same fixture, but the true tau now varies across the very leaves the
    predictions average over, so `oob_cate` genuinely tracks it. Measured PASS
    rates over 400 fixed seeds: 0.045 at spread 0 (the nominal size), 0.113 at
    0.5, 0.497 at 1.0, 1.000 at 2.0.
    """
    detections = sum(
        calibration_test(
            _leaking_forest_fit(2000 + rep, _leaf_driven_tau(2000 + rep, spread=2.0))
        ).status
        == "PASS"
        for rep in range(400)
    )
    assert detections == 400, (
        f"only {detections}/400 detections on a fixture whose out-of-bag CATE "
        f"tracks a true tau spanning four noise standard deviations -- the "
        f"statistic has been made conservative to the point of uselessness"
    )


def test_calibration_test_does_not_open_the_gate_on_an_inverted_prediction():
    """One-sidedness, which the first draft did not have.

    A large NEGATIVE differential coefficient says the forest's predicted
    ordering is backwards -- the units it calls high-effect are the low-effect
    ones. That is not evidence of usable heterogeneity, and it must not license
    subgroup reporting. Every one of the first draft's rejections, on both the
    null and the rich scenario, was negative, so a two-sided rule would have
    opened the gate on exactly this.

    Negating a fixture whose predictions DO track the truth is what inverts
    them; giving the true tau the opposite slope does not, since the leaf
    average tracks whatever tau it is handed. Measured over these 200 seeds the
    statistic averages -4.17 and never rises above -2.33.
    """
    negatives = 0
    for rep in range(200):
        tracking = _leaking_forest_fit(2000 + rep, _leaf_driven_tau(2000 + rep, spread=2.0))
        inverted = ForestFit(
            cate=-tracking.cate,
            per_tree_predictions=-tracking.per_tree_predictions,
            membership=tracking.membership,
            y_res=tracking.y_res,
            w_res=tracking.w_res,
        )
        result = calibration_test(inverted)
        assert result.status == "FAIL", (
            "an inverted out-of-bag CATE was reported as detectable heterogeneity; "
            "the calibration test must be one-sided"
        )
        negatives += result.statistic is not None and result.statistic < -1.96
    assert negatives == 200, (
        f"only {negatives}/200 fixtures produced a strongly negative statistic, so "
        f"this test is not actually exercising the inverted-prediction case"
    )
