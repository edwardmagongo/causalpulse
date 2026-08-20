"""CausalForest: nuisance residualization, honest tree, ensemble, variance,
the registered estimator.
"""
from __future__ import annotations

import numpy as np
import pytest
from cp_causal.estimators.causal_forest import (
    CausalForest,
    ForestFit,
    aggregate_effect,
    assign_estimation_leaves,
    build_honest_tree,
    cross_fitted_outcome_model,
    fit_forest,
    infinitesimal_jackknife_terms,
    infinitesimal_jackknife_variance,
    leaf_tau,
    oob_cate,
    predict_leaf_tau,
    residualize,
)
from cp_core.values import EstimatorConfig
from cp_simulation.cate_scenarios import CATE_N_UNITS, build_cate_scenario


def test_cross_fitted_outcome_model_predictions_are_out_of_fold():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=(n, 3))
    y = x[:, 0] + rng.normal(scale=0.1, size=n)

    predictions = cross_fitted_outcome_model(x, y, cross_fit_folds=5, rng=rng)

    assert predictions.shape == (n,)
    # A cross-fitted model should recover the true linear relationship
    # reasonably well without having been evaluated in-sample.
    residual_sd = float(np.std(y - predictions))
    assert residual_sd < 0.2, f"residual_sd={residual_sd:.3f}, expected close to noise sd 0.1"


def test_cross_fitted_outcome_model_requires_an_explicit_rng_for_more_than_one_fold():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 2))
    y = rng.normal(size=100)
    try:
        cross_fitted_outcome_model(x, y, cross_fit_folds=5, rng=None)  # type: ignore[arg-type]
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_residualize_returns_arrays_shaped_like_the_input():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=(n, 3))
    w = rng.binomial(1, 0.5, n)
    y = x[:, 0] + w * (1 + x[:, 1]) + rng.normal(scale=0.1, size=n)

    y_res, w_res = residualize(x, w, y, cross_fit_folds=5, rng=rng)

    assert y_res.shape == (n,)
    assert w_res.shape == (n,)
    # w_res should average near zero under random assignment (propensity ~0.5).
    assert abs(float(w_res.mean())) < 0.1


def test_honest_tree_recovers_a_clean_binary_split():
    # A single binary covariate perfectly determines tau: 0.0 vs 0.5.
    rng = np.random.default_rng(0)
    n = 400
    X = np.column_stack([rng.binomial(1, 0.5, n).astype(float), rng.normal(size=n)])
    W_res = rng.normal(scale=0.3, size=n)  # pretend-residualized treatment
    true_tau = np.where(X[:, 0] == 1, 0.5, 0.0)
    Y_res = true_tau * W_res + rng.normal(scale=0.01, size=n)

    idx = np.arange(n)
    structure_idx, estimation_idx = idx[: n // 2], idx[n // 2 :]

    tree = build_honest_tree(
        X, Y_res, W_res, structure_idx, min_leaf_structure=20, max_depth=3
    )
    assign_estimation_leaves(tree, X, estimation_idx)

    pred_high = predict_leaf_tau(tree, np.array([1.0, 0.0]), Y_res, W_res)
    pred_low = predict_leaf_tau(tree, np.array([0.0, 0.0]), Y_res, W_res)
    assert pred_high > 0.3, f"expected the high-tau leaf near 0.5, got {pred_high}"
    assert pred_low < 0.2, f"expected the low-tau leaf near 0.0, got {pred_low}"


def test_leaf_tau_returns_zero_for_a_degenerate_index_set():
    Y_res = np.array([1.0, 2.0])
    W_res = np.array([0.0, 0.0])  # zero W_res everywhere -> zero denominator
    assert leaf_tau(Y_res, W_res, np.array([0, 1])) == 0.0


def test_fit_forest_returns_a_forestfit_with_consistent_shapes():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 3))
    W = rng.binomial(1, 0.5, n)
    Y = X[:, 0] + W * (1 + X[:, 1]) + rng.normal(scale=0.1, size=n)

    fit = fit_forest(
        X, W, Y, n_trees=20, min_leaf_structure=10, max_depth=3,
        subsample_frac=0.5, cross_fit_folds=5, rng=rng,
    )

    assert isinstance(fit, ForestFit)
    assert fit.cate.shape == (n,)
    assert fit.per_tree_predictions.shape == (20, n)
    assert fit.membership.shape == (20, n)
    assert fit.membership.dtype == bool
    assert fit.propensity_scores.shape == (n,)


def test_oob_cate_uses_only_trees_that_excluded_the_unit():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 3))
    W = rng.binomial(1, 0.5, n)
    Y = X[:, 0] + W * (1 + X[:, 1]) + rng.normal(scale=0.1, size=n)

    fit = fit_forest(
        X, W, Y, n_trees=50, min_leaf_structure=10, max_depth=3,
        subsample_frac=0.5, cross_fit_folds=5, rng=rng,
    )
    oob = oob_cate(fit)
    assert oob.shape == (n,)
    # Every unit should have at least some out-of-bag trees at 50%
    # subsampling and 50 trees (near-certain, not asserting a specific count).
    assert np.all(np.isfinite(oob))


def test_infinitesimal_jackknife_variance_is_non_negative_and_finite():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 3))
    W = rng.binomial(1, 0.5, n)
    Y = X[:, 0] + W * (1 + X[:, 1]) + rng.normal(scale=0.1, size=n)

    fit = fit_forest(
        X, W, Y, n_trees=200, min_leaf_structure=10, max_depth=3,
        subsample_frac=0.5, cross_fit_folds=5, rng=rng,
    )
    variances = infinitesimal_jackknife_variance(fit)
    assert variances.shape == (n,)
    assert np.all(variances >= 0)
    assert np.all(np.isfinite(variances))
    # This is the acceptance bar the "Prototype findings" section flagged as
    # needing task-time derivation: at least some units must show a
    # strictly positive variance estimate, or the formula is still the
    # broken always-zero one measured during plan-writing.
    assert np.any(variances > 0), (
        "infinitesimal_jackknife_variance returned all zeros -- this is the "
        "exact failure mode measured during plan-writing (see this plan's "
        "Prototype findings section); the naive bias-corrected formula "
        "cannot be used as-is"
    )
    # NOTE: this test does not pin down the bias-correction factor. The
    # objective-Bayes debiasing step is strictly positive by construction, so
    # `np.any(variances > 0)` holds under the correct p*(1-p) factor and under
    # the broken bootstrap factor alike. The factor itself is guarded by
    # test_infinitesimal_jackknife_correction_cancels_the_raw_term_under_independence.


def _independent_forest_fit(
    n: int, n_trees: int, subsample_frac: float, seed: int
) -> ForestFit:
    """A ForestFit whose per-tree predictions are pure noise, drawn with no
    reference whatsoever to which units each tree subsampled.

    Membership is drawn the way `fit_forest` draws it (without replacement,
    fixed subsample size `subsample_frac * n` per tree); predictions are iid
    normal. `y_res`/`w_res` are unused by the variance path and are filled
    with zeros. Because prediction and membership are independent by
    construction, the true infinitesimal-jackknife variance is zero and the
    entire raw term is Monte-Carlo noise -- which is exactly what the
    correction term is supposed to estimate.
    """
    rng = np.random.default_rng(seed)
    subsample_size = int(n * subsample_frac)
    membership = np.zeros((n_trees, n), dtype=bool)
    for tree_index in range(n_trees):
        membership[tree_index, rng.choice(n, size=subsample_size, replace=False)] = True
    predictions = rng.normal(size=(n_trees, n))
    return ForestFit(
        cate=predictions.mean(axis=0),
        per_tree_predictions=predictions,
        membership=membership,
        y_res=np.zeros(n),
        w_res=np.zeros(n),
        propensity_scores=np.full(n, 0.5),
    )


def test_infinitesimal_jackknife_correction_cancels_the_raw_term_under_independence():
    """Regression guard on the Monte-Carlo bias-correction factor.

    When per-tree predictions are generated independently of subsample
    membership there is no real signal for the raw infinitesimal-jackknife sum
    to carry: all of it is finite-B noise, so the correction term must match
    it, `raw / correction ~= 1`. That holds for the without-replacement factor
    `n * p * (1 - p) * v / B` the forest actually needs. The textbook
    *bootstrap* factor `n * v / B` -- the broken form measured during
    plan-writing -- is `1 / (p * (1 - p))` times too large, so it would drive
    this ratio to `p * (1 - p)`: 0.25 at p = 0.5, 0.21 at p = 0.3.

    Two subsample rates are checked precisely so the guard pins the *shape* of
    the factor and not just its value at one rate: a wrong-but-constant factor
    cannot hit 1.0 at both rates.
    """
    for subsample_frac in (0.5, 0.3):
        fit = _independent_forest_fit(
            n=300, n_trees=400, subsample_frac=subsample_frac, seed=17
        )
        raw, correction = infinitesimal_jackknife_terms(fit)

        assert np.all(correction > 0)
        ratio = float(np.mean(raw / correction))
        # Measured across 8 seeds and several (n, n_trees, p) configurations,
        # this mean ratio stays inside [0.98, 1.02]; the 10% band is slack.
        assert 0.9 < ratio < 1.1, (
            f"at subsample_frac={subsample_frac} the mean raw/correction ratio is "
            f"{ratio:.4f}, not ~1.0. Under independence the raw term is pure "
            f"Monte-Carlo noise and the correction must cancel it. A ratio near "
            f"{subsample_frac * (1 - subsample_frac):.2f} means the correction is "
            f"using the with-replacement bootstrap factor n*v/B instead of the "
            f"without-replacement factor n*p*(1-p)*v/B this forest requires."
        )


CONFIG = EstimatorConfig(
    random_seed=0, options={"n_trees": 50, "max_depth": 3, "min_leaf_structure": 10}
)


def _generate_cate(scenario_id: str, seed: int = 0):
    dgp = build_cate_scenario(scenario_id)
    return dgp.generate(CATE_N_UNITS, np.random.default_rng(seed))


def test_causal_forest_is_registered_under_its_own_name():
    from cp_causal.registry import get_estimator

    assert get_estimator("causal_forest") is CausalForest


def test_causal_forest_requirements_need_covariates_and_reject_panels():
    assert CausalForest.requirements.needs_covariates is True
    assert CausalForest.requirements.needs_panel is False


def test_causal_forest_fit_returns_ate_with_per_unit_cate_diagnostic():
    simulated = _generate_cate("RICH")
    result = CausalForest().fit(simulated.dataset, CONFIG)

    assert result.estimand == "ATE"
    assert result.ci.lower <= result.point <= result.ci.upper
    assert "per_unit_cate" in result.diagnostics
    per_unit = result.diagnostics["per_unit_cate"].value
    assert per_unit.shape == (CATE_N_UNITS,)
    variances = result.diagnostics["per_unit_cate_variance"].value
    assert variances.shape == (CATE_N_UNITS,)
    assert np.all(variances >= 0.0)


def test_causal_forest_fit_reports_subgroup_findings_as_a_tuple():
    simulated = _generate_cate("RICH")
    result = CausalForest().fit(simulated.dataset, CONFIG)
    assert isinstance(result.subgroup_findings, tuple)
    for finding in result.subgroup_findings:
        assert 0.0 <= finding.q_value <= 1.0


def test_causal_forest_fit_reports_the_calibration_test_assumption():
    simulated = _generate_cate("RICH")
    result = CausalForest().fit(simulated.dataset, CONFIG)
    names = {c.name for c in result.assumptions}
    assert "calibration_test" in names


def test_aggregate_effect_is_the_r_learner_score_not_the_mean_prediction():
    """The headline ATE must come from the residuals, not from `fit.cate`.

    This pins the *identity* of the estimator only. It cannot detect an
    interval that undercovers -- the reported SE's coverage is a rate, scored
    across seeds in
    `test_causal_forest_recovery.py::test_causal_forest_ate_interval_covers_the_true_ate`,
    and the formula this test excludes was the one that measured 0.35.
    """
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    W = rng.binomial(1, 0.5, n)
    Y = X[:, 0] + W * (1 + X[:, 1]) + rng.normal(scale=0.1, size=n)

    fit = fit_forest(
        X, W, Y, n_trees=20, min_leaf_structure=10, max_depth=3,
        subsample_frac=0.5, cross_fit_folds=5, rng=rng,
    )
    point, se = aggregate_effect(fit)

    expected = float(np.sum(fit.w_res * fit.y_res) / np.sum(fit.w_res**2))
    assert point == pytest.approx(expected)
    assert se > 0.0
    # Not the prediction average: those differ here, so a regression back to
    # `fit.cate.mean()` cannot slip through as numerically indistinguishable.
    assert abs(point - float(fit.cate.mean())) > 1e-6


def test_causal_forest_rejects_a_non_integer_tuning_option():
    simulated = _generate_cate("RICH")
    config = EstimatorConfig(random_seed=0, options={"n_trees": 50.0})

    with pytest.raises(TypeError, match="n_trees"):
        CausalForest().fit(simulated.dataset, config)
