"""CausalForest: nuisance residualization, honest tree, ensemble, variance,
the registered estimator.
"""
from __future__ import annotations

import numpy as np
from cp_causal.estimators.causal_forest import (
    assign_estimation_leaves,
    build_honest_tree,
    cross_fitted_outcome_model,
    leaf_tau,
    predict_leaf_tau,
    residualize,
)


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
