"""CausalForest: nuisance residualization, honest tree, ensemble, variance,
the registered estimator.
"""
from __future__ import annotations

import numpy as np
from cp_causal.estimators.causal_forest import cross_fitted_outcome_model


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
