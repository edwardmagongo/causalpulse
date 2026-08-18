"""Honest causal forest: R-learner residualization, hand-rolled honest
recursive splitting, ensemble aggregation, infinitesimal-jackknife variance.

Guards against fabricated subgroup findings (omnibus calibration test,
minimum cell size, Benjamini-Hochberg FDR control) live in
`heterogeneity_checks.py`. scikit-learn fits only the R-learner's nuisance
models (`cross_fitted_outcome_model` here, `fit_propensity` from
`propensity.py`) -- the splits themselves, which determine the CATE
estimates, are hand-rolled from primitives, the same treatment every other
causal quantity in this project gets.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from cp_causal.propensity import fit_propensity


def cross_fitted_outcome_model(
    covariates: np.ndarray,
    outcome: np.ndarray,
    *,
    cross_fit_folds: int,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Out-of-fold predictions of `m(x) = E[Y|X]`.

    Mirrors `propensity.fit_propensity`'s cross-fitting discipline exactly
    (explicit `rng`, `cross_fit_folds`), but fits a single combined
    regression rather than per-treatment-arm potential outcomes --
    `doubly_robust.py`'s `_cross_fitted_outcome_models` doesn't fit this
    shape, since AIPW needs potential outcomes per arm and the R-learner
    needs one combined regression to residualize against.
    """
    n = covariates.shape[0]
    if cross_fit_folds > 1 and rng is None:
        raise ValueError(
            "cross_fit_folds > 1 requires an explicit rng: the fold assignment "
            "is random and must derive from the caller's SeedSequence rather "
            "than a generator seeded here"
        )
    predictions = np.zeros(n)
    if cross_fit_folds > 1:
        assert rng is not None
        order = rng.permutation(n)
        for fold in range(cross_fit_folds):
            held_out = order[fold::cross_fit_folds]
            train = np.setdiff1d(order, held_out)
            model = LinearRegression().fit(covariates[train], outcome[train])
            predictions[held_out] = model.predict(covariates[held_out])
    else:
        model = LinearRegression().fit(covariates, outcome)
        predictions = model.predict(covariates)
    return predictions


def residualize(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    cross_fit_folds: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """R-learner residualization: `Y_res = Y - m(X)`, `W_res = W - e(X)`.

    Both nuisance models are cross-fitted (out-of-fold predictions only) --
    reusing in-sample predictions would reintroduce the same overfitting
    bias AIPW's own cross-fitting exists to remove. This is the "R-learner-
    style local centering" the parent spec calls for: residuals orthogonal
    to X on average mean a split on the residualized pseudo-effect isn't
    confounded by X's direct effect on Y.
    """
    m_hat = cross_fitted_outcome_model(
        covariates, outcome, cross_fit_folds=cross_fit_folds, rng=rng
    )
    propensity_fit = fit_propensity(
        covariates, treatment, cross_fit_folds=cross_fit_folds, rng=rng
    )
    y_res = outcome - m_hat
    w_res = treatment - propensity_fit.scores
    return y_res, w_res
