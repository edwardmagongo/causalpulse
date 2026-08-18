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


class HonestNode:
    """One node of a hand-rolled honest regression tree.

    `__slots__` since a forest builds many thousands of these.
    """

    __slots__ = ("feature", "threshold", "left", "right", "leaf_idx")

    def __init__(self) -> None:
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: HonestNode | None = None
        self.right: HonestNode | None = None
        self.leaf_idx: np.ndarray | None = None


def leaf_tau(y_res: np.ndarray, w_res: np.ndarray, idx: np.ndarray) -> float:
    """The R-learner's closed-form leaf estimate: weighted least squares of
    `Y_res` on `W_res` alone, i.e. `sum(W_res*Y_res) / sum(W_res**2)`."""
    denom = float(np.sum(w_res[idx] ** 2))
    if denom < 1e-6 or idx.size < 2:
        return 0.0
    return float(np.sum(w_res[idx] * y_res[idx]) / denom)


def _node_objective(y_res: np.ndarray, w_res: np.ndarray, idx: np.ndarray) -> float:
    """Weighted sum of squared residuals of the R-learner pseudo-outcome
    around the node's own leaf_tau -- the quantity honest splitting
    minimizes, evaluated only on the structure sample."""
    tau = leaf_tau(y_res, w_res, idx)
    resid = y_res[idx] - tau * w_res[idx]
    return float(np.sum(resid**2))


def build_honest_tree(
    covariates: np.ndarray,
    y_res: np.ndarray,
    w_res: np.ndarray,
    structure_idx: np.ndarray,
    *,
    min_leaf_structure: int,
    max_depth: int,
    depth: int = 0,
) -> HonestNode:
    """Recursive binary splitting on the R-learner objective, evaluated only
    on `structure_idx` (the honest-splitting "structure sample"). The
    "estimation sample" that will actually populate each leaf's CATE
    estimate is assigned afterward by `assign_estimation_leaves` -- no
    observation ever does both jobs for the same tree.
    """
    node = HonestNode()
    n = structure_idx.size
    if depth >= max_depth or n < 2 * min_leaf_structure:
        node.leaf_idx = structure_idx
        return node

    n_features = covariates.shape[1]
    best_gain = -np.inf
    best_feature: int | None = None
    best_threshold: float | None = None
    parent_objective = _node_objective(y_res, w_res, structure_idx)

    for feature in range(n_features):
        values = covariates[structure_idx, feature]
        candidates = np.unique(
            np.quantile(values, [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
        )
        for threshold in candidates:
            left_mask = values <= threshold
            left_idx = structure_idx[left_mask]
            right_idx = structure_idx[~left_mask]
            if left_idx.size < min_leaf_structure or right_idx.size < min_leaf_structure:
                continue
            left_objective = _node_objective(y_res, w_res, left_idx)
            right_objective = _node_objective(y_res, w_res, right_idx)
            gain = parent_objective - (left_objective + right_objective)
            if gain > best_gain:
                best_gain = gain
                best_feature, best_threshold = feature, float(threshold)

    if best_feature is None or best_gain <= 0:
        node.leaf_idx = structure_idx
        return node

    node.feature = best_feature
    node.threshold = best_threshold
    left_mask = covariates[structure_idx, best_feature] <= best_threshold
    node.left = build_honest_tree(
        covariates, y_res, w_res, structure_idx[left_mask],
        min_leaf_structure=min_leaf_structure, max_depth=max_depth, depth=depth + 1,
    )
    node.right = build_honest_tree(
        covariates, y_res, w_res, structure_idx[~left_mask],
        min_leaf_structure=min_leaf_structure, max_depth=max_depth, depth=depth + 1,
    )
    return node


def assign_estimation_leaves(
    node: HonestNode, covariates: np.ndarray, estimation_idx: np.ndarray
) -> None:
    """Walk the estimation sample down the already-built tree structure,
    replacing each leaf's structure-sample indices with estimation-sample
    ones. Must be called once per tree, after `build_honest_tree`."""
    if node.feature is None:
        node.leaf_idx = estimation_idx
        return
    assert node.threshold is not None and node.left is not None and node.right is not None
    left_mask = covariates[estimation_idx, node.feature] <= node.threshold
    assign_estimation_leaves(node.left, covariates, estimation_idx[left_mask])
    assign_estimation_leaves(node.right, covariates, estimation_idx[~left_mask])


def predict_leaf_tau(
    node: HonestNode, x_row: np.ndarray, y_res: np.ndarray, w_res: np.ndarray
) -> float:
    """Walk a single covariate row down the tree, return that leaf's
    (estimation-sample) tau estimate."""
    while node.feature is not None:
        assert node.threshold is not None and node.left is not None and node.right is not None
        node = node.left if x_row[node.feature] <= node.threshold else node.right
    if node.leaf_idx is None or node.leaf_idx.size == 0:
        return 0.0
    return leaf_tau(y_res, w_res, node.leaf_idx)
