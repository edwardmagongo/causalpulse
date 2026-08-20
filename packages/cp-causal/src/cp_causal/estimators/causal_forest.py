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

import time
from dataclasses import dataclass

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import EffectEstimate, FitMetadata
from cp_core.values import Diagnostic, EstimatorConfig
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from cp_causal._panel import outcome_scale
from cp_causal._support import arrays, config_hash, normal_p_value, wald_interval
from cp_causal.base import Requirements
from cp_causal.propensity import (
    fit_propensity,
    overlap_checks,
    standardised_mean_differences,
)
from cp_causal.registry import register


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


def _residualize_with_propensity(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    cross_fit_folds: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`residualize`, additionally returning the fitted propensity scores.

    Split out so `fit_forest` can carry the cross-fitted scores on `ForestFit`
    for the estimator's overlap diagnostics, without either refitting the
    propensity model a second time (a second, differently-seeded fit would
    diagnose overlap for a model the estimate was not built on) or changing
    `residualize`'s two-value contract. The draw order on `rng` is the outcome
    model first, then the propensity model -- unchanged from when this body
    lived in `residualize`, so every measured number downstream still holds.
    """
    m_hat = cross_fitted_outcome_model(
        covariates, outcome, cross_fit_folds=cross_fit_folds, rng=rng
    )
    propensity_fit = fit_propensity(
        covariates, treatment, cross_fit_folds=cross_fit_folds, rng=rng
    )
    y_res = outcome - m_hat
    w_res = treatment - propensity_fit.scores
    return y_res, w_res, propensity_fit.scores


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
    y_res, w_res, _ = _residualize_with_propensity(
        covariates, treatment, outcome, cross_fit_folds=cross_fit_folds, rng=rng
    )
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


def residual_effect(
    y_res: np.ndarray, w_res: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    """The R-learner effect on a subset of units, with a
    heteroskedasticity-robust standard error.

    `tau = sum(w_res * y_res) / sum(w_res**2)` is the least-squares
    coefficient of `y_res` on `w_res` over the subset -- exactly the
    quantity `leaf_tau` above computes inside a leaf, here applied to an
    arbitrary subset: a covariate-defined cell (`heterogeneity_checks`) or
    the whole sample (`CausalForest`'s headline ATE). The standard error is the usual sandwich form
    `sqrt(sum(w_res**2 * r**2)) / sum(w_res**2)` with `r` the residual from
    that fit, so a cell whose noise varies with treatment is not understated.

    Returns `(0.0, 0.0)` when the subset carries no residual treatment
    variation, which callers read as "no contrast identified" rather than as
    an effect of zero.
    """
    u = w_res[mask]
    denominator = float(np.sum(u**2))
    if denominator <= 0.0:
        return 0.0, 0.0
    tau = float(np.sum(u * y_res[mask]) / denominator)
    residual = y_res[mask] - tau * u
    return tau, float(np.sqrt(np.sum(u**2 * residual**2)) / denominator)


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


@dataclass(frozen=True, eq=False)
class ForestFit:
    """Everything downstream consumers (per-unit CATE, OOB calibration,
    variance) need from a fitted ensemble.

    `eq=False`: every field is an ndarray, so a generated `__eq__` would
    raise on the ambiguous truth value of an array rather than compare.
    """

    cate: np.ndarray
    per_tree_predictions: np.ndarray
    membership: np.ndarray
    y_res: np.ndarray
    w_res: np.ndarray
    propensity_scores: np.ndarray


def fit_forest(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    n_trees: int,
    min_leaf_structure: int,
    max_depth: int,
    subsample_frac: float,
    cross_fit_folds: int,
    rng: np.random.Generator,
) -> ForestFit:
    """Grow `n_trees` honest trees, each on its own without-replacement
    subsample of `subsample_frac * n` units, split in half into that tree's
    structure and estimation samples.

    Residualization happens once for the whole forest (the R-learner nuisance
    models are already cross-fitted, so they are not a source of per-tree
    in-sample bias); only the tree-growing subsample varies across trees.
    """
    n = covariates.shape[0]
    y_res, w_res, propensity_scores = _residualize_with_propensity(
        covariates, treatment, outcome, cross_fit_folds=cross_fit_folds, rng=rng
    )

    per_tree_predictions = np.full((n_trees, n), np.nan)
    membership = np.zeros((n_trees, n), dtype=bool)

    for tree_index in range(n_trees):
        subsample_size = int(n * subsample_frac)
        subsample = rng.choice(n, size=subsample_size, replace=False)
        membership[tree_index, subsample] = True
        rng.shuffle(subsample)
        half = subsample_size // 2
        structure_idx, estimation_idx = subsample[:half], subsample[half:]

        tree = build_honest_tree(
            covariates, y_res, w_res, structure_idx,
            min_leaf_structure=min_leaf_structure, max_depth=max_depth,
        )
        assign_estimation_leaves(tree, covariates, estimation_idx)

        for unit_index in range(n):
            per_tree_predictions[tree_index, unit_index] = predict_leaf_tau(
                tree, covariates[unit_index], y_res, w_res
            )

    cate = np.nanmean(per_tree_predictions, axis=0)
    return ForestFit(
        cate=cate,
        per_tree_predictions=per_tree_predictions,
        membership=membership,
        y_res=y_res,
        w_res=w_res,
        propensity_scores=propensity_scores,
    )


def oob_cate(fit: ForestFit) -> np.ndarray:
    """Per-unit CATE averaging only over trees where the unit was excluded
    from the subsample entirely -- the standard random-forest
    out-of-bag mechanism, extended to an honest forest. Used by the omnibus
    calibration test (`heterogeneity_checks.py`), never by the headline
    per-unit CATE (`ForestFit.cate`, which uses every tree for the tightest
    point estimate): evaluating a forest's own fit on itself is exactly the
    in-sample overfitting bias measured during plan-writing to produce a
    false positive on a null-heterogeneity scenario.
    """
    out_of_bag_mask = ~fit.membership
    masked = np.where(out_of_bag_mask, fit.per_tree_predictions, np.nan)
    return np.asarray(np.nanmean(masked, axis=0))


# Peak memory of the infinitesimal-jackknife covariance block is
# `_IJ_CHUNK * n * 8` bytes; 512 target units against a 100k-row training
# sample is ~400MB, and the full un-chunked (n, n) product would be 80GB.
_IJ_CHUNK = 512


def infinitesimal_jackknife_terms(fit: ForestFit) -> tuple[np.ndarray, np.ndarray]:
    """The two per-unit intermediate quantities `infinitesimal_jackknife_variance`
    differences, returned before any debiasing: `(raw, correction)`.

    Split out of `infinitesimal_jackknife_variance` so the Monte-Carlo
    bias-correction factor can be tested directly. The debiasing step applied
    downstream is strictly positive by construction, so it masks the
    difference between the correct `p * (1 - p)` (without-replacement) factor
    and the textbook bootstrap factor -- a test written against the final
    variances alone cannot tell them apart, but one written against these two
    terms can (see
    `test_infinitesimal_jackknife_correction_cancels_the_raw_term_under_independence`).

    `raw` is the rescaled infinitesimal-jackknife sum
    `scale * sum_j cov[i, j]**2` with

        cov[i, j] = (1/B) * sum_t (N[t, j] - p) * (tau_t(x_i) - taubar(x_i))

    the across-tree covariance between training unit `j`'s subsample
    membership and the tree's prediction at target unit `i`, and
    `scale = 1 / (1 - p)**2` the standard rescaling from a subsample of size
    `s = p * n` back to the full sample.

    `correction` is the finite-`B` Monte-Carlo bias of that sum: even when the
    prediction is independent of the membership indicators, the sampled
    covariances are nonzero noise, and their squares accumulate. For a
    *bootstrap* (with-replacement, Poisson) resample that inflation is
    `scale * n * v / B`; for the without-replacement subsample `fit_forest`
    actually draws it is `scale * n * p * (1 - p) * v / B`, where `v` is the
    across-tree variance of the prediction at `i`.
    """
    predictions = fit.per_tree_predictions
    membership = fit.membership.astype(np.float64)
    n_trees, n = predictions.shape
    inclusion_rate = float(membership.mean())

    if n_trees < 2:
        raise ValueError("infinitesimal-jackknife variance needs at least 2 trees")
    if not 0.0 < inclusion_rate < 1.0:
        raise ValueError(
            f"infinitesimal-jackknife variance needs a subsample rate strictly between "
            f"0 and 1, got {inclusion_rate}: with no variation in which units each tree "
            f"saw, the membership-prediction covariance carries no information"
        )

    # (n/(n - s))**2 == 1/(1 - p)**2, the standard rescaling from a
    # subsample of size s back to the full sample.
    scale = 1.0 / (1.0 - inclusion_rate) ** 2
    centered_predictions = predictions - predictions.mean(axis=0, keepdims=True)
    centered_membership = membership - membership.mean(axis=0, keepdims=True)

    raw = np.empty(n)
    for start in range(0, n, _IJ_CHUNK):
        block = centered_predictions[:, start : start + _IJ_CHUNK]
        covariance = (block.T @ centered_membership) / n_trees
        raw[start : start + _IJ_CHUNK] = scale * np.sum(covariance**2, axis=1)

    across_tree_variance = np.mean(centered_predictions**2, axis=0)
    correction = (
        scale * n * inclusion_rate * (1.0 - inclusion_rate) * across_tree_variance / n_trees
    )
    return raw, correction


def infinitesimal_jackknife_variance(fit: ForestFit) -> np.ndarray:
    """Per-unit sampling variance of the ensemble's CATE estimate.

    The estimator is the infinitesimal jackknife of Efron (2014) / Wager,
    Hastie & Efron (2014): for target unit `i`, the covariance across trees
    between a training unit `j`'s subsample membership and the tree's
    prediction at `i`,

        cov[i, j] = (1/B) * sum_t (N[t, j] - p) * (tau_t(x_i) - taubar(x_i))

    is the directional derivative of the ensemble prediction with respect to
    `j`'s sampling weight, and the variance is `sum_j cov[i, j]**2` rescaled
    for the subsampling rate `p = s/n`. `infinitesimal_jackknife_terms`
    computes that raw sum and its Monte-Carlo bias correction; this function
    differences and debiases them.

    Two corrections are applied on top of that raw sum, both of which the
    naive form measured during plan-writing got wrong:

    1. **The Monte-Carlo bias correction is the without-replacement one.**
       With finitely many trees, `sum_j cov[i, j]**2` is inflated by the
       sampling noise of the membership indicators even when the prediction
       is independent of them. For a *bootstrap* (with-replacement, Poisson)
       resample that inflation is `n * v / B`; for the without-replacement
       subsample this forest actually draws it is `n * p * (1 - p) * v / B`,
       where `v` is the across-tree variance of the prediction at `i`. The
       plan-writing attempt subtracted the bootstrap form from a
       without-replacement forest run at `p = 0.5`, i.e. a correction
       `1 / (p * (1 - p)) = 4x` too large, which is why the raw term sat at a
       near-constant ~0.27 (~1/4) of the correction at every tree count and
       the difference clamped to zero regardless of `B`.

       With the matching correction, both terms still decay at `1/B` -- `raw`
       measured 7.56e-2 at `B = 200` and 1.40e-2 at `B = 1600`, still falling
       toward its ~4.7e-3 asymptote -- but their *difference*, the quantity
       actually reported, stabilizes early, because the `1/B` component of
       `raw` is exactly what `correction` estimates and removes. It is the gap
       between the two terms that converges, not `raw` on its own.

    2. **The residual is debiased rather than clamped at zero.** Both terms
       are noisy, so their difference is negative for some units at any
       finite `B`. This follows `econml`'s own generalized-random-forest
       inference (`econml/grf/_base_grf.py`, `BaseGRF._predict_point_and_var`,
       the block commented "Objective bayes debiasing for the diagonals where
       we know a-prior they are positive"): treat the difference as a draw
       from `N(V, se**2)` with `se = max(raw, correction) * sqrt(2 / B)` and
       report the mean of that normal truncated to the positive half-line,
       `naive + se * phi(z) / Phi(z)` with `z = naive / se`. That inverse
       Mills ratio is strictly positive, so a unit whose point estimate
       happens to land below zero gets a small positive variance rather than
       a hard zero, and the adjustment vanishes as `B` grows and `se -> 0`.

    Only the objective-Bayes step is taken from `econml`; the covariance term
    itself is not `econml`'s estimator, which uses the bootstrap of little
    bags over groups of trees sharing a half-sample -- a grouping
    `fit_forest` does not build, since it draws every tree's subsample
    independently.

    **The variance is conditional on the fitted nuisance models.**
    `fit_forest` residualizes once on the full sample (`residualize`, i.e. the
    cross-fitted `m(x)` and `e(x)`) before drawing a single subsample, so
    `y_res` and `w_res` -- and therefore every tree's prediction -- depend on
    all `n` units through those nuisance fits. The only quantity that varies
    across trees is the subsample, so what this function measures is
    ensemble/subsampling variance *given* `m_hat` and `e_hat`; it does not
    propagate the estimation uncertainty of the nuisance models themselves.
    Intervals built from it are therefore conditional, not fully
    unconditional, and will run slightly narrow to the extent nuisance
    uncertainty matters. This matches standard practice (`econml` and `grf`
    both residualize once and report the same conditional variance), but
    callers that build confidence intervals or measure their coverage should
    know the omission is there.
    """
    raw, correction = infinitesimal_jackknife_terms(fit)
    n_trees = fit.per_tree_predictions.shape[0]
    n = raw.size

    naive = raw - correction
    standard_error = np.maximum(raw, correction) * np.sqrt(2.0 / n_trees)
    # A unit whose predictions never varied across trees has raw == correction
    # == 0 and a genuinely zero variance estimate; leave it there rather than
    # dividing by a floor.
    informative = standard_error > 0.0
    variance = np.zeros(n)
    z_score = naive[informative] / standard_error[informative]
    # logpdf - logcdf rather than econml's erfc ratio: both numerator and
    # denominator of that ratio underflow to 0.0 in the lower tail, and 0/0
    # in numpy is `nan` (with an invalid-value warning), not a fallback to
    # anything. The nan would propagate through `naive + se * nan` and through
    # `np.maximum(nan, 0.0)`, so the failure would surface as a non-finite
    # variance -- caught by the `np.isfinite` assertion in the covering test,
    # but only after having silently poisoned any interval built from it.
    # Working in log space keeps the ratio accurate that far out instead.
    inverse_mills = np.exp(norm.logpdf(z_score) - norm.logcdf(z_score))
    variance[informative] = naive[informative] + standard_error[informative] * inverse_mills
    # Analytically the line above is strictly positive; this only absorbs
    # floating-point cancellation when `naive` is many standard errors below
    # zero and the Mills term nearly cancels it.
    return np.maximum(variance, 0.0)


def _int_option(config: EstimatorConfig, key: str, default: int) -> int:
    """Read an integer tuning option, rejecting a wrong type rather than
    coercing it: a float `n_trees` silently truncating is the kind of quiet
    misconfiguration that only shows up as an unexplained accuracy change."""
    raw = config.options.get(key, default)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise TypeError(f"options[{key!r}] must be an int, got {type(raw).__name__}")
    if raw < 1:
        raise ValueError(f"options[{key!r}] must be positive, got {raw}")
    return raw


def aggregate_effect(fit: ForestFit) -> tuple[float, float]:
    """The forest's headline aggregate effect and its standard error,
    estimated from the cross-fitted residuals rather than from the forest's
    own predictions.

    This is `residual_effect` over every unit: the R-learner least-squares
    coefficient `sum(w_res * y_res) / sum(w_res**2)` with the same
    heteroskedasticity-robust sandwich standard error each subgroup cell gets.

    **Why not the mean of `fit.cate`, which is the obvious candidate.** Not
    because the point estimate is bad -- it is fine, and measured bias is
    near-identical (RICH, 20 seeds: +0.0022 for the prediction average against
    +0.0025 here) -- but because nothing honest can be said about its
    uncertainty from the forest's own output. Two formulas were measured
    against realised truth on `CATE_SCENARIOS`, 20 seeds each, n=2000,
    nominal 95%. The first two rows were measured on a prototype at
    `n_trees=200`; the third was measured through `CausalForest.fit` itself at
    the recovery harness's own `n_trees=500` config, and reproduces the
    prototype's numbers to four decimals (this estimate does not depend on the
    ensemble at all, only on the residuals):

    | ATE point and SE                          | NULL | RICH |
    |-------------------------------------------|------|------|
    | mean(cate), mean per-unit IJ variance / n | 0.40 | 0.35 |
    | mean(cate), IJ of the mean prediction     | 0.65 | 0.60 |
    | this function                             | 1.00 | 0.95 |

    The first row is the natural-looking formula and it is wrong by a factor
    of six on the standard error: averaging the per-unit IJ variances and
    dividing by `n` treats 2000 per-unit CATEs as independent draws, when
    every one of them is built from the same trees fitted to the same
    residuals. The second row keeps the correlation by applying the
    infinitesimal jackknife to the mean prediction directly, which helps and
    still undercovers -- the IJ measures the forest's sensitivity to *which
    units it resampled*, not the sampling variability of the units themselves.

    **What this estimates, precisely.** With a non-constant propensity the
    `w_res**2` weights make this an overlap-weighted average effect rather
    than the ATE proper -- units in the region where treatment is nearly
    deterministic contribute almost nothing. Under the 50/50 random assignment
    every `CATE_SCENARIOS` scenario uses, `e(X)` is constant and the two
    coincide exactly. `estimand` is reported as `"ATE"` on that basis; a
    future observational scenario with strongly varying propensity would make
    the distinction real and is the case to revisit this on.

    Per-unit heterogeneity is still the forest's job and is reported unchanged
    in the `per_unit_cate` diagnostic; it is only the aggregate number, and
    the interval around it, that comes from the residuals. That is the same
    correction `_subgroup_statistics` applies for the same reason: a model's
    predictions cannot testify about the data that trained them.
    """
    return residual_effect(fit.y_res, fit.w_res, np.ones(fit.cate.size, dtype=bool))


@register
class CausalForest:
    """Honest causal forest: per-unit CATE, an aggregate ATE, and subgroup
    discovery behind the guards in `heterogeneity_checks.py`.

    `config.options` accepts three integer keys, all with defaults tuned for
    production rather than for test speed:

    - `"n_trees"` (default 500): ensemble size.
    - `"max_depth"` (default 4): maximum honest-tree depth.
    - `"min_leaf_structure"` (default 20): minimum structure-sample units a
      leaf must retain for a split to be considered.

    Three quantities that look interchangeable are deliberately not.

    The headline `point` and `se` come from `aggregate_effect` -- the
    R-learner score on the cross-fitted residuals -- and not from the forest's
    predictions at all; that function documents the measured coverage the
    obvious alternatives lose. `per_unit_cate` is `ForestFit.cate`, which uses
    every tree, since a point estimate wants the tightest ensemble available.
    The calibration test, the best linear projection and subgroup discovery
    all read the *out-of-bag* CATE instead, because each asks a question about
    the forest's fit that evaluating the forest on its own training units
    cannot answer honestly.
    """

    name = "causal_forest"
    requirements = Requirements(
        needs_panel=False,
        needs_covariates=True,
        min_treated_units=50,
        min_control_units=50,
    )

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate:
        # Deferred rather than module-level: `heterogeneity_checks` imports
        # `ForestFit` and `oob_cate` from this module, so importing it at the
        # top would be a genuine cycle -- whichever of the two modules a caller
        # reached first would find the other half-initialised. The estimator is
        # a composition layer over both the forest primitives above and those
        # checks, and it is only the file layout that puts it here rather than
        # in a module of its own.
        from cp_causal.heterogeneity_checks import (
            best_linear_projection,
            calibration_test,
            discover_subgroups,
        )

        started = time.perf_counter()
        covariates, treatment, outcome = arrays(dataset)

        n_trees = _int_option(config, "n_trees", 500)
        max_depth = _int_option(config, "max_depth", 4)
        min_leaf_structure = _int_option(config, "min_leaf_structure", 20)

        seed = 0 if config.random_seed is None else config.random_seed
        # One generator seeded at the estimator boundary and threaded down, so
        # every random choice in this fit -- cross-fitting folds and each
        # tree's subsample alike -- traces back to `config.random_seed`.
        rng = np.random.default_rng(np.random.SeedSequence(seed))

        fit = fit_forest(
            covariates,
            treatment,
            outcome,
            n_trees=n_trees,
            min_leaf_structure=min_leaf_structure,
            max_depth=max_depth,
            subsample_frac=0.5,
            cross_fit_folds=5,
            rng=rng,
        )

        point, se = aggregate_effect(fit)
        interval = wald_interval(point, se, config.ci_level)

        covariate_names = dataset.schema.covariates
        # The forest's own cross-fitted propensity scores, carried on
        # `ForestFit`, rather than a second independently-seeded fit: overlap
        # is diagnosed for the model the estimate was actually built on.
        #
        # Weighted rather than raw standardised mean differences, matching
        # `AugmentedIPW` and `InverseProbabilityWeighting`, because
        # `overlap_checks` reports this number as balance "after weighting" --
        # handing it an unweighted SMD would put a true number under a false
        # label. This forest never reweights anything, but these are the
        # scores that produced `w_res`, so the question the check answers is
        # the one that matters here: whether the propensity model driving
        # residualization actually accounts for the covariates.
        weights = np.where(
            treatment == 1,
            1.0 / fit.propensity_scores,
            1.0 / (1.0 - fit.propensity_scores),
        )
        smd_after = standardised_mean_differences(covariates, treatment, weights)
        assumptions = (
            *overlap_checks(fit.propensity_scores, treatment, smd_after),
            calibration_test(fit),
        )

        findings = discover_subgroups(
            fit,
            covariates,
            covariate_names,
            min_cell_size=50,
            fdr_level=0.10,
            ci_level=config.ci_level,
        )
        projection = best_linear_projection(fit, covariates, covariate_names)

        return EffectEstimate(
            estimand="ATE",
            method=self.name,
            point=point,
            se=se,
            ci=interval,
            ci_basis="analytic",
            bootstrap_ci=None,
            p_value=normal_p_value(point, se),
            n_treated=int((treatment == 1).sum()),
            n_control=int((treatment == 0).sum()),
            scale=outcome_scale(dataset),
            diagnostics={
                "per_unit_cate": Diagnostic(
                    name="per_unit_cate",
                    value=fit.cate,
                    unit=None,
                    interpretation=(
                        "Per-unit CATE estimate, one per row of the input dataset, "
                        "in the same order."
                    ),
                ),
                "per_unit_cate_variance": Diagnostic(
                    name="per_unit_cate_variance",
                    value=infinitesimal_jackknife_variance(fit),
                    unit=None,
                    interpretation=(
                        "Infinitesimal-jackknife sampling variance of each unit's "
                        "CATE estimate, aligned with per_unit_cate. Per-unit only: "
                        "these do not aggregate into the variance of their own mean, "
                        "because units sharing trees and residuals are far from "
                        "independent -- see aggregate_effect for the measured cost "
                        "of assuming otherwise."
                    ),
                ),
                "best_linear_projection": Diagnostic(
                    name="best_linear_projection",
                    value=np.array(list(projection.values())),
                    unit=None,
                    interpretation=(
                        f"Out-of-bag CATE projected onto covariates "
                        f"{list(projection.keys())}, in that order. Ranks and signs "
                        f"the drivers; carries no standard errors and does not test "
                        f"them."
                    ),
                ),
            },
            assumptions=assumptions,
            counterfactual=None,
            fit_metadata=FitMetadata.capture(
                estimator=self.name,
                config_hash=config_hash(config),
                runtime_seconds=time.perf_counter() - started,
                random_seed=config.random_seed,
            ),
            subgroup_findings=findings,
        )
