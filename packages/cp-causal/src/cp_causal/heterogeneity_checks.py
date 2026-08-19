"""Guards against fabricated subgroup findings, in the parameterised-check
style `panel_checks.py`/`bayesian_checks.py` already established.

Not folded into `causal_forest.py`: these operate on an already-fitted
`ForestFit` and are a self-contained statistical layer, the same separation
`bayesian_checks.py` keeps from `bsts.py`'s model/MCMC code.
"""

from __future__ import annotations

import numpy as np
from cp_core.results import AssumptionCheck
from scipy.stats import norm

from cp_causal.estimators.causal_forest import ForestFit, oob_cate


def calibration_test(fit: ForestFit, alpha: float = 0.05) -> AssumptionCheck:
    """Omnibus calibration test (Athey-Wager / Chernozhukov et al. "generic
    ML inference" style): regress the residualized outcome on
    `W_res * mean(cate_hat)` (mean calibration) and
    `W_res * (cate_hat - mean(cate_hat))` (differential-prediction
    calibration), using OUT-OF-BAG CATE, never the in-sample ensemble
    average.

    Measured during plan-writing: evaluated in-sample, this test fires
    (t~7.6) on both a genuinely heterogeneous scenario and a null scenario
    with zero heterogeneity -- an overfitting artifact from testing a
    forest's predictions against the same data used to fit it, the same
    class of bias honest splitting and cross-fitting exist to prevent
    elsewhere in this estimator. Out-of-bag CATE is what this function uses
    instead.

    **It does not currently separate the two scenarios, and must not be
    relied on to gate subgroup reporting until it does.** Measured over 10
    data seeds at `n_trees=500` (full table and protocol in
    `tests/causal/test_heterogeneity_checks.py`): the differential
    coefficient, which should be ~1 under real heterogeneity and ~0 under
    none, has mean -1.15 on `CATE_SCENARIOS["RICH"]` and -1.90 on
    `CATE_SCENARIOS["NULL"]`, with |t| > 1.96 on 5/10 NULL seeds (a ~50%
    false-positive rate at alpha=0.05) and on 3/10 RICH seeds, all three of
    those in the wrong (negative) direction. Raising `n_trees` to 1000 or
    2000 moves the statistic by less than 0.1 at a fixed seed, so the
    ensemble's Monte-Carlo noise is not the binding constraint. What is
    *not* at fault: the mean-calibration coefficient is 1.00 +/- 0.03 on
    both scenarios, HC0 standard errors move every t by less than 0.07, and
    the forest genuinely carries signal (median corr(`oob_cate`,
    `true_cate`) ~ +0.4 on RICH). The bias is in the differential
    coefficient itself -- the out-of-bag prediction is deducted from a mean
    that the unit's own outcome still influences through the other units'
    leaves, so the demeaned regressor stays coupled to the regressand's own
    noise.

    Intended contract, once the statistic is fixed: if the
    differential-prediction coefficient's confidence interval excludes
    zero, there is real, detectable heterogeneity; if not, this estimator
    reports "no reliable evidence of heterogeneity" and skips subgroup
    discovery entirely (parent spec §5.7's explicit requirement).
    """
    cate_oob = oob_cate(fit)
    valid = np.isfinite(cate_oob)
    cate_oob = cate_oob[valid]
    y_res = fit.y_res[valid]
    w_res = fit.w_res[valid]

    mean_prediction = float(cate_oob.mean())
    demeaned = cate_oob - mean_prediction
    design = np.column_stack([w_res * mean_prediction, w_res * demeaned])
    coefficients, *_ = np.linalg.lstsq(design, y_res, rcond=None)
    residuals = y_res - design @ coefficients
    n, p = design.shape
    sigma_squared = float(np.sum(residuals**2) / (n - p))
    covariance = sigma_squared * np.linalg.inv(design.T @ design)
    differential_se = float(np.sqrt(covariance[1, 1]))
    differential_coefficient = float(coefficients[1])
    t_statistic = differential_coefficient / differential_se if differential_se > 0 else 0.0

    p_value = float(2 * norm.sf(abs(t_statistic)))
    detected = p_value < alpha

    return AssumptionCheck(
        name="calibration_test",
        status="PASS" if detected else "FAIL",
        statistic=t_statistic,
        threshold=float(norm.ppf(1 - alpha / 2)),
        interpretation=(
            f"Differential-prediction calibration coefficient {differential_coefficient:.4f} "
            f"(t={t_statistic:.2f}, p={p_value:.4f}), evaluated on out-of-bag CATE. "
            + (
                "The forest's heterogeneity predictions are detectably informative."
                if detected
                else "No reliable evidence of heterogeneity -- subgroup reporting is skipped."
            )
        ),
        remedy=(
            None
            if detected
            else "Report the aggregate ATE/ATT only; do not report per-unit CATE or "
            "subgroups as if heterogeneity were established."
        ),
    )


def best_linear_projection(
    fit: ForestFit, covariates: np.ndarray, covariate_names: tuple[str, ...]
) -> dict[str, float]:
    """Best linear projection of out-of-bag CATE onto the covariates: an
    interpretable linear summary of which covariates the forest's
    heterogeneity estimate actually moves with, not a claim that the true
    CATE function is linear.
    """
    cate_oob = oob_cate(fit)
    valid = np.isfinite(cate_oob)
    design = np.column_stack([np.ones(int(valid.sum())), covariates[valid]])
    coefficients, *_ = np.linalg.lstsq(design, cate_oob[valid], rcond=None)
    return {
        name: float(coefficient)
        for name, coefficient in zip(covariate_names, coefficients[1:], strict=True)
    }
