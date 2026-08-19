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

# The out-of-bag prediction is built from the same `y_res` the calibration
# regression puts on the left-hand side, so its numerator is a *quadratic* form
# in `y_res`, `num = y_res' B y_res`, rather than the linear one ordinary least
# squares assumes. That single fact sets both constants below. See
# `calibration_terms` for the derivation and the measurements behind them.
#
# `E[num] = sigma^2 * trace(B)`, and `trace(B) = -1` exactly: out-of-bag zeroes
# the diagonal, while each row of the forest's leaf-averaging kernel sums to one
# in the `w_res`-inner product. So the numerator carries a `-sigma^2` bias that
# has to be added back.
#
# The identity is exact only while no leaf degenerates. `leaf_tau` returns 0.0
# for a leaf with fewer than two units or no treatment variation, and
# `predict_leaf_tau` returns 0.0 for an empty leaf; each such (tree, unit) pair
# contributes row mass 0 rather than 1, pulling `trace(B)` above -1 and making
# this constant an OVER-correction, which biases the statistic positive -- the
# anti-conservative direction for a gate. Measured at the shipped
# configuration: 0 degenerate leaf-rows out of 5108. A deeper `max_depth` or a
# smaller `min_leaf_structure` could reach it, so
# `test_calibration_kernel_row_mass_is_exactly_one` asserts the row mass
# directly rather than trusting that it stays true.
_KERNEL_ROW_MASS = 1.0
# `Var[num] = sigma^4 * (trace(B @ B) + trace(B @ B.T))` while OLS's
# `sigma^2 * denominator` estimates only the `trace(B @ B.T)` half.
#
# This is a BOUND, not an equality. `trace(A @ A) <= trace(A @ A.T)` for every
# real matrix -- their difference is `0.5 * ||A - A.T||_F**2 >= 0` -- so the
# true inflation is at most 2, with equality exactly when `B` is symmetric.
# Measured: the closed-form fixture in the tests is exactly symmetric and hits
# 2.000, but the real forest's kernel is not (relative asymmetry 0.44), and its
# inflation is 1.90. Using 2.0 therefore cannot be anti-conservative, which is
# the property that matters for a check that gates subgroup reporting.
_VARIANCE_INFLATION = 2.0


def calibration_terms(fit: ForestFit) -> tuple[float, float, float]:
    """The three quantities `calibration_test` combines, returned before any
    bias correction: `(numerator, denominator, residual_variance)`.

    Split out for the same reason `infinitesimal_jackknife_terms` is: the two
    corrections `calibration_test` applies are a fixed additive shift and a
    fixed variance multiplier, and a test written against the finished
    t-statistic on real scenario data cannot pin either of them down (the
    scenarios are far too noisy per seed -- see
    `tests/causal/test_heterogeneity_checks.py`). A test written against these
    terms can, and does.

    Setup. Write `u = w_res`, `W = sum(u**2)`, and let `lambda` be the
    `u**2`-weighted mean of `cate_oob`. Then

        differential = u * (cate_oob - lambda)
        numerator    = differential @ y_res
        denominator  = differential @ differential

    and `numerator / denominator` is *exactly* the differential-prediction
    coefficient of the two-column Athey-Wager / GRF `test_calibration`
    regression of `y_res` on `[u * mean(cate_oob), u * (cate_oob -
    mean(cate_oob))]`. The two columns of that regression span
    `{u, u * cate_oob}` whatever scalar is used to demean, so the coefficient
    does not depend on `mean(cate_oob)` at all -- measured identical to twelve
    decimal places with the mean replaced by 1.0, -3.7 and 100.0. Using the
    `u**2`-weighted mean here just makes that partialling-out explicit, and
    makes the design orthogonal.

    Why the terms need correcting. `cate_oob` is not an exogenous regressor:
    it is a linear functional of the same `y_res` on the left-hand side,
    `cate_oob = C @ y_res`, with `C[j, i]` the weight leaf-averaging gives
    training unit `i` in the prediction at unit `j`. So

        numerator = y_res' B y_res,   B = diag(u) (I - 1 (u**2)' / W) C

    a quadratic form, and its first two moments are traces of `B`:

    1. **`E[numerator] = sigma**2 * trace(B) = -sigma**2`.** Out-of-bag
       averaging makes `C[i, i] = 0` -- unit `i`'s own outcome never enters
       its own prediction -- but it does nothing about `C[j, i]` for `j != i`,
       and those survive into the level `lambda` that the regression's `u`
       column subtracts. Each row of `C` puts total mass exactly one on the
       training units, in the sense `sum_i u[i] C[j, i] = 1` (leaf tau is
       `sum(u*y) / sum(u**2)` over the leaf), so the surviving off-diagonal
       mass is exactly one and the bias is exactly `-sigma**2`. This is a
       deterministic identity, not an approximation: reconstructing `C`
       explicitly gives `max|diag(C)| = 0`, row mass `1.000000000000` with a
       spread of 4e-17, and `trace(B) = -1.000000000000` -- on the tests'
       closed-form fixture and on a real 500-tree forest alike. It holds only
       while no leaf degenerates; see `_KERNEL_ROW_MASS`.

       A related exactness worth knowing when reading the null behaviour:
       `B @ u = 0` to 1e-16, so a constant true tau is annihilated outright.

    2. **`Var[numerator] = sigma**4 (trace(B @ B) + trace(B @ B.T))`**, of
       which OLS's `sigma**2 * denominator` estimates only the
       `trace(B @ B.T)` half (`E[denominator] = sigma**2 trace(B @ B.T)`).
       The two traces coincide when `B` is symmetric, giving a standard error
       `sqrt(2)` times the OLS one. The forest's kernel is NOT symmetric in
       general -- measured relative asymmetry `||B - B.T||_F / ||B||_F = 0.44`
       on a real forest, where the trace ratio is 0.903 rather than 1, so the
       true inflation there is 1.90. `sqrt(2)` is used anyway because
       `trace(A @ A) <= trace(A @ A.T)` always, making 2.0 a provable upper
       bound and the correction conservative by construction. On the tests'
       fixture, where `u**2` is constant and `B` comes out exactly symmetric,
       the ratio is 1.000000 and the bound is attained.

    Two assumptions this leaves implicit. Everything is conditional on the
    tree structure -- for a real forest `C` depends on `y_res` through the
    splits, and honest splitting is what makes that dependence weak enough to
    ignore; the 40-seed null sweep in the tests is the evidence that it is.
    And `E[numerator] = sum_i sigma_i**2 B[i, i]` under heteroskedasticity,
    while the code subtracts a single pooled `sigma**2`; on a binary outcome
    the two differ, and the residual shows up as the null statistic's mean of
    -0.30 rather than 0. Both leftovers point the conservative way.

    Both effects are invisible at the sample sizes forests are usually
    reported at, because they are `O(sigma**2 / denominator)` and
    `denominator` is normally large. This forest's out-of-bag predictions have
    a cross-sectional standard deviation around 0.03, which makes
    `denominator` ~0.4 and `sigma**2 / denominator` ~0.6 -- the same order as
    the coefficient being estimated.

    `residual_variance` is the plain OLS residual variance of that two-column
    fit, unchanged from the uncorrected implementation (the reparametrisation
    above leaves the column space, and so the residuals, identical).
    """
    cate_oob = oob_cate(fit)
    valid = np.isfinite(cate_oob)
    cate_oob = cate_oob[valid]
    y_res = fit.y_res[valid]
    w_res = fit.w_res[valid]

    if cate_oob.size < 3:
        # Every unit was in-bag in every tree, so nothing has an out-of-bag
        # prediction at all. Reached by `subsample_frac=1.0` or a single tree,
        # not by anything the estimator's own defaults produce -- and a
        # distinct cause from the zero-treatment-variation case below.
        raise ValueError(
            f"calibration test needs out-of-bag predictions for at least 3 units, "
            f"got {cate_oob.size}. Every unit was in-bag in every tree; check that "
            f"subsample_frac is below 1.0 and that the forest has several trees."
        )

    weight = w_res**2
    total_weight = float(weight.sum())
    if total_weight <= 0.0:
        raise ValueError(
            "calibration test needs a residualized treatment with nonzero variation: "
            "sum(w_res**2) is zero, so no treatment-effect coefficient is identified"
        )

    level = float(np.sum(weight * cate_oob) / total_weight)
    differential = w_res * (cate_oob - level)
    numerator = float(differential @ y_res)
    denominator = float(differential @ differential)

    design = np.column_stack([w_res, differential])
    coefficients, *_ = np.linalg.lstsq(design, y_res, rcond=None)
    residuals = y_res - design @ coefficients
    residual_variance = float(np.sum(residuals**2) / (y_res.size - design.shape[1]))
    return numerator, denominator, residual_variance


def calibration_test(fit: ForestFit, alpha: float = 0.05) -> AssumptionCheck:
    """Omnibus calibration test (Athey-Wager / Chernozhukov et al. "generic
    ML inference" style): regress the residualized outcome on the forest's
    OUT-OF-BAG CATE interacted with `w_res`, and ask whether the
    differential-prediction coefficient is positive.

    If it is, the forest's per-unit ordering carries real information about
    the true effect and subgroup reporting is licensed; if it is not, this
    estimator reports "no reliable evidence of heterogeneity" and skips
    subgroup discovery entirely (parent spec §5.7's explicit requirement).

    **Status semantics.** `PASS` means heterogeneity was detected, `WARN`
    means it was not. `WARN` rather than `FAIL` is deliberate and differs
    from the sibling checks in `panel_checks.py`/`bayesian_checks.py`, where
    `FAIL` means an identification assumption is violated. Finding no
    heterogeneity is not a violation: the analysis is sound, the effect is
    simply homogeneous, and the correct response -- report the aggregate
    ATE/ATT and no subgroups -- happens automatically. Since
    `EffectEstimate.worst_assumption_status` rolls up the maximum severity
    across all checks, returning `FAIL` here would mark every correct
    null-heterogeneity analysis as broken. `WARN` keeps the caveat visible
    without making that claim.

    Three things this implementation does that the textbook form does not,
    each of which was measured to matter here:

    1. **Out-of-bag CATE, never the in-sample ensemble average.** Measured
       during plan-writing: in-sample, this test fires (t~7.6) on a null
       scenario with zero heterogeneity. Reproduced exactly on a synthetic
       forest with a known kernel -- in-sample the coefficient sits at +1.01
       with t=3.85 and rejects on 100% of replications with tau held
       constant, because in-sample `C[i, i] > 0` puts each unit's own noise
       into its own prediction.

    2. **The `-sigma**2` out-of-bag bias is added back**, and

    3. **the standard error is inflated by `sqrt(2)`.**

       `calibration_terms` derives both: `cate_oob` is a function of the same
       `y_res` being regressed, so the numerator is a quadratic form whose
       mean is `-sigma**2` and whose variance is twice what OLS reports.
       Out-of-bag fixes the diagonal of that form and nothing else. Without
       the two corrections the statistic is biased *negative* -- which is how
       this was found: it rejected on 30% of null-scenario seeds, always in
       the wrong direction.

    The test is one-sided. Only a *positive* differential coefficient is
    evidence that the forest's heterogeneity predictions are informative; a
    large negative one says the predicted ordering is inverted, which must
    not open the subgroup gate. The two-sided rule the first draft used did
    exactly that: every one of its null-scenario rejections, and every one of
    its rich-scenario rejections, was negative.

    **Power, measured, not assumed.** Over 40 data seeds at `n_trees=500`
    and `CATE_N_UNITS=2000`, the shipped statistic separates the two
    scenarios cleanly: `CATE_SCENARIOS["RICH"]` gives t mean +3.22 (sd 0.70,
    min +1.75) and detects on 40/40 seeds, while `CATE_SCENARIOS["NULL"]`
    gives t mean -0.30 (sd 0.97) and detects on 0/40. Full tables in
    `tests/causal/test_heterogeneity_checks.py`.

    That separation depends on the scenario carrying heterogeneity above the
    detection floor, which is a property of the fixture rather than of this
    code. An earlier draft of RICH had an effect spread of sd ~0.038 against
    a binary outcome's sigma ~0.50, which is below the floor for *any*
    differential-calibration test at this sample size -- an oracle holding
    the true per-unit tau detected it on only 50% of seeds, and this test on
    0/12. `cate_scenarios._RICH_EFFECT` was raised to sd ~0.123 on that
    measurement; see its comment. If this check ever reports no
    heterogeneity on data believed to be heterogeneous, compare the effect
    spread against the outcome's noise before suspecting the statistic.
    """
    numerator, denominator, residual_variance = calibration_terms(fit)

    if denominator <= 0.0 or residual_variance <= 0.0:
        # A forest whose out-of-bag prediction is constant across units carries
        # no heterogeneity signal at all; there is nothing to test, and the
        # honest answer is the same one a non-rejection gives.
        return _no_evidence(
            alpha,
            "The forest's out-of-bag CATE is constant across units, so no "
            "differential-prediction coefficient is identified.",
        )

    coefficient = (numerator + _KERNEL_ROW_MASS * residual_variance) / denominator
    standard_error = float(
        np.sqrt(_VARIANCE_INFLATION * residual_variance / denominator)
    )
    t_statistic = coefficient / standard_error

    p_value = float(norm.sf(t_statistic))
    detected = p_value < alpha

    return AssumptionCheck(
        name="calibration_test",
        status="PASS" if detected else "WARN",
        statistic=t_statistic,
        threshold=float(norm.ppf(1 - alpha)),
        interpretation=(
            f"Differential-prediction calibration coefficient {coefficient:.4f} "
            f"(t={t_statistic:.2f}, one-sided p={p_value:.4f}), evaluated on "
            f"out-of-bag CATE and corrected for the out-of-bag numerator's "
            f"-sigma^2 bias and doubled variance. "
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


def _no_evidence(alpha: float, reason: str) -> AssumptionCheck:
    # `statistic=None`, not 0.0: no statistic was computed, and the contract
    # allows saying so. Reporting 0.0 would render as a real-looking t of 0.00
    # for a quantity that was never estimated.
    return AssumptionCheck(
        name="calibration_test",
        status="WARN",
        statistic=None,
        threshold=float(norm.ppf(1 - alpha)),
        interpretation=(
            f"{reason} No reliable evidence of heterogeneity -- subgroup "
            f"reporting is skipped."
        ),
        remedy=(
            "Report the aggregate ATE/ATT only; do not report per-unit CATE or "
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

    Two honest limits on how far these numbers can be read.

    This is an ordinary least-squares fit of the forest's *estimated*
    out-of-bag CATE, so the coefficients are attenuated toward zero by that
    estimate's own noise and carry no standard errors -- they rank and sign
    the drivers, they do not test them. The name is a term of art
    (Semenova-Chernozhukov; grf's `best_linear_projection`) for a
    doubly-robust projection of the *true* CATE that does come with valid
    inference; this is the descriptive cousin, not that estimator.

    And the design is not checked for rank. `lstsq` returns the minimum-norm
    solution when covariates are collinear, which silently splits a
    coefficient between them rather than failing, so a driver can look weak
    because it shares its variance with a neighbour.
    """
    if covariates.ndim != 2:
        raise ValueError(f"covariates must be 2-dimensional, got shape {covariates.shape}")
    if covariates.shape[0] != fit.cate.size:
        raise ValueError(
            f"covariates has {covariates.shape[0]} rows but the forest was fitted on "
            f"{fit.cate.size} units"
        )
    if covariates.shape[1] != len(covariate_names):
        raise ValueError(
            f"got {len(covariate_names)} covariate names for {covariates.shape[1]} "
            f"covariate columns"
        )

    cate_oob = oob_cate(fit)
    valid = np.isfinite(cate_oob)
    design = np.column_stack([np.ones(int(valid.sum())), covariates[valid]])
    coefficients, *_ = np.linalg.lstsq(design, cate_oob[valid], rcond=None)
    return {
        name: float(coefficient)
        for name, coefficient in zip(covariate_names, coefficients[1:], strict=True)
    }
