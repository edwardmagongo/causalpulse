"""Sensitivity analysis: how hard would it be to overturn this finding?

Three independent lines of attack, and they are independent on purpose. A
confounding bound says nothing about whether the result survives a different
covariate set; a specification curve says nothing about whether a placebo
period looks the same. Reporting all three under one type is what stops a
reader taking one clean answer as the whole picture.
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import combinations

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import (
    CinelliHazlettResult,
    CounterfactualPoint,
    EffectEstimate,
    EValueResult,
    SpecificationCurve,
    SpecificationPoint,
)
from cp_core.values import EstimatorConfig
from cp_statistics.covariance import OLSFit, ols_fit
from scipy import stats

from cp_causal._panel import panel_arrays
from cp_causal._support import arrays
from cp_causal.estimators.did import twfe_design, two_by_two_design
from cp_causal.estimators.regression import lin_design
from cp_causal.propensity import POSITIVITY_THRESHOLD, resolve_trim
from cp_causal.registry import all_estimators, get_estimator

# Chinn (2000)'s conversion from a standardised mean difference to an
# approximate log odds ratio is `log(OR) ~= 1.81 * d`; VanderWeele's E-value
# guidance halves the exponent to approximate a risk ratio, giving
# `RR ~= exp(0.91 * d)`. The factor is not decoration -- `exp(d)` is a
# different and larger quantity, and using it would overstate every E-value
# computed on this path.
_STANDARDISED_TO_LOG_RR = 0.91

_CONVERSION_NOTE = (
    "The reported effect is on an absolute scale, so the risk ratio was "
    f"approximated as exp({_STANDARDISED_TO_LOG_RR} * effect / outcome_sd). "
    "That conversion assumes an approximately normal, non-rare outcome -- the "
    "opposite assumption from the rare-outcome derivation the classical "
    "E-value is stated under. Read this number as indicative, not as the "
    "exact ratio-scale quantity."
)

_ARM_MEANS_NOTE = (
    "The outcome is binary, so the risk ratio is the ratio of the reported "
    "arm means (treated mean over the predicted untreated mean) rather than a "
    "conversion from the effect size. No standardised-effect approximation "
    "was applied."
)


def _e_value_from_ratio(ratio: float) -> float:
    """VanderWeele-Ding: the minimum joint confounder-association strength.

    Symmetric about the null, so a protective ratio is inverted first: a risk
    ratio of 0.5 is exactly as hard to explain away as one of 2.
    """
    if ratio <= 0.0 or not math.isfinite(ratio):
        raise ValueError(f"risk ratio must be finite and positive, got {ratio}")
    if ratio < 1.0:
        ratio = 1.0 / ratio
    return ratio + math.sqrt(ratio * (ratio - 1.0))


def _bound_nearer_null(lower: float, upper: float, null: float) -> float | None:
    """The interval bound closer to the null, or None if the interval spans it."""
    if lower <= null <= upper:
        return None
    return lower if lower > null else upper


def e_value(
    estimate: EffectEstimate, *, outcome_sd: float | None = None
) -> EValueResult:
    """How strong an unmeasured confounder would have to be, on the risk-ratio
    scale, to move this estimate to the null.

    Not a probability that confounding is present, and not a test for it. It
    is a statement of the form "an unmeasured confounder would have to be
    associated with both treatment and outcome by a risk ratio of at least E,
    above and beyond the measured covariates, to explain this away".
    """
    counterfactual = estimate.counterfactual
    if estimate.scale == "log_ratio":
        ratio = math.exp(estimate.point)
        bound = _bound_nearer_null(estimate.ci.lower, estimate.ci.upper, 0.0)
        ratio_bound = None if bound is None else math.exp(bound)
        note, approximated = None, False
    elif (
        estimate.scale == "percentage_point"
        and isinstance(counterfactual, CounterfactualPoint)
        and counterfactual.predicted_absent_treatment > 0.0
    ):
        baseline = counterfactual.predicted_absent_treatment
        ratio = counterfactual.observed_mean / baseline
        bound = _bound_nearer_null(estimate.ci.lower, estimate.ci.upper, 0.0)
        ratio_bound = None if bound is None else (baseline + bound) / baseline
        note, approximated = _ARM_MEANS_NOTE, False
    else:
        if outcome_sd is None or outcome_sd <= 0.0:
            raise ValueError(
                f"scale={estimate.scale!r} has no ratio available from the "
                "estimate itself, so a positive outcome_sd is required to "
                f"convert the effect size; got outcome_sd={outcome_sd!r}"
            )
        ratio = math.exp(_STANDARDISED_TO_LOG_RR * estimate.point / outcome_sd)
        bound = _bound_nearer_null(estimate.ci.lower, estimate.ci.upper, 0.0)
        ratio_bound = (
            None
            if bound is None
            else math.exp(_STANDARDISED_TO_LOG_RR * bound / outcome_sd)
        )
        note, approximated = _CONVERSION_NOTE, True

    return EValueResult(
        e_value=_e_value_from_ratio(ratio),
        # An interval already covering the null needs no confounding to
        # explain it away, which is E = 1, not an undefined quantity.
        e_value_ci_bound=1.0 if ratio_bound is None else _e_value_from_ratio(ratio_bound),
        approximated=approximated,
        approximation_note=note,
    )


# The OLS-based estimators whose reported point is a single coefficient of a
# single design matrix. `event_study` is deliberately absent: its point is the
# mean of several horizon coefficients, so the partial-R2 quantity this method
# is interpreted through has no one regressor to attach to. Bounding a linear
# combination's t-statistic would still compute, but the number would not mean
# what the method says it means.
OLS_SENSITIVITY_ESTIMATORS: frozenset[str] = frozenset(
    {"lin_regression", "difference_in_means", "did_2x2", "twfe"}
)


def _ols_treatment_coefficient(
    dataset: CausalDataset, estimator_name: str
) -> tuple[OLSFit, int]:
    """Refit the design the named estimator fitted, and say which column is treatment.

    Every branch calls the estimator's own design function rather than
    rebuilding the matrix, so a change to the estimator's specification
    reaches this bound automatically. `difference_in_means` is the exception
    only because it has no design function: it computes a Welch t-test, and
    OLS on `[1, treatment]` is the same estimand written as a regression.
    `test_the_reconstructed_design_reproduces_the_estimator_s_own_point`
    pins all four against what the estimator actually reports.
    """
    if estimator_name == "lin_regression":
        covariates, treatment, outcome = arrays(dataset)
        return ols_fit(lin_design(covariates, treatment), outcome), 1
    if estimator_name == "difference_in_means":
        _, treatment, outcome = arrays(dataset)
        design = np.column_stack([np.ones(treatment.size), treatment])
        return ols_fit(design, outcome), 1
    if estimator_name == "twfe":
        panel = panel_arrays(dataset)
        return ols_fit(twfe_design(panel), panel.outcome), 1
    if estimator_name == "did_2x2":
        panel = panel_arrays(dataset)
        adoption = panel.adoption_periods
        if len(adoption) != 1:
            raise ValueError(
                f"did_2x2 requires a single adoption period; this panel has "
                f"{len(adoption)}: {list(adoption)}"
            )
        # Column 3 is the ever-treated x post interaction, which is the DiD
        # estimate; see `two_by_two_design`.
        return ols_fit(two_by_two_design(panel, adoption[0]), panel.outcome), 3
    raise KeyError(f"no OLS design is registered for estimator {estimator_name!r}")


def _bias_share(f_value: float) -> float:
    """`0.5 * (sqrt(f**4 + 4 * f**2) - f**2)`, in the form that survives large f.

    Written literally, that difference loses every significant digit once
    `f**4` dominates `4 * f**2`: measured, it returns exactly 1.0 by f = 1e5
    and collapses to 0.0 by f = 1e11, inverting the quantity rather than
    merely blurring it. Multiplying through by the conjugate cancels the
    `f**4` terms exactly and leaves `2 / (sqrt(1 + 4 / f**2) + 1)`, which
    agrees bit-for-bit with the literal form through f = 100 and stays
    correct beyond it. Approaches, but never reaches, 1.
    """
    if f_value <= 0.0:
        return 0.0
    return 2.0 / (math.sqrt(1.0 + 4.0 / f_value**2) + 1.0)


def _robustness_value(
    t_statistic: float, df: int, *, q: float = 1.0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """`(RV_q, RV_q_alpha, partial_r2)` from a t-statistic and residual df.

    Depends on nothing but those two numbers, which is why it is a free
    function: it is reproducible by hand from any published regression table,
    and the worked-example test does exactly that.
    """
    t_squared = t_statistic**2
    partial_r2 = t_squared / (t_squared + df)
    root_df = math.sqrt(df)

    f_q = q * abs(t_statistic) / root_df
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    # An estimate already inside the critical region needs no confounding to
    # make it insignificant, so the bar is zero rather than negative.
    f_alpha = max(f_q - critical / root_df, 0.0)
    return _bias_share(f_q), _bias_share(f_alpha), partial_r2


def cinelli_hazlett(
    dataset: CausalDataset, estimator_name: str, config: EstimatorConfig
) -> CinelliHazlettResult | None:
    """How much residual variation a confounder would have to explain, in both
    treatment and outcome, to overturn an OLS estimate.

    `None` for any estimator not fitted by least squares with a single
    treatment coefficient; the caller pairs that with
    `cinelli_hazlett_reason`. Cheap by design -- one `ols_fit` -- so it does
    not carry the runtime cost the refit-based techniques in this module do.
    `config` is accepted because the caller passes one uniformly across every
    technique here; this bound reads nothing from it, because the design it
    refits is fixed by the estimator rather than by the fit configuration.

    The t-statistic is the ordinary-least-squares one. `twfe` and `did_2x2`
    report cluster-robust standard errors, but these bounds are derived under
    homoskedastic OLS, so a clustered t-statistic would produce a number that
    matches no quantity in the derivation. Read the bound as ignoring
    clustering.
    """
    # Raises KeyError on an unknown name rather than returning None: a typo
    # must not read as "this technique does not apply to that estimator".
    get_estimator(estimator_name)
    if estimator_name not in OLS_SENSITIVITY_ESTIMATORS:
        return None

    fit, column = _ols_treatment_coefficient(dataset, estimator_name)
    df = fit.n - fit.k
    if df < 1:
        return None
    residual_variance = float(fit.residuals @ fit.residuals) / df
    standard_error = math.sqrt(residual_variance * float(fit.xtx_inv[column, column]))
    if standard_error <= 0.0:
        return None

    rv, rv_alpha, partial_r2 = _robustness_value(
        float(fit.beta[column]) / standard_error, df
    )
    return CinelliHazlettResult(
        robustness_value=rv,
        robustness_value_alpha=rv_alpha,
        partial_r2_treatment_outcome=partial_r2,
        degrees_of_freedom=df,
    )


# The cross-sectional estimators whose answer actually moves with an analyst's
# discretionary choices. `difference_in_means` is excluded because it never
# reads covariates -- measured during planning, it returns a bit-identical
# point for every subset, so its curve would be N copies of one number, which
# reads as robustness rather than as an inapplicable analysis. `causal_forest`
# is excluded on cost: 4.56s per fit against lin_regression's 0.02s.
CURVE_ESTIMATORS: frozenset[str] = frozenset({"lin_regression", "ipw", "aipw"})

# Running upward from POSITIVITY_THRESHOLD rather than straddling it. The
# obvious choice, (0.005, 0.01, 0.02), was measured and rejected: it clips
# zero rows at all three values on scenarios A and D, so it added two exact
# duplicates per specification rather than two alternative analyses, and a
# curve of 18 points that is really 6 reads as robustness it has not earned.
# These values clip 0/3/18 of scenario D's 600 rows and 2/16/51 of scenario
# G's. They still clip nothing on randomised scenario A, whose propensity
# scores span 0.40 to 0.54 -- that is the correct answer there, not a gap:
# under near-perfect overlap there is genuinely nothing for a trim to remove.
TRIM_AXIS: tuple[float, ...] = (POSITIVITY_THRESHOLD, 0.05, 0.10)

_PROPENSITY_CURVE_ESTIMATORS: frozenset[str] = frozenset({"ipw", "aipw"})

_UNSUPPORTED_CURVE_REASON = (
    "specification_curve is built for {supported} this phase; {name} is not "
    "one of them. This is a scope limit, not a statement that the analysis "
    "would be meaningless -- see the phase 9 design note."
)


def _high_leverage_rows(dataset: CausalDataset) -> np.ndarray:
    """Boolean mask of rows the outlier axis excludes.

    Computed once, from the FULL declared covariate set, so the same rows are
    dropped at every point on the curve. A per-subset rule would move two
    things at once and the axis would no longer isolate the outlier choice.

    `leverage > 2k/n` is the conventional rule of thumb; `OLSFit.leverage` is
    already computed by `ols_fit`, so this costs one extra fit per curve.
    """
    covariates, treatment, outcome = arrays(dataset)
    design = np.column_stack([np.ones(treatment.size), treatment, covariates])
    fit = ols_fit(design, outcome)
    return np.asarray(fit.leverage > 2.0 * fit.k / fit.n)


def _subset_dataset(
    dataset: CausalDataset, covariates: tuple[str, ...], drop: np.ndarray | None
) -> CausalDataset:
    frame = (
        dataset.frame
        if drop is None
        else dataset.frame.loc[~drop].reset_index(drop=True)
    )
    return CausalDataset(
        frame=frame, schema=replace(dataset.schema, covariates=covariates)
    )


def specification_curve(
    dataset: CausalDataset,
    estimator_name: str,
    base_config: EstimatorConfig,
    *,
    max_specifications: int = 256,
) -> SpecificationCurve:
    """Refit the same question under every combination of the discretionary
    choices an analyst could defensibly have made differently.

    Every point is a genuine refit through the registry, not an approximation.
    `original_index` identifies the analysis as actually run, so a reader can
    see where the reported result sits inside its own curve rather than
    treating every point as equally the answer.

    Raises `ValueError` for an estimator outside `CURVE_ESTIMATORS` -- an
    explicit refusal rather than an empty curve, because "not built for this
    estimator yet" is a fact a caller should catch immediately.
    """
    if max_specifications < 1:
        raise ValueError(
            f"max_specifications must be positive, got {max_specifications}"
        )
    estimator_class = all_estimators().get(estimator_name)
    if estimator_class is None:
        raise KeyError(
            f"unknown estimator {estimator_name!r}; available: {sorted(all_estimators())}"
        )
    if estimator_name not in CURVE_ESTIMATORS:
        raise ValueError(
            _UNSUPPORTED_CURVE_REASON.format(
                supported=sorted(CURVE_ESTIMATORS), name=estimator_name
            )
        )

    declared = dataset.schema.covariates
    subsets = [
        tuple(subset)
        for size in range(len(declared) + 1)
        for subset in combinations(declared, size)
    ]
    base_trim = resolve_trim(base_config.options)
    # The caller's own trim is always on the axis. Without this a caller who
    # passed options={"trim": 0.03} would have a base specification that is not
    # among its own candidates, and the curve could not locate the analysis as
    # actually run.
    trims = (
        tuple(dict.fromkeys((*TRIM_AXIS, base_trim)))
        if estimator_name in _PROPENSITY_CURVE_ESTIMATORS
        else (base_trim,)
    )
    high_leverage = _high_leverage_rows(dataset)
    base = (declared, False, base_trim)

    candidates = [
        (subset, excluded, trim)
        for subset in subsets
        for excluded in (False, True)
        for trim in trims
    ]
    # Only keep specifications the estimator itself would accept. The empty
    # covariate subset is the case this exists for: `fit_propensity` raises on
    # it, and `Requirements.unmet` already knows that, so the rule lives in one
    # place rather than being restated here.
    requirements = estimator_class.requirements
    candidates = [
        candidate
        for candidate in candidates
        if not requirements.unmet(
            _subset_dataset(
                dataset, candidate[0], high_leverage if candidate[1] else None
            )
        )
    ]
    if base not in candidates:
        raise ValueError(
            f"{estimator_name} cannot be fitted on the dataset as declared, so "
            "it has no specification curve to sit inside"
        )

    capped = len(candidates) > max_specifications
    if capped:
        kept = candidates[:max_specifications]
        # The analysis as actually run must survive truncation: a curve that
        # dropped it would fail SpecificationCurve's own invariant, and would
        # be unreadable even if it did not.
        if base not in kept:
            kept[-1] = base
        candidates = kept

    points: list[SpecificationPoint] = []
    for covariates, excluded, trim in candidates:
        config = replace(
            base_config, options={**dict(base_config.options), "trim": trim}
        )
        subset = _subset_dataset(
            dataset, covariates, high_leverage if excluded else None
        )
        estimate = estimator_class().fit(subset, config)
        points.append(
            SpecificationPoint(
                specification={
                    "covariates": covariates,
                    "outliers_excluded": excluded,
                    "trim": trim,
                },
                point=estimate.point,
                ci=estimate.ci,
            )
        )

    return SpecificationCurve(
        specifications=tuple(points),
        original_index=candidates.index(base),
        capped=capped,
    )
