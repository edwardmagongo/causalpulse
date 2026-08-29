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
    ConfoundingSensitivity,
    CounterfactualPoint,
    EffectEstimate,
    EValueResult,
    FalsificationResult,
    PlaceboTestResult,
    SensitivityReport,
    SpecificationCurve,
    SpecificationPoint,
)
from cp_core.values import EstimatorConfig
from cp_statistics.covariance import OLSFit, ols_fit
from scipy import stats

from cp_causal._panel import MIN_UNITS_PER_ARM, donor_matrix, panel_arrays
from cp_causal._support import arrays
from cp_causal.estimators.did import twfe_design, two_by_two_design
from cp_causal.estimators.regression import lin_design
from cp_causal.estimators.synthetic_control import (
    fit_donor_pool_point,
    leave_one_donor_out_points,
    permutation_p_value,
    placebo_runs,
)
from cp_causal.panel_checks import placebo_leads
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

_COUNTERFACTUAL_RATIO_NOTE = (
    "The outcome is binary, so the risk ratio is available on the data's own "
    "scale rather than through a conversion: the predicted untreated mean "
    "plus the reported effect, over the predicted untreated mean. It is built "
    "from the reported effect rather than from the raw treated-arm mean so "
    "that it describes the estimate this report is about -- for an adjusted "
    "or reweighted estimator the two differ. No standardised-effect "
    "approximation was applied."
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
        # A ratio must be positive, and an effect more negative than the
        # baseline would make it zero or negative. This also covers the CI
        # bound: `_bound_nearer_null` returns a value strictly between the
        # null and the point, so `baseline + bound` is at least
        # `min(baseline, baseline + point)`, and both are positive here.
        and counterfactual.predicted_absent_treatment + estimate.point > 0.0
    ):
        baseline = counterfactual.predicted_absent_treatment
        # Derived from the reported point, NOT from `observed_mean / baseline`.
        # Those agree only where the baseline is derived from the point --
        # true for aipw, difference_in_means, did_2x2 and callaway_santanna,
        # false for lin_regression (whose observed_mean is the raw treated-arm
        # mean while its point is the Lin-adjusted ATE) and for ipw (whose
        # observed_mean is unweighted while its baseline is Hajek-weighted).
        # On those two the arm-mean form answered a different question from
        # the one the report asks, always in the direction of overstating
        # robustness, and disagreed with `e_value_ci_bound` one line below,
        # which was already derived from the reported interval.
        ratio = (baseline + estimate.point) / baseline
        bound = _bound_nearer_null(estimate.ci.lower, estimate.ci.upper, 0.0)
        ratio_bound = None if bound is None else (baseline + bound) / baseline
        note, approximated = _COUNTERFACTUAL_RATIO_NOTE, False
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


_NO_PRE_PERIODS_REASON = (
    "A cross-section has no pre-treatment periods, so there is no earlier date "
    "to run a placebo treatment at."
)
_TOO_FEW_PRE_PERIODS_REASON = (
    "No cohort has two pre-treatment periods to compare, so the placebo-date "
    "cells cannot be formed. Reported as untested rather than as a pass."
)
_DONOR_POOL_PLACEBO_DATES_REASON = (
    "A donor pool has exactly one treated unit, and the placebo-date cells are "
    "built from a within-group variance needing at least two units on each "
    "side, so not one cell can be formed however long the pre-period is. This "
    "is not a claim that the design has no pre-treatment periods -- it usually "
    "has many. The in-time placebo the synthetic-control estimator reports as "
    "a diagnostic is the technique that fits this family."
)
_CROSS_SECTION_PLACEBO_UNITS_REASON = (
    "On a cross-section every row is a unit and roughly half of them are "
    "treated, so relabelling one control row is not a placebo of the same "
    "design as the analysis. Deferred rather than approximated."
)
_NO_DONOR_POOL_REASON = (
    "Leave-one-donor-out needs a donor pool, and {name} does not build one: "
    "it draws on every control unit at once rather than weighting a few."
)
# `_family` reads `max_treated_units == 1` off Requirements, and TWO estimators
# declare it: synthetic_control and bsts. The donor-pool machinery below is
# synthetic_control's own -- simplex placebo runs and a convex-hull-aware
# weight vector -- so bsts gets these two reasons rather than another
# estimator's numbers reported under its name.
# `test_the_donor_pool_family_is_exactly_these_two` pins the family membership
# these two strings are written for.
_NON_SIMPLEX_PLACEBO_UNITS_REASON = (
    "The donor-pool placebo reassigns treatment to one donor at a time and "
    "refits the simplex weights, and {name} does not fit simplex donor "
    "weights: its donors enter as state-space regressors under shrinkage "
    "priors, so the distribution those runs build is not the one this "
    "estimator would produce. Each round would also cost a full NUTS refit. "
    "Reported as untested rather than as a pass."
)
_NON_SIMPLEX_DONOR_POOL_REASON = (
    "Leave-one-donor-out drops one weighted donor and refits, and {name} "
    "carries no simplex weight vector to say which donors the fit rests on -- "
    "its donors are state-space regressors shrunk toward zero rather than "
    "weighted toward a convex combination. A version faithful to this "
    "estimator would refit the full posterior once per donor, which this "
    "phase does not build."
)
_TOO_FEW_NEVER_TREATED_REASON = (
    "This panel's never-treated pool cannot be split into a placebo treated "
    "group and a placebo control group that both clear the minimum arm size, "
    "so there is no placebo of the same design to run."
)
_PLACEBO_ROUNDS_ALL_FAILED_REASON = (
    "Every placebo round was drawn but none could be fitted: at this panel's "
    "size the placebo group is small enough that the estimator's own bootstrap "
    "degenerates on most resamples. Reported as untested rather than as a pass."
)
_PANEL_RI_REASON = (
    "Randomization inference here permutes the treatment column, which on a "
    "panel would break the unit and period structure and the absorbing "
    "adoption pattern. A correct panel version permutes adoption at the unit "
    "level, which is different machinery and is not built yet."
)


def _family(estimator_name: str) -> str:
    """Which data shape an estimator declares it needs.

    Read off `Requirements` rather than a hardcoded name list, so a new
    estimator routes correctly without this module being edited. Phase 6
    already found one exact-set assertion that a newly registered estimator
    silently broke.
    """
    requirements = get_estimator(estimator_name).requirements
    if not requirements.needs_panel:
        return "cross_section"
    if requirements.max_treated_units == 1:
        return "donor_pool"
    return "panel"


def _rank_p_value(real_effect: float, placebo_effects: np.ndarray) -> float:
    """Two-sided rank p-value, reusing Phase 4's construction exactly.

    `permutation_p_value` ranks index 0 against the whole array, counting the
    real value itself -- so the smallest achievable p-value is 1/(n+1), which
    is what makes it honest at small permutation counts. Prepending the real
    effect and taking absolute values is the whole adaptation; no second
    formula for the same idea.
    """
    stacked = np.concatenate([[abs(real_effect)], np.abs(placebo_effects)])
    return permutation_p_value(stacked)


def randomization_inference(
    dataset: CausalDataset,
    estimator_name: str,
    config: EstimatorConfig,
    *,
    n_permutations: int = 500,
    seed: int = 0,
) -> PlaceboTestResult:
    """Refit under many random reassignments of treatment.

    Under a true null the real estimate is one unremarkable draw from the
    permutation distribution; under a real effect it sits in the tail. The
    p-value is the real estimate's rank, not a normal approximation, so it
    stays valid at small samples where a Wald interval would not.

    Cross-sections only -- see `_PANEL_RI_REASON`.
    """
    if dataset.schema.is_panel:
        raise ValueError(_PANEL_RI_REASON)
    estimator_class = get_estimator(estimator_name)
    real = estimator_class().fit(dataset, config).point

    rng = np.random.default_rng(np.random.SeedSequence(seed))
    treatment = dataset.frame[dataset.schema.treatment].to_numpy()
    effects = np.empty(n_permutations)
    for index in range(n_permutations):
        frame = dataset.frame.copy()
        frame[dataset.schema.treatment] = rng.permutation(treatment)
        permuted = CausalDataset(frame=frame, schema=dataset.schema)
        effects[index] = estimator_class().fit(permuted, config).point

    return PlaceboTestResult(
        placebo_effects=effects,
        real_effect=real,
        p_value=_rank_p_value(real, effects),
    )


def _panel_placebo_units(
    dataset: CausalDataset,
    estimator_name: str,
    config: EstimatorConfig,
    n_placebo_units: int,
    seed: int,
) -> tuple[PlaceboTestResult | None, str | None]:
    """Relabel never-treated units as treated, refit, and see what turns up.

    The genuinely treated units are REMOVED from each placebo frame, not left
    in. Leaving them in would let the real effect leak into the comparison the
    placebo exists to null out.

    Returns the result paired with `None`, or `None` paired with the reason it
    could not be run -- the three ways it can decline are different facts about
    the data and a single blanket reason would misreport two of them.
    """
    schema = dataset.schema
    panel = panel_arrays(dataset)
    adoption = panel.adoption_periods
    if not adoption:
        return None, _TOO_FEW_PRE_PERIODS_REASON
    if panel.n_never_treated_units < 2 * MIN_UNITS_PER_ARM:
        return None, _TOO_FEW_NEVER_TREATED_REASON

    estimator_class = get_estimator(estimator_name)
    real = estimator_class().fit(dataset, config).point
    never_treated = panel.units[panel.cohort == 0]
    # The placebo group is drawn only from never-treated units, and the units
    # it does not take have to remain as its controls -- so it cannot simply
    # match the real treated group's size. Measured across all seven panel
    # scenarios in this repo, the never-treated pool is never more than about
    # half the units (P1/P2: 31 never against 29 treated; P3-P5, P7: 45-48
    # against 72-75), so a placebo group sized to the real one would consume
    # the entire pool and leave no controls. An earlier version required
    # `n_never_treated >= 2 * n_treated` before running at all, which no
    # scenario in this project satisfies -- the technique would have been dead
    # code that always reported itself absent.
    #
    # Splitting the pool in half is what makes it run. The consequence, worth
    # stating because it is not free: on these scenarios the placebo group is
    # roughly half the size of the real treated group, so each placebo
    # estimate is noisier than the real one and the placebo distribution is
    # wider than the estimator's own sampling distribution. That biases the
    # rank p-value toward the null -- it is harder, not easier, for a placebo
    # to look as extreme as the real effect. Conservative is the safe
    # direction for a falsification test, but it is a real limit on power.
    n_fake = min(panel.n_treated_units, panel.n_never_treated_units // 2)
    if n_fake < MIN_UNITS_PER_ARM:
        return None, _TOO_FEW_NEVER_TREATED_REASON
    start = adoption[0]
    # `panel_arrays` above raises on a schema carrying no time column, so this
    # is unreachable; it is here to narrow the type for the checker rather
    # than to handle a case that can occur.
    time_column = schema.time
    if time_column is None:
        return None, _TOO_FEW_PRE_PERIODS_REASON

    rng = np.random.default_rng(np.random.SeedSequence(seed))
    effects: list[float] = []
    for _ in range(n_placebo_units):
        fake = rng.choice(never_treated, size=n_fake, replace=False)
        frame = dataset.frame[dataset.frame[schema.unit].isin(never_treated)].copy()
        placebo_treated = (
            frame[schema.unit].isin(fake) & (frame[time_column] >= start)
        ).astype(int)
        frame[schema.treatment] = placebo_treated
        if schema.cohort is not None:
            # `cohort_by_unit` derives cohorts from the treatment column, so
            # nothing reads this today -- but leaving the original column in
            # place would have it assert every placebo unit is never-treated
            # while the treatment column says otherwise, and a frame that
            # contradicts itself is one refactor away from being believed.
            frame[schema.cohort] = np.where(frame[schema.unit].isin(fake), start, 0)
        candidate = CausalDataset(frame=frame.reset_index(drop=True), schema=schema)
        if estimator_class.requirements.unmet(candidate):
            continue
        try:
            effects.append(float(estimator_class().fit(candidate, config).point))
        except ValueError:
            # A placebo group is half the never-treated pool, so on a small
            # panel it can clear `Requirements` and still be too small for the
            # estimator's own bootstrap, which refuses once more than half its
            # replicates come back degenerate. Measured on P1: this fires at
            # n_units=60 (15 placebo-treated against 16 controls) and stops by
            # n_units=100 (25 against 25). Skipping the round matches how an
            # unfittable candidate is already handled one line above; if every
            # round fails, the caller gets a stated absence rather than this
            # exception surfacing from three frames down.
            continue

    if not effects:
        return None, _PLACEBO_ROUNDS_ALL_FAILED_REASON
    array = np.asarray(effects)
    result = PlaceboTestResult(
        placebo_effects=array, real_effect=real, p_value=_rank_p_value(real, array)
    )
    return result, None


def falsify(
    dataset: CausalDataset,
    estimator_name: str,
    config: EstimatorConfig,
    *,
    n_permutations: int = 500,
    n_placebo_units: int = 20,
    seed: int = 0,
) -> FalsificationResult:
    """Four tests a spurious finding should fail, each routed to the estimator
    family it is defined for.

    Expensive on the MCMC estimators: a panel placebo round is a full refit,
    which for `bsts` and `hierarchical_did` means a full NUTS run.
    `n_placebo_units` defaults to 20 for that reason, and a caller running
    those estimators should lower it deliberately rather than discover the
    cost.
    """
    get_estimator(estimator_name)
    family = _family(estimator_name)

    placebo_dates: PlaceboTestResult | None = None
    placebo_dates_reason: str | None = None
    if family == "cross_section":
        placebo_dates_reason = _NO_PRE_PERIODS_REASON
    elif family == "donor_pool":
        # Routed by family rather than discovered by running `placebo_leads`
        # and reporting whatever it returns. It returns empty here for every
        # donor pool -- `group_change` yields nan below two units per group --
        # and the empty-result reason would then claim the pre-period was too
        # short, which on S1 (16 usable pre-periods) is simply false.
        placebo_dates_reason = _DONOR_POOL_PLACEBO_DATES_REASON
    else:
        panel = panel_arrays(dataset)
        _, coefficients, _ = placebo_leads(panel, "never_treated")
        if coefficients.size == 0:
            placebo_dates_reason = _TOO_FEW_PRE_PERIODS_REASON
        else:
            real = get_estimator(estimator_name)().fit(dataset, config).point
            placebo_dates = PlaceboTestResult(
                placebo_effects=coefficients,
                real_effect=real,
                p_value=_rank_p_value(real, coefficients),
            )

    placebo_units: PlaceboTestResult | None = None
    placebo_units_reason: str | None = None
    leave_one_donor_out: tuple[float, ...] | None = None
    leave_one_donor_out_reason: str | None = None
    # Gated on the estimator by name, not on `family`, and that is the one
    # place in this module where a name beats the Requirements-derived family.
    # Everything below is synthetic_control's own machinery: `placebo_runs`,
    # `fit_donor_pool_point` and `leave_one_donor_out_points` all fit simplex
    # (or ridge-augmented) donor weights, and none of them consults
    # `estimator_name`. `bsts` declares `max_treated_units == 1` too, so
    # routing by family alone fitted a synthetic control and reported its
    # numbers under the bsts name -- measured on S1, a bsts point of 1.17743
    # against a reported real_effect of 1.41401, with both reason fields None
    # asserting those were bsts's own results.
    if estimator_name == "synthetic_control":
        panel = panel_arrays(dataset)
        treated_series, donor_series, pre_mask = donor_matrix(panel)
        post_mask = ~pre_mask
        runs = placebo_runs(treated_series, donor_series, pre_mask, post_mask)
        # `runs.point_estimates` (including index 0, the real treated unit)
        # is always simplex-based -- see `PlaceboRuns`: it exists to build
        # the null/reference distribution the simplex fit is judged against,
        # not to report the headline number. The real point, and the weight
        # form leave-one-donor-out must match, come from
        # `fit_donor_pool_point` instead, which is convex-hull-aware exactly
        # the way `SyntheticControl.fit()` itself is. Using
        # `runs.point_estimates[0]` unconditionally here previously reported
        # a sign-flipped, ~2.5x-off number whenever the convex-hull check
        # failed -- caught by Task 7's review reproducing it against S3.
        donor_fit = fit_donor_pool_point(treated_series, donor_series, pre_mask, post_mask)
        placebo_units = PlaceboTestResult(
            placebo_effects=runs.point_estimates[1:],
            real_effect=donor_fit.point,
            p_value=permutation_p_value(runs.rmspe_ratios),
        )
        _, leave_one_donor_out = leave_one_donor_out_points(
            treated_series,
            donor_series,
            pre_mask,
            post_mask,
            reference_weights=donor_fit.reference_weights,
            use_ridge=donor_fit.use_ridge,
        )
    elif family == "donor_pool":
        placebo_units_reason = _NON_SIMPLEX_PLACEBO_UNITS_REASON.format(
            name=estimator_name
        )
        leave_one_donor_out_reason = _NON_SIMPLEX_DONOR_POOL_REASON.format(
            name=estimator_name
        )
    else:
        leave_one_donor_out_reason = _NO_DONOR_POOL_REASON.format(name=estimator_name)
        if family == "cross_section":
            placebo_units_reason = _CROSS_SECTION_PLACEBO_UNITS_REASON
        else:
            placebo_units, placebo_units_reason = _panel_placebo_units(
                dataset, estimator_name, config, n_placebo_units, seed
            )

    randomization: PlaceboTestResult | None = None
    randomization_reason: str | None = None
    if family == "cross_section":
        randomization = randomization_inference(
            dataset,
            estimator_name,
            config,
            n_permutations=n_permutations,
            seed=seed,
        )
    else:
        randomization_reason = _PANEL_RI_REASON

    return FalsificationResult(
        placebo_dates=placebo_dates,
        placebo_dates_reason=placebo_dates_reason,
        placebo_units=placebo_units,
        placebo_units_reason=placebo_units_reason,
        leave_one_donor_out=leave_one_donor_out,
        leave_one_donor_out_reason=leave_one_donor_out_reason,
        randomization_inference=randomization,
        randomization_inference_reason=randomization_reason,
    )


# Paired with cinelli_hazlett's own `None` returns inside sensitivity_report,
# below -- cinelli_hazlett itself stays a pure "does this apply" function that
# reports nothing about why, and the report is the one place that turns the
# absence into a stated reason. `event_study` gets its own wording because its
# exclusion is a modelling fact (its point is a mean of several horizon
# coefficients, not one coefficient to bound), not a "this estimator is not
# OLS" fact.
_NOT_OLS_REASON = (
    "cinelli_hazlett bounds the t-statistic of a single OLS treatment "
    "coefficient; {name} is not fitted by least squares against one such "
    "coefficient, so there is no coefficient for the bound to attach to."
)

_EVENT_STUDY_REASON = (
    "cinelli_hazlett bounds a single OLS treatment coefficient, and "
    "event_study's reported point is the mean of several horizon "
    "coefficients rather than one -- bounding a linear combination's "
    "t-statistic would still compute, but the result would not mean what "
    "this method's derivation assumes it means."
)

# cinelli_hazlett also returns None for an estimator that IS in
# OLS_SENSITIVITY_ESTIMATORS, when its own refit is numerically degenerate
# (too few residual degrees of freedom, or a zero coefficient standard
# error). That is a different fact from "not OLS-fitted" and _NOT_OLS_REASON
# would misstate it, so it gets its own wording. Deliberately silent on
# which of the two degeneracy conditions fired -- cinelli_hazlett's contract
# does not say, and restructuring it to say is out of scope here.
_DEGENERATE_OLS_FIT_REASON = (
    "{name} is fitted by least squares against a single treatment "
    "coefficient, so cinelli_hazlett normally applies to it, but this "
    "particular refit could not compute a usable coefficient standard error, so "
    "the bound could not be computed."
)


def sensitivity_report(
    dataset: CausalDataset,
    estimator_name: str,
    estimate: EffectEstimate,
    config: EstimatorConfig,
    *,
    outcome_sd: float | None = None,
    n_permutations: int = 500,
    n_placebo_units: int = 20,
    seed: int = 0,
) -> SensitivityReport:
    """Everything this module can say about how robust one finding is.

    Three independent lines of attack under one type, with a stated reason
    wherever a technique does not apply. None of them is a test of whether
    the effect is real: a clean sensitivity report is consistent with an
    unmeasured confounder nobody thought to look for, and the report says so
    rather than implying otherwise.
    """
    bound = cinelli_hazlett(dataset, estimator_name, config)
    if bound is not None:
        cinelli_hazlett_reason = None
    elif estimator_name == "event_study":
        cinelli_hazlett_reason = _EVENT_STUDY_REASON
    elif estimator_name in OLS_SENSITIVITY_ESTIMATORS:
        # In the OLS set, yet still None: the refit itself was degenerate
        # (see _DEGENERATE_OLS_FIT_REASON), not "this estimator doesn't
        # apply".
        cinelli_hazlett_reason = _DEGENERATE_OLS_FIT_REASON.format(name=estimator_name)
    else:
        cinelli_hazlett_reason = _NOT_OLS_REASON.format(name=estimator_name)
    confounding = ConfoundingSensitivity(
        e_value=e_value(estimate, outcome_sd=outcome_sd),
        cinelli_hazlett=bound,
        cinelli_hazlett_reason=cinelli_hazlett_reason,
    )

    curve: SpecificationCurve | None = None
    curve_reason: str | None = None
    if estimator_name in CURVE_ESTIMATORS:
        # Called directly, not inside a try/except: specification_curve can
        # also raise ValueError for a *supported* estimator whose base
        # specification cannot be fitted on the dataset as declared, and
        # estimator_class().fit() can raise ValueError from inside a
        # candidate refit (e.g. aipw's cross-fitting-fold-too-small guard).
        # Both are genuine failures, not "this estimator is out of scope",
        # so they must propagate rather than be relabelled as an absence.
        # Checking membership up front means the only case handled here is
        # the one this report is actually about.
        curve = specification_curve(dataset, estimator_name, config)
    else:
        curve_reason = _UNSUPPORTED_CURVE_REASON.format(
            supported=sorted(CURVE_ESTIMATORS), name=estimator_name
        )

    return SensitivityReport(
        confounding=confounding,
        specification_curve=curve,
        specification_curve_reason=curve_reason,
        falsification=falsify(
            dataset,
            estimator_name,
            config,
            n_permutations=n_permutations,
            n_placebo_units=n_placebo_units,
            seed=seed,
        ),
    )
