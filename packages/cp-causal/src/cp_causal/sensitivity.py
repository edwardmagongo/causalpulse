"""Sensitivity analysis: how hard would it be to overturn this finding?

Three independent lines of attack, and they are independent on purpose. A
confounding bound says nothing about whether the result survives a different
covariate set; a specification curve says nothing about whether a placebo
period looks the same. Reporting all three under one type is what stops a
reader taking one clean answer as the whole picture.
"""

from __future__ import annotations

import math

from cp_core.results import (
    CounterfactualPoint,
    EffectEstimate,
    EValueResult,
)

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
