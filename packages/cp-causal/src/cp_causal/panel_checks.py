"""Assumption checks for the difference-in-differences family.

Built once and parameterised, so that four estimators cannot let the same
statement drift apart across four copies.
"""

from __future__ import annotations

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import AssumptionCheck
from scipy import stats

from cp_causal._panel import PanelArrays

_ALPHA = 0.05


def parallel_trends_check(
    lead_coefficients: np.ndarray,
    lead_covariance: np.ndarray,
    alpha: float = _ALPHA,
) -> AssumptionCheck:
    """Joint Wald test on the pre-treatment leads.

    With a single pre-period there are no leads to test at all, and the status
    is `NOT_TESTABLE` rather than a confident PASS. That is the easy bug in this
    family and exactly what the status exists for: reporting PASS would turn
    "we could not look" into "we looked and it was fine".
    """
    if lead_coefficients.size == 0:
        return AssumptionCheck(
            name="parallel_trends",
            status="NOT_TESTABLE",
            statistic=None,
            threshold=None,
            interpretation=(
                "There are no pre-treatment leads to test: the design has a "
                "single pre-period, so no differential trend can be measured. "
                "This is not evidence that trends are parallel."
            ),
            remedy=(
                "Extend the panel backwards so at least two pre-treatment "
                "periods are observed."
            ),
        )

    precision = np.linalg.pinv(lead_covariance)
    statistic = float(lead_coefficients @ precision @ lead_coefficients)
    degrees = int(lead_coefficients.size)
    p_value = float(stats.chi2.sf(statistic, degrees))
    rejected = p_value < alpha

    return AssumptionCheck(
        name="parallel_trends",
        status="FAIL" if rejected else "PASS",
        statistic=statistic,
        threshold=float(stats.chi2.ppf(1 - alpha, degrees)),
        interpretation=(
            (
                f"The {degrees} pre-treatment leads are jointly different from "
                f"zero (chi2 = {statistic:.2f}, p = {p_value:.4f}). Treated and "
                "control units were already diverging before treatment, so the "
                "estimate confounds the effect with a pre-existing trend."
            )
            if rejected
            else (
                f"The {degrees} pre-treatment leads are not jointly "
                f"distinguishable from zero (chi2 = {statistic:.2f}, "
                f"p = {p_value:.4f}). Failure to reject is not evidence that "
                "parallel trends holds: the test has limited power, and the "
                "assumption concerns counterfactual post-treatment trends that "
                "no pre-treatment data can observe."
            )
        ),
        remedy=(
            "Inspect the event-study path, consider a comparison group with "
            "closer pre-trends, or report the breakdown value alongside."
            if rejected
            else None
        ),
    )


def no_anticipation_check(
    coefficient: float, standard_error: float, alpha: float = _ALPHA
) -> AssumptionCheck:
    """The single lead immediately before adoption.

    Kept separate from the joint test because anticipation and differential
    trends are different violations with different remedies: anticipation is
    fixed by moving the reference period back, a differential trend is not.
    """
    if standard_error <= 0:
        z = 0.0
        p_value = 1.0
    else:
        z = float(coefficient / standard_error)
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    rejected = p_value < alpha

    return AssumptionCheck(
        name="no_anticipation",
        status="FAIL" if rejected else "PASS",
        statistic=coefficient,
        threshold=None,
        interpretation=(
            f"The lead immediately before adoption is {coefficient:+.4f} "
            f"(z = {z:.2f}, p = {p_value:.4f}). "
            + (
                "Outcomes were already moving in the period before treatment, "
                "which is what anticipation looks like."
                if rejected
                else "No detectable movement in the period before treatment."
            )
        ),
        remedy=(
            "Move the reference period one or more periods earlier so the "
            "anticipation window sits inside the pre-treatment baseline."
            if rejected
            else None
        ),
    )


def absorbing_treatment_check(dataset: CausalDataset) -> AssumptionCheck:
    """Whether any unit switches treatment back off.

    The contract layer deliberately permits this — some interventions genuinely
    switch off — so something must check it, and DiD is where it changes the
    answer.
    """
    absorbing = dataset.treatment_is_absorbing
    return AssumptionCheck(
        name="absorbing_treatment",
        status="PASS" if absorbing else "FAIL",
        statistic=None,
        threshold=None,
        interpretation=(
            "Treatment is absorbing: no unit returns to untreated once treated."
            if absorbing
            else (
                "At least one unit switches treatment back off. The estimators "
                "in this family define cohorts by first adoption and treat "
                "them as treated thereafter, so a switching unit is "
                "misclassified for every period after it reverts."
            )
        ),
        remedy=(
            None
            if absorbing
            else "Use a design that models switching, or restrict the sample to "
            "units whose treatment never reverts."
        ),
    )


def sufficient_pre_periods_check(
    panel: PanelArrays, minimum: int
) -> AssumptionCheck:
    """Observed pre-treatment periods against the estimator's declared minimum."""
    adoption = panel.adoption_periods
    earliest = min(adoption) if adoption else int(panel.periods.max()) + 1
    observed = int(np.count_nonzero(panel.periods < earliest))
    ok = observed >= minimum

    return AssumptionCheck(
        name="sufficient_pre_periods",
        status="PASS" if ok else "FAIL",
        statistic=float(observed),
        threshold=float(minimum),
        interpretation=(
            f"{observed} pre-treatment period(s) are observed before the "
            f"earliest adoption at period {earliest}; this estimator declares a "
            f"minimum of {minimum}."
        ),
        remedy=None if ok else "Extend the panel backwards.",
    )


def no_concurrent_intervention_check() -> AssumptionCheck:
    """Hardcoded NOT_TESTABLE, with no branch and no computed condition.

    Identical treatment to `unconfoundedness` in propensity.py. Whether some
    other change coincided with adoption is a fact about the world, not about
    the data: nothing in the panel can distinguish the treatment's effect from
    a simultaneous unrelated shock hitting only treated units.
    """
    return AssumptionCheck(
        name="no_concurrent_intervention",
        status="NOT_TESTABLE",
        statistic=None,
        threshold=None,
        interpretation=(
            "Difference-in-differences attributes the post-adoption divergence "
            "to the treatment. Whether another change hit the treated units at "
            "the same time cannot be determined from this data at all, and no "
            "diagnostic here should be read as ruling it out."
        ),
        remedy=(
            "Check deployment and campaign calendars for the adoption window, "
            "and run a placebo on an outcome the treatment should not affect."
        ),
    )
