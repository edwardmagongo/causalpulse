"""Assumption checks for the difference-in-differences family.

Built once and parameterised, so that four estimators cannot let the same
statement drift apart across four copies.
"""

from __future__ import annotations

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import AssumptionCheck
from cp_statistics.linalg_safety import safe_symmetric_pinv
from scipy import stats

from cp_causal._panel import PanelArrays

_ALPHA = 0.05


def unit_means(panel: PanelArrays) -> np.ndarray:
    """Outcome by (unit, period), with nan where a cell is unobserved.

    Built once per fit. An unbalanced panel leaves nan cells, and every group
    comparison drawn from this grid should treat a nan as "not observed", not
    as zero.
    """
    grid = np.full((panel.n_units, panel.n_periods), np.nan)
    grid[panel.unit_index, panel.period_index] = panel.outcome
    return grid


def group_change(
    grid: np.ndarray, members: np.ndarray, column: int, baseline: int
) -> tuple[float, float, int]:
    """Mean within-unit change (column vs. baseline column), its variance, and
    the contributing unit count.

    The change is differenced *within each unit* before averaging, so each
    unit contributes one observation and the resulting standard error is
    clustered on unit by construction rather than by a separate correction.
    """
    if not members.any():
        return float("nan"), float("nan"), 0
    changes = grid[members, column] - grid[members, baseline]
    finite = changes[np.isfinite(changes)]
    if finite.size < 2:
        return float("nan"), float("nan"), int(finite.size)
    return float(finite.mean()), float(finite.var(ddof=1)), int(finite.size)


def placebo_leads(
    panel: PanelArrays, control_group: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-treatment placebo cells: the 2x2 machinery run on periods *before*
    adoption, where the true effect is zero by construction. A non-zero
    placebo is a differential trend.

    Shared by `CallawaySantAnna` (which folds the result straight into
    `parallel_trends_check`) and `HierarchicalDiD` (which additionally picks
    the single cell nearest its own cohort's adoption for
    `no_anticipation_check`). This function does not pick a "nearest" cell
    itself -- that is a caller-specific concern -- which is why `keys` is
    returned: without the (cohort, period) identifying each cell, a caller
    that needs the nearest one would have no way to find it after the fact.

    For each cohort `g` with baseline period `g - 1` (its last untreated
    period) and each earlier period `t < g - 1`, compares cohort `g`'s mean
    change from `t` to the baseline against the same change among the
    `control_group` -- "never_treated" (only never-treated units) or any
    other value, treated as "not_yet_treated" (never-treated plus every
    later-adopting cohort).

    Returns `(keys, coefficients, covariance)`. `keys` is `(n, 2)`, each row
    `(cohort, period)` identifying the cell at that index of `coefficients`
    and of `covariance`'s diagonal. All three are empty (`(0, 2)`, `(0,)`,
    `(0, 0)`) when no cohort has two pre-treatment periods to compare --
    callers report that as NOT_TESTABLE, not as a passing test.

    Diagonal covariance from EACH CELL'S OWN analytic variance, via the same
    two-independent-samples formula the real ATT(g,t)/treatment-effect cells
    use -- not a single variance pooled across cells.

    An earlier version pooled `values.var(ddof=1)` across cells and divided
    by the cell count, which is the standard-error-of-the-mean formula
    misapplied to a per-observation covariance -- and even after removing
    the erroneous division, a single pooled variance still ignores that the
    placebo cells sharing one cohort share the same baseline period and are
    therefore correlated, not independent draws of a common distribution.
    Testing correlated values against a diagonal covariance that assumes
    independence inflates the chi-square statistic and over-rejects.

    Measured on scenario P1 (clean, no violation, 3 correlated placebo cells
    sharing one baseline), 40 seeds: the original pooled-and-divided form
    FAILED 16/20 seeds against a nominal 5% rate; pooled-without-dividing
    still FAILED 8/20; this per-cell-variance form FAILS 4/40 (10%) -- still
    somewhat above nominal, consistent with a chi-square approximation on
    only 3 correlated degrees of freedom, but no longer a gross
    miscalibration. P5 (a real violation) is still caught 10/10.
    """
    grid = unit_means(panel)
    keys: list[tuple[int, int]] = []
    placebos: list[float] = []
    placebo_variances: list[float] = []
    for cohort in panel.adoption_periods:
        baseline = panel.period_position(cohort - 1)
        if baseline < 0:
            continue
        members = panel.cohort == cohort
        if not members.any():
            continue
        for column, period in enumerate(panel.periods.astype(int)):
            if period >= cohort - 1 or column >= baseline:
                continue
            controls = (
                (panel.cohort == 0)
                if control_group == "never_treated"
                else (panel.cohort == 0) | (panel.cohort > cohort)
            )
            if not controls.any():
                continue
            treated_change, treated_var, n_treated = group_change(
                grid, members, column, baseline
            )
            control_change, control_var, n_control = group_change(
                grid, controls, column, baseline
            )
            if not np.isfinite(treated_change) or not np.isfinite(control_change):
                continue
            keys.append((int(cohort), int(period)))
            placebos.append(treated_change - control_change)
            placebo_variances.append(treated_var / n_treated + control_var / n_control)

    if not placebos:
        return np.zeros((0, 2), dtype=int), np.asarray([]), np.zeros((0, 0))

    values = np.asarray(placebos)
    diagonal = np.maximum(np.asarray(placebo_variances), 1e-12)
    return np.asarray(keys, dtype=int), values, np.diag(diagonal)


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
            kind="identification",
            status="NOT_TESTABLE",
            statistic=None,
            threshold=None,
            interpretation=(
                "There are no pre-treatment leads to test: the design has a "
                "single pre-period, so no differential trend can be measured. "
                "This is not evidence that trends are parallel."
            ),
            remedy=(
                "Extend the panel backwards so at least two pre-treatment periods are observed."
            ),
        )

    # safe_symmetric_pinv, not np.linalg.pinv directly: the same platform SVD
    # gap `cp_statistics.linalg_safety` documents can equally strike this
    # covariance-matrix pinv as it did the two calls found inside `ols_fit`.
    precision = safe_symmetric_pinv(lead_covariance)
    statistic = float(lead_coefficients @ precision @ lead_coefficients)
    degrees = int(lead_coefficients.size)
    p_value = float(stats.chi2.sf(statistic, degrees))
    rejected = p_value < alpha

    return AssumptionCheck(
        name="parallel_trends",
        kind="identification",
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
    coefficient: float,
    standard_error: float,
    alpha: float = _ALPHA,
    horizon_label: str = "lead immediately before adoption",
) -> AssumptionCheck:
    """A single pre-treatment lead, chosen and described by the caller.

    Kept separate from the joint test because anticipation and differential
    trends are different violations with different remedies: anticipation is
    fixed by moving the reference period back, a differential trend is not.

    `horizon_label` names which lead `coefficient` actually came from, so the
    generated text never claims to describe a period the caller did not
    measure. The default ("lead immediately before adoption") assumes the
    caller is passing exactly that lead; a caller substituting a different
    lead -- e.g. because the true immediately-prior period is the omitted
    reference category and structurally unmeasurable, as in `EventStudy` --
    must override it with an accurate description of what was tested.
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
        kind="identification",
        status="FAIL" if rejected else "PASS",
        statistic=coefficient,
        threshold=None,
        interpretation=(
            f"The {horizon_label} is {coefficient:+.4f} "
            f"(z = {z:.2f}, p = {p_value:.4f}). "
            + (
                f"Outcomes were already moving at the {horizon_label}, "
                "which is what anticipation looks like."
                if rejected
                else f"No detectable movement at the {horizon_label}."
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
        kind="data_adequacy",
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


def sufficient_pre_periods_check(panel: PanelArrays, minimum: int) -> AssumptionCheck:
    """Observed pre-treatment periods against the estimator's declared minimum."""
    adoption = panel.adoption_periods
    earliest = min(adoption) if adoption else int(panel.periods.max()) + 1
    observed = int(np.count_nonzero(panel.periods < earliest))
    ok = observed >= minimum

    return AssumptionCheck(
        name="sufficient_pre_periods",
        kind="data_adequacy",
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


def pre_treatment_fit_check(
    pre_rmspe: float, outcome_sd: float, threshold: float = 1.75
) -> AssumptionCheck:
    """Pre-treatment fit quality, relative to the outcome's own dispersion.

    Threshold measured (design doc §2 item 5), 100 seeds each: on the clean
    S1 scenario the ratio's mean is 0.65 and its max is 1.53; on S3 (the
    convex-hull-violation scenario, before the ridge fallback) the mean is
    4.76 and the min is 2.04. 1.75 sits cleanly between the two.
    """
    ratio = pre_rmspe / outcome_sd if outcome_sd > 0 else float("inf")
    ok = ratio <= threshold
    return AssumptionCheck(
        name="pre_treatment_fit",
        kind="data_adequacy",
        status="PASS" if ok else "FAIL",
        statistic=ratio,
        threshold=threshold,
        interpretation=(
            f"Pre-treatment RMSPE is {ratio:.2f}x the outcome's own "
            f"pre-period standard deviation " + ("(a good fit)." if ok else "(a poor fit).")
        ),
        remedy=(
            None
            if ok
            else "The donor pool cannot closely track this unit's "
            "pre-treatment path; consider a larger or more comparable "
            "donor pool."
        ),
    )


def convex_hull_check(treated_pre_mean: float, donor_pre_means: np.ndarray) -> AssumptionCheck:
    """Whether the treated unit's pre-period mean lies within the donor range.

    Interpolation bias is synthetic control's principal failure mode (parent
    spec §5.4): a simplex-constrained combination of donors can only ever
    reproduce values inside their convex hull, so a treated unit outside it
    cannot be matched by construction, not merely by chance.
    """
    lower = float(donor_pre_means.min())
    upper = float(donor_pre_means.max())
    inside = lower <= treated_pre_mean <= upper
    return AssumptionCheck(
        name="convex_hull",
        kind="identification",
        status="PASS" if inside else "FAIL",
        statistic=treated_pre_mean,
        threshold=None,
        interpretation=(
            f"The treated unit's pre-period mean ({treated_pre_mean:.3f}) "
            + (
                f"lies within the donor pool's range [{lower:.3f}, {upper:.3f}]."
                if inside
                else f"lies outside the donor pool's range [{lower:.3f}, "
                f"{upper:.3f}]. No non-negative, simplex-constrained "
                "combination of donors can match it."
            )
        ),
        remedy=(
            None
            if inside
            else "A ridge-augmented fit was used instead of the plain "
            "simplex weights; treat the result with more caution than an "
            "in-hull match, and consider expanding the donor pool."
        ),
    )


def donor_contamination_check(
    donor_rmspe_ratios: np.ndarray, z_threshold: float = 5.5
) -> AssumptionCheck:
    """Flags a donor whose own placebo RMSPE ratio is a robust outlier.

    Reuses the same placebo procedure permutation inference already runs
    (each donor playing "treated" in turn, `placebo_runs` in
    `synthetic_control.py`) rather than a bespoke statistical test.

    `worst` is the *maximum* robust z-score across all 17 donors, not one
    pre-specified donor's z-score -- and that distinction is load-bearing.
    An earlier version of this check was calibrated (threshold 2.0) against
    a single pre-specified donor's z-score, which measured a false-positive
    rate of 5/100. Deployed against the real `max` statistic this task
    actually needs (nobody knows in advance which donor might be
    contaminated), that same threshold produced 72/100 false positives --
    the classic multiple-comparisons gap between "is this one donor
    extreme" and "is the most extreme of 17 donors extreme", caught by
    Task 7's recovery harness, not by this check's own unit tests, which
    never measured the max-over-all-donors statistic's real null
    distribution. Recalibrated by measuring the *actual* `max` statistic's
    distribution directly (300 seeds): threshold 5.5, false-positive rate
    7/100 on a clean donor pool, true-positive rate 96/100 against a donor
    carrying its own +5.0 hidden shock (also corrected up from +1.5, which
    was not reliably detectable once the check was properly calibrated --
    see `donor_scenarios.py`'s S4).
    """
    median = float(np.median(donor_rmspe_ratios))
    mad = float(np.median(np.abs(donor_rmspe_ratios - median))) + 1e-9
    z_scores = (donor_rmspe_ratios - median) / (1.4826 * mad)
    worst = float(np.max(z_scores))
    flagged = worst > z_threshold
    return AssumptionCheck(
        name="donor_contamination",
        kind="identification",
        status="FAIL" if flagged else "PASS",
        statistic=worst,
        threshold=z_threshold,
        interpretation=(
            f"The most extreme donor's own placebo ratio is {worst:.2f} "
            "robust standard deviations from the rest of the pool's median."
            + (
                " At least one donor's post-period behaviour looks "
                "unlike what its own donor pool would have predicted, "
                "which is what a hidden shock to that donor looks like "
                "from this test's point of view."
                if flagged
                else " No donor stands out from the rest."
            )
        ),
        remedy=(
            "Inspect the flagged donor's own series for an unrelated shock "
            "or intervention around the treatment date, and consider "
            "excluding it from the donor pool."
            if flagged
            else None
        ),
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
        kind="identification",
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
