"""Propensity model and the diagnostics that decide whether to trust it.

scikit-learn fits the model; every causal quantity here — the balance metric,
the effective sample size, the overlap judgements — is computed directly, since
those are what the conclusion actually rests on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from cp_core.results import AssumptionCheck, AssumptionStatus
from sklearn.linear_model import LogisticRegression

# Standard applied threshold: |SMD| under 0.1 is conventionally read as
# adequate balance on a covariate.
BALANCE_THRESHOLD = 0.1
# Below this, a unit is close enough to deterministic assignment that its
# inverse weight explodes and the estimate is driven by a handful of rows.
POSITIVITY_THRESHOLD = 0.01


@dataclass(frozen=True, eq=False)
class PropensityFit:
    scores: np.ndarray
    trim_bounds: tuple[float, float]
    trimmed_mask: np.ndarray


def resolve_trim(options: Mapping[str, object]) -> float:
    """Read a propensity-clipping threshold from an estimator's options.

    Shared by the two propensity-based estimators rather than written twice:
    the specification curve varies this axis across both, and two readers
    would be free to validate it differently.

    Bounded below 0.5 because clipping at or above it would collapse every
    score onto a single value and turn the weights into a constant.
    """
    raw = options.get("trim", POSITIVITY_THRESHOLD)
    # `bool` is a subclass of `int`, and `True` would otherwise silently
    # resolve to a trim of 1.0 -- which the range check below would reject,
    # but with an error naming a number the caller never wrote.
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise TypeError(f"options['trim'] must be a number, got {type(raw).__name__}")
    trim = float(raw)
    if not 0.0 <= trim < 0.5:
        raise ValueError(
            f"options['trim'] must lie in [0, 0.5), got {trim}; at or above 0.5 "
            "every propensity score clips to one value and the weights go flat"
        )
    return trim


def fit_propensity(
    covariates: np.ndarray,
    treatment: np.ndarray,
    *,
    cross_fit_folds: int = 0,
    rng: np.random.Generator | None = None,
    trim: float = POSITIVITY_THRESHOLD,
) -> PropensityFit:
    """Estimate P(treated | covariates).

    With `cross_fit_folds > 0` each unit's score comes from a model that never
    saw it. That matters for doubly-robust estimation, where reusing in-sample
    predictions reintroduces the overfitting bias the estimator exists to
    remove.
    """
    if covariates.shape[0] != treatment.shape[0]:
        raise ValueError(
            f"covariates has {covariates.shape[0]} rows but treatment has "
            f"{treatment.shape[0]} entries"
        )
    if covariates.shape[1] == 0:
        raise ValueError("propensity estimation requires at least one covariate")

    n = treatment.shape[0]
    if cross_fit_folds and cross_fit_folds > 1:
        if rng is None:
            raise ValueError(
                "cross_fit_folds > 1 requires an explicit rng: the fold assignment "
                "is random and must derive from the caller's SeedSequence rather "
                "than a generator seeded here"
            )
        scores = np.zeros(n)
        order = rng.permutation(n)
        for fold in range(cross_fit_folds):
            held_out = order[fold::cross_fit_folds]
            train = np.setdiff1d(order, held_out)
            model = LogisticRegression(max_iter=1000).fit(
                covariates[train], treatment[train]
            )
            scores[held_out] = model.predict_proba(covariates[held_out])[:, 1]
    else:
        model = LogisticRegression(max_iter=1000).fit(covariates, treatment)
        scores = model.predict_proba(covariates)[:, 1]

    clipped = np.clip(scores, trim, 1.0 - trim)
    return PropensityFit(
        scores=clipped,
        trim_bounds=(trim, 1.0 - trim),
        trimmed_mask=(scores < trim) | (scores > 1.0 - trim),
    )


def standardised_mean_differences(
    covariates: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Per-covariate standardised mean difference between arms.

    The denominator uses the *unweighted* pooled standard deviation so that
    before-and-after values are on one scale; using weighted variances would
    let a weighting scheme flatter itself.
    """
    treated = treatment == 1
    control = treatment == 0
    if weights is None:
        weights = np.ones(treatment.shape[0])

    def weighted_mean(mask: np.ndarray) -> np.ndarray:
        w = weights[mask]
        return np.asarray((covariates[mask] * w[:, None]).sum(axis=0) / w.sum())

    pooled = np.sqrt(
        (covariates[treated].var(axis=0, ddof=1) + covariates[control].var(axis=0, ddof=1))
        / 2.0
    )
    pooled = np.where(pooled == 0, np.nan, pooled)
    return np.asarray((weighted_mean(treated) - weighted_mean(control)) / pooled)


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size.

    A weighted analysis with a few dominant weights carries far less
    information than its row count suggests; this is the number that should be
    reported alongside n.
    """
    total = float(weights.sum())
    if total == 0:
        return 0.0
    return float(total**2 / np.square(weights).sum())


def overlap_checks(
    scores: np.ndarray, treatment: np.ndarray, smd_after: np.ndarray
) -> tuple[AssumptionCheck, ...]:
    """Judge whether a propensity-weighted comparison is defensible."""
    treated_scores = scores[treatment == 1]
    control_scores = scores[treatment == 0]
    shared_low = max(treated_scores.min(), control_scores.min())
    shared_high = min(treated_scores.max(), control_scores.max())
    outside = float(np.mean((scores < shared_low) | (scores > shared_high)))
    extreme = float(
        np.mean((scores <= POSITIVITY_THRESHOLD) | (scores >= 1 - POSITIVITY_THRESHOLD))
    )
    worst_smd = float(np.nanmax(np.abs(smd_after)))

    positivity_status: AssumptionStatus
    if extreme > 0.10 or outside > 0.10:
        positivity_status = "FAIL"
    elif extreme > 0.01 or outside > 0.02:
        positivity_status = "WARN"
    else:
        positivity_status = "PASS"

    balance_status: AssumptionStatus = "PASS" if worst_smd < BALANCE_THRESHOLD else "WARN"

    return (
        AssumptionCheck(
            name="positivity",
            kind="identification",
            status=positivity_status,
            statistic=extreme,
            threshold=0.01,
            interpretation=(
                f"{extreme:.1%} of units have a propensity score within "
                f"{POSITIVITY_THRESHOLD} of 0 or 1, and {outside:.1%} lie outside the "
                "region where both arms are represented. Such units have almost no "
                "counterpart in the other arm, so their contribution rests on "
                "extrapolation rather than comparison."
            ),
            remedy=(
                "Trim to the common support, or restrict the population to "
                "where both arms exist."
            ),
        ),
        AssumptionCheck(
            name="covariate_balance",
            kind="data_adequacy",
            status=balance_status,
            statistic=worst_smd,
            threshold=BALANCE_THRESHOLD,
            interpretation=(
                f"Largest standardised mean difference after weighting is "
                f"{worst_smd:.3f}; below {BALANCE_THRESHOLD} is conventionally read "
                "as adequate balance on the covariates supplied."
            ),
            remedy="Enrich the propensity model, or match rather than weight.",
        ),
        AssumptionCheck(
            name="unconfoundedness",
            kind="identification",
            status="NOT_TESTABLE",
            statistic=None,
            threshold=None,
            interpretation=(
                "Balance on the covariates supplied says nothing about covariates "
                "not supplied. No diagnostic computed from this data can establish "
                "that treatment assignment is independent of the potential outcomes "
                "given these covariates."
            ),
            remedy=(
                "Assess sensitivity to unobserved confounding, and justify "
                "the covariate set from domain knowledge."
            ),
        ),
    )
