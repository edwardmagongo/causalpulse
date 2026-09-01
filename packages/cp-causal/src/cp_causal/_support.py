"""Helpers shared by every estimator in this package."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.values import Diagnostic, EstimatorConfig, Interval
from cp_statistics.bootstrap import BootstrapResult, bootstrap_interval
from scipy import stats


def require_no_missing(dataset: CausalDataset) -> None:
    """Refuse a dataset carrying missing values in any required column.

    No estimator here handles a missing value, and before this guard each one
    failed differently and late -- three of them only because a downstream
    interval or integer conversion happened to notice a `nan`, and none of
    them naming the column at fault.

    Dropping the incomplete rows is a modelling choice rather than a repair,
    so it is not done here on the caller's behalf. The message names
    `complete_cases` as the way to make that choice, not as the way to make
    an error go away: complete-case analysis is unbiased when values are
    missing completely at random, and biased in general when they are not.
    """
    frame = dataset.frame
    offending = tuple(
        column
        for column in dataset.schema.required_columns
        if frame[column].isna().any()
    )
    if offending:
        raise ValueError(
            f"columns {sorted(offending)} contain missing values, and no "
            "estimator here handles them. Reduce to complete cases with "
            "cp_core.dataset.complete_cases if that assumption suits the "
            "analysis, or supply data without gaps."
        )


def arrays(dataset: CausalDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract `(covariates, treatment, outcome)` as float arrays."""
    require_no_missing(dataset)
    frame = dataset.frame
    schema = dataset.schema
    covariates = (
        frame[list(schema.covariates)].to_numpy(dtype=float)
        if schema.covariates
        else np.empty((len(frame), 0))
    )
    treatment = frame[schema.treatment].to_numpy(dtype=float)
    outcome = frame[schema.outcome].to_numpy(dtype=float)
    return covariates, treatment, outcome


def config_hash(
    config: EstimatorConfig, *, extra: Mapping[str, object] | None = None
) -> str:
    """Stable hash of a configuration, for reproducibility metadata.

    Uses sha256 over sorted JSON rather than `hash()`, which is salted per
    process and would make the recorded value meaningless across runs.

    `extra` carries settings that change the result but do not live on
    `EstimatorConfig` -- `multiple_impute`'s `m` and `seed` are the case this
    exists for. Without it a caller with such settings fingerprints two
    materially different estimates identically, which is worse than no
    fingerprint at all.
    """
    payload = {
        "bootstrap_reps": config.bootstrap_reps,
        "bootstrap_kind": config.bootstrap_kind,
        "ci_level": config.ci_level,
        "random_seed": config.random_seed,
        "options": {k: repr(v) for k, v in sorted(config.options.items())},
        "extra": {} if extra is None else {k: repr(v) for k, v in sorted(extra.items())},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def bootstrap_effect(
    point_fn: Callable[[np.ndarray], float],
    n: int,
    config: EstimatorConfig,
) -> BootstrapResult:
    """Bootstrap an estimator by resampling row indices.

    `bootstrap_interval` resamples a 1-D array and casts it to float, so the
    indices arrive back as floats and `point_fn` must cast them to int. Doing
    that conversion here keeps every estimator's own code free of the detail.

    Returns the full `BootstrapResult`, not just its `interval`, so that
    `n_degenerate` (how many resamples happened to contain only one treatment
    arm and were dropped) reaches the caller. Every estimator that bootstraps
    through this helper must surface that count as a diagnostic rather than
    silently discard it.
    """
    seed = 0 if config.random_seed is None else config.random_seed
    # Derived through a SeedSequence rather than seeding a generator directly, so
    # this shared helper sits on the same root-seed chain as the rest of the
    # project and nothing downstream has to seed its own generator.
    rng = np.random.default_rng(np.random.SeedSequence(seed))

    def statistic(sample: np.ndarray) -> float:
        return point_fn(sample.astype(np.int64))

    return bootstrap_interval(
        statistic,
        np.arange(n, dtype=float),
        reps=config.bootstrap_reps,
        rng=rng,
        level=config.ci_level,
        kind=config.bootstrap_kind,
    )


def degeneracy_diagnostic(n_degenerate: int, min_arm_size: int) -> Diagnostic:
    """Build the `bootstrap_degenerate_replicates` Diagnostic.

    Every estimator's `point_from` closure discards a bootstrap resample when
    one treatment arm falls below its own `min_arm_size` threshold — that
    threshold differs by estimator (a bare non-empty-arm check for
    `DifferenceInMeans`, up to 25 units for `AugmentedIPW`). Building the
    interpretation text here from the caller's actual threshold, rather than
    copy-pasting a fixed sentence into every estimator, is what keeps the text
    accurate as thresholds change or new estimators are added.
    """
    return Diagnostic(
        name="bootstrap_degenerate_replicates",
        value=float(n_degenerate),
        unit="replicates",
        interpretation=(
            "Count of bootstrap resamples that ended up with fewer than "
            f"{min_arm_size} units in one treatment arm and were excluded "
            "from the bootstrap interval. A high count means the interval "
            "rests on fewer effective replicates than requested."
        ),
    )


def critical_value(ci_level: float) -> float:
    """Two-sided normal critical value for a confidence level."""
    return float(stats.norm.ppf(1 - (1 - ci_level) / 2))


def wald_interval(point: float, se: float, ci_level: float) -> Interval:
    """Symmetric normal-approximation interval.

    Extracted from four estimators that each built it inline. A zero standard
    error yields a degenerate interval rather than raising: a perfectly
    separated arm can produce one, and a point interval is the honest report.
    """
    critical = critical_value(ci_level)
    return Interval(
        lower=point - critical * se, upper=point + critical * se, level=ci_level
    )


def normal_p_value(point: float, se: float) -> float:
    """Two-sided p-value against a null of zero, or 1.0 when se is zero."""
    if se <= 0:
        return 1.0
    return float(2 * (1 - stats.norm.cdf(abs(point / se))))


def coefficient_se(covariance: np.ndarray, index: int) -> float:
    """Standard error of one coefficient, read from its own diagonal entry.

    Deliberately not `np.sqrt(np.diag(covariance))`. On a two-way fixed-effects
    design the unit- and period-dummy coefficients have near-zero true
    variances, and the sandwich returns them as small negative numbers —
    measured at around -1.7e-27, on roughly one seed in thirty-six, on the
    cluster-robust path. A blanket square root turns that floating-point noise
    into `RuntimeWarning: invalid value encountered in sqrt`, which
    `filterwarnings = ["error"]` promotes to an intermittent failure with no
    obvious cause.

    A negative variance on the coefficient actually being reported is a
    different matter — that is a real defect, not noise — so it raises.
    """
    variance = float(covariance[index, index])
    if variance < 0:
        raise ValueError(
            f"negative variance {variance!r} for coefficient {index}; the "
            "covariance estimate is not usable for this coefficient"
        )
    return float(np.sqrt(variance))
