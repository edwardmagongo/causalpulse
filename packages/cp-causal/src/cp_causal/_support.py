"""Helpers shared by every estimator in this package."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.values import Diagnostic, EstimatorConfig
from cp_statistics.bootstrap import BootstrapResult, bootstrap_interval


def arrays(dataset: CausalDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract `(covariates, treatment, outcome)` as float arrays."""
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


def config_hash(config: EstimatorConfig) -> str:
    """Stable hash of a configuration, for reproducibility metadata.

    Uses sha256 over sorted JSON rather than `hash()`, which is salted per
    process and would make the recorded value meaningless across runs.
    """
    payload = {
        "bootstrap_reps": config.bootstrap_reps,
        "bootstrap_kind": config.bootstrap_kind,
        "ci_level": config.ci_level,
        "random_seed": config.random_seed,
        "options": {k: repr(v) for k, v in sorted(config.options.items())},
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
