"""Resampling interval estimates.

Three resampling schemes, because the unit of independence differs by design:
i.i.d. rows for cross-sections, whole clusters when errors correlate within
groups, and contiguous blocks when observations are serially dependent.
Resampling the wrong unit produces intervals that are confidently too narrow.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from cp_core.values import Interval

BootstrapMethod = Literal["percentile", "basic"]
_VALID_KINDS = frozenset({"iid", "cluster", "block"})
_VALID_METHODS = frozenset({"percentile", "basic"})


@dataclass(frozen=True)
class BootstrapResult:
    interval: Interval
    replicates: np.ndarray
    standard_error: float
    point_estimate: float


def _resample_iid(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(0, data.shape[0], size=data.shape[0])
    return data[indices]


def _resample_cluster(
    data: np.ndarray, clusters: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    labels = np.unique(clusters)
    drawn = rng.integers(0, labels.size, size=labels.size)
    pieces = [data[clusters == labels[i]] for i in drawn]
    return np.concatenate(pieces)


def _resample_block(
    data: np.ndarray, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    n = data.shape[0]
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
    pieces = [data[s : s + block_size] for s in starts]
    return np.concatenate(pieces)[:n]


def bootstrap_interval(
    statistic: Callable[[np.ndarray], float],
    data: np.ndarray,
    *,
    reps: int,
    rng: np.random.Generator,
    level: float = 0.95,
    kind: str = "iid",
    clusters: np.ndarray | None = None,
    block_size: int | None = None,
    method: BootstrapMethod = "percentile",
) -> BootstrapResult:
    """Bootstrap a statistic and return an interval, replicates and standard error."""
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}")
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {sorted(_VALID_METHODS)}, got {method!r}")
    if kind == "cluster" and clusters is None:
        raise ValueError("kind='cluster' requires clusters to be provided")
    if kind == "block" and block_size is None:
        raise ValueError("kind='block' requires block_size to be provided")
    if reps < 1:
        raise ValueError(f"reps must be positive, got {reps}")

    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    if clusters is not None:
        clusters = np.asarray(clusters)
        if clusters.shape[0] != data.shape[0]:
            raise ValueError(
                f"clusters has length {clusters.shape[0]} but data has {data.shape[0]}"
            )
    if kind == "block" and block_size is not None:
        if block_size < 1:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if block_size > n:
            raise ValueError(f"block_size ({block_size}) cannot exceed data length ({n})")

    replicates = np.empty(reps, dtype=float)
    for i in range(reps):
        if kind == "iid":
            sample = _resample_iid(data, rng)
        elif kind == "cluster":
            assert clusters is not None
            sample = _resample_cluster(data, clusters, rng)
        else:
            assert block_size is not None
            sample = _resample_block(data, block_size, rng)
        replicates[i] = float(statistic(sample))

    point = float(statistic(data))
    alpha = 1.0 - level
    lower_q, upper_q = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0])

    if method == "percentile":
        lower, upper = float(lower_q), float(upper_q)
    else:
        # Basic (reverse percentile) interval reflects the replicate distribution
        # around the point estimate rather than reading quantiles directly.
        lower, upper = float(2 * point - upper_q), float(2 * point - lower_q)

    return BootstrapResult(
        interval=Interval(lower=lower, upper=upper, level=level),
        replicates=replicates,
        standard_error=float(replicates.std(ddof=1)),
        point_estimate=point,
    )
