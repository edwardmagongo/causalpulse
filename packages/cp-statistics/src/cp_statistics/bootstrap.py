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


# eq=False: this holds array-valued fields, so a generated __eq__ would raise
# on the ambiguous truth value of an array rather than compare. Identity
# equality is the honest semantics for a container of measurements.
@dataclass(frozen=True, eq=False)
class BootstrapResult:
    interval: Interval
    # Raw and unfiltered: this is every replicate produced, including any
    # non-finite ("degenerate") entries from resamples that happened to
    # contain only one treatment arm. `interval` and `standard_error` are
    # computed from the finite subset only — `n_degenerate` reports how many
    # entries here are non-finite. A consumer calling `np.mean`/`np.std`
    # directly on this array must filter with `np.isfinite` first, or a
    # single NaN silently propagates through the aggregate.
    replicates: np.ndarray
    standard_error: float
    point_estimate: float
    n_degenerate: int

    def __post_init__(self) -> None:
        # Mirrors the finiteness guard on `Interval`: a NaN here (e.g. from
        # `std(ddof=1)` on a single finite replicate) is the same failure
        # class one field over, and should fail loudly rather than ride
        # along as a silently-broken summary statistic.
        if not np.isfinite(self.standard_error):
            raise ValueError(f"standard_error must be finite, got {self.standard_error}")
        if not np.isfinite(self.point_estimate):
            raise ValueError(f"point_estimate must be finite, got {self.point_estimate}")


def _resample_iid(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(0, data.shape[0], size=data.shape[0])
    return np.asarray(data[indices])


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

    # A resample can land on a degenerate draw (e.g. missing a treatment arm
    # entirely), which by design makes `statistic` return `nan` rather than a
    # fabricated number. `np.quantile` propagates a single `nan` to every
    # requested quantile regardless of its position, so degenerate replicates
    # must be dropped before quantiles or the standard error are computed —
    # otherwise the interval silently becomes `Interval(nan, nan)`.
    finite_mask = np.isfinite(replicates)
    n_finite = int(finite_mask.sum())
    n_degenerate = reps - n_finite
    if n_finite < reps / 2.0:
        raise ValueError(
            f"{n_degenerate} of {reps} bootstrap replicates were non-finite "
            f"(degenerate resamples, e.g. missing a group entirely); only "
            f"{n_finite} finite replicates remain, which is fewer than half. "
            "The resample distribution is not trustworthy enough to report."
        )
    finite_replicates = replicates[finite_mask]

    point = float(statistic(data))
    alpha = 1.0 - level
    lower_q, upper_q = np.quantile(
        finite_replicates, [alpha / 2.0, 1.0 - alpha / 2.0]
    )

    if method == "percentile":
        lower, upper = float(lower_q), float(upper_q)
    else:
        # Basic (reverse percentile) interval reflects the replicate distribution
        # around the point estimate rather than reading quantiles directly.
        lower, upper = float(2 * point - upper_q), float(2 * point - lower_q)

    return BootstrapResult(
        interval=Interval(lower=lower, upper=upper, level=level),
        replicates=replicates,
        standard_error=float(finite_replicates.std(ddof=1)),
        point_estimate=point,
        n_degenerate=n_degenerate,
    )
