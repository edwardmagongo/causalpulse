"""Metrics over a grid's rows, each with a Monte Carlo standard error.

The parent spec's §8.2 is explicit about why the MCSE is not optional: a
coverage table without one cannot distinguish 94% from 95%, and a benchmark
that cannot do that is not a study.

Every metric also reports `n_used`. A grid row can be `ineligible` or
`refused` as well as `fitted`, and a metric computed over zero usable rows has
to be distinguishable from one that happens to equal zero -- otherwise those
two statuses quietly become a number nobody chose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricValue:
    """One metric, its Monte Carlo standard error, and how many rows fed it.

    `value` is nan when no rows were usable; `mcse` is nan when fewer than two
    were, since a standard error over a single observation is undefined rather
    than zero.
    """

    value: float
    mcse: float
    n_used: int


_EMPTY = MetricValue(value=float("nan"), mcse=float("nan"), n_used=0)


def _fitted(frame: pd.DataFrame) -> pd.DataFrame:
    """Only the rows that produced an estimate.

    `ineligible` and `refused` rows carry no point estimate by construction;
    they are kept in the frame so their absence is visible, and excluded here
    so they never enter a numerator.
    """
    return frame[frame["status"] == "fitted"]


def _mean_with_mcse(values: np.ndarray) -> MetricValue:
    n = int(values.size)
    if n == 0:
        return _EMPTY
    mean = float(np.mean(values))
    if n < 2:
        return MetricValue(value=mean, mcse=float("nan"), n_used=n)
    mcse = float(np.std(values, ddof=1) / np.sqrt(n))
    return MetricValue(value=mean, mcse=mcse, n_used=n)


def accuracy_metrics(frame: pd.DataFrame) -> dict[str, MetricValue]:
    """Bias, relative bias, RMSE and MAE over one group's fitted rows."""
    rows = _fitted(frame)
    errors = (rows["point"] - rows["truth"]).to_numpy(dtype=float)
    errors = errors[np.isfinite(errors)]

    bias = _mean_with_mcse(errors)
    mae = _mean_with_mcse(np.abs(errors))
    squared = _mean_with_mcse(errors**2)
    rmse = (
        _EMPTY
        if squared.n_used == 0
        else MetricValue(
            value=float(np.sqrt(squared.value)),
            # Delta method: sd(sqrt(X)) ~ sd(X) / (2 sqrt(E[X])). Undefined at
            # a mean squared error of exactly zero, which only a perfect fit
            # across every replication would produce.
            mcse=(
                float("nan")
                if not np.isfinite(squared.mcse) or squared.value <= 0.0
                else float(squared.mcse / (2.0 * np.sqrt(squared.value)))
            ),
            n_used=squared.n_used,
        )
    )

    truths = rows["truth"].to_numpy(dtype=float)
    truths = truths[np.isfinite(truths)]
    mean_truth = float(np.mean(truths)) if truths.size else float("nan")
    if bias.n_used == 0 or mean_truth == 0.0 or not np.isfinite(mean_truth):
        # Truth is exactly zero in the null scenarios (B and P2), which are
        # the cells the false-positive test exists for. `bias / 0` is nan by
        # explicit guard rather than inf by accident.
        relative = MetricValue(float("nan"), float("nan"), bias.n_used)
    else:
        relative = MetricValue(
            value=bias.value / mean_truth,
            mcse=(
                float("nan")
                if not np.isfinite(bias.mcse)
                else abs(bias.mcse / mean_truth)
            ),
            n_used=bias.n_used,
        )

    return {"bias": bias, "relative_bias": relative, "rmse": rmse, "mae": mae}
