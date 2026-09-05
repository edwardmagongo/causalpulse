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


def _share_with_mcse(hits: np.ndarray) -> MetricValue:
    """A binomial share, with an MCSE that stays honest at the endpoints.

    At a unanimous 0-of-n or n-of-n share the binomial standard error
    `sqrt(p(1-p)/n)` is exactly 0.0, which reads as infinite precision from a
    handful of replications. The rule of three (3/n) is the standard upper
    bound on the unobserved tail and is used instead.
    """
    n = int(hits.size)
    if n == 0:
        return _EMPTY
    share = float(np.mean(hits))
    if share in (0.0, 1.0):
        return MetricValue(value=share, mcse=3.0 / n, n_used=n)
    return MetricValue(
        value=share, mcse=float(np.sqrt(share * (1.0 - share) / n)), n_used=n
    )


def _interval_hits(
    rows: pd.DataFrame, lower_column: str, upper_column: str
) -> np.ndarray:
    lower = rows[lower_column].to_numpy(dtype=float)
    upper = rows[upper_column].to_numpy(dtype=float)
    truth = rows["truth"].to_numpy(dtype=float)
    usable = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(truth)
    hits: np.ndarray = (
        (lower[usable] <= truth[usable]) & (truth[usable] <= upper[usable])
    ).astype(float)
    return hits


def coverage_metrics(frame: pd.DataFrame) -> dict[str, MetricValue]:
    """Coverage for the analytic and bootstrap intervals, and mean CI width.

    Reported separately rather than combined, because the parent spec's
    headline question is precisely where the two diverge -- where asymptotic
    standard errors break down and the bootstrap does not, or the reverse.
    """
    rows = _fitted(frame)

    widths = (rows["ci_upper"] - rows["ci_lower"]).to_numpy(dtype=float)
    widths = widths[np.isfinite(widths)]

    return {
        "coverage_analytic": _share_with_mcse(
            _interval_hits(rows, "ci_lower", "ci_upper")
        ),
        "coverage_bootstrap": _share_with_mcse(
            _interval_hits(rows, "bootstrap_ci_lower", "bootstrap_ci_upper")
        ),
        "mean_ci_width": _mean_with_mcse(widths),
    }


def se_calibration(
    frame: pd.DataFrame, *, n_bootstrap: int = 1000, seed: int = 0
) -> MetricValue:
    """Mean reported standard error divided by the empirical SD of estimates.

    A ratio near 1 means the reported standard errors describe the actual
    spread of the estimator; below 1 they are dishonest, and the interval
    built from them will under-cover. Estimators reporting no standard error
    at all (permutation and posterior inference) are excluded rather than
    counted as zero -- `n_used` says how many rows survived.

    The MCSE is bootstrapped rather than closed-form. The ratio's
    denominator, the empirical SD, is itself a random quantity estimated
    from these same rows, and in practice it carries most of the ratio's
    sampling variability -- a closed-form MCSE that propagates only the
    numerator's uncertainty (`std(se, ddof=1) / sqrt(n) / empirical_sd`)
    treats that denominator as fixed. For an estimator reporting a
    *constant* `se` across replications -- common, and entirely legitimate
    -- that closed-form MCSE comes out to exactly 0.0 regardless of how much
    the point estimates themselves vary, which is a metric reading as
    infinite precision when it has none. Resampling rows with replacement
    and recomputing the ratio on each resample makes no distributional
    assumption and reflects the variability of both numerator and
    denominator together.

    Resamples whose resampled points are all identical give an empirical SD
    of exactly 0.0 and are dropped rather than divided by; if fewer than two
    resamples remain usable, the MCSE is nan rather than a number fabricated
    from too little information.
    """
    rows = _fitted(frame)
    se = rows["se"].to_numpy(dtype=float)
    point = rows["point"].to_numpy(dtype=float)
    usable = np.isfinite(se) & np.isfinite(point)
    se, point = se[usable], point[usable]

    n = int(se.size)
    if n < 2:
        return MetricValue(value=float("nan"), mcse=float("nan"), n_used=n)

    empirical_sd = float(np.std(point, ddof=1))
    if empirical_sd == 0.0:
        # Every replication produced the same estimate. The ratio is
        # undefined rather than infinite; a degenerate estimator is a finding
        # for the accuracy metrics to report, not a division to force
        # through here.
        return MetricValue(value=float("nan"), mcse=float("nan"), n_used=n)

    ratio = float(np.mean(se)) / empirical_sd

    rng = np.random.default_rng(np.random.SeedSequence(seed))
    resample_indices = rng.integers(0, n, size=(n_bootstrap, n))
    se_resamples = se[resample_indices]
    point_resamples = point[resample_indices]
    resample_sds = np.std(point_resamples, ddof=1, axis=1)
    resample_valid = resample_sds > 0.0
    if int(np.count_nonzero(resample_valid)) < 2:
        mcse = float("nan")
    else:
        resample_ratios = (
            np.mean(se_resamples[resample_valid], axis=1)
            / resample_sds[resample_valid]
        )
        mcse = float(np.std(resample_ratios, ddof=1))
    return MetricValue(value=ratio, mcse=mcse, n_used=n)
