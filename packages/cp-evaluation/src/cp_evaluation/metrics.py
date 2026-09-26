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

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from cp_simulation.params import AXIS_NAMES


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


def _basis(rows: pd.DataFrame, basis: str) -> pd.DataFrame:
    """Rows whose own `ci_basis` is `basis`: one basis-labelled coverage column's population.

    Filters on `ci_basis` alone. `summarise` groups by `method` first, so two
    estimators sharing a basis never share a group and a shape gate adds nothing for
    per-cell groups. A caller pooling with `by=[]` across shapes must scope rows itself, as E5 must.
    """
    return rows[rows["ci_basis"] == basis]


def coverage_metrics(frame: pd.DataFrame) -> dict[str, MetricValue]:
    """Coverage of each kind of interval an estimator reports, and mean CI width.

    `ci_lower`/`ci_upper` is the estimator's primary interval and its meaning
    depends on `ci_basis`, so its coverage is reported per basis rather than
    under one name: `coverage_analytic` is analytic intervals only, and
    posterior, permutation and bootstrap-SE intervals each have their own
    column. `coverage_bootstrap` is a different quantity: coverage of the
    separately reported `bootstrap_ci` (a percentile bootstrap), whatever the
    primary interval's basis. `mean_ci_width` pools every basis.
    """
    rows = _fitted(frame)

    widths = (rows["ci_upper"] - rows["ci_lower"]).to_numpy(dtype=float)
    widths = widths[np.isfinite(widths)]

    def primary(basis: str) -> MetricValue:
        return _share_with_mcse(_interval_hits(_basis(rows, basis), "ci_lower", "ci_upper"))

    return {
        "coverage_analytic": primary("analytic"),
        "coverage_bootstrap": _share_with_mcse(
            _interval_hits(rows, "bootstrap_ci_lower", "bootstrap_ci_upper")
        ),
        "coverage_posterior": primary("posterior"),
        "coverage_permutation": primary("permutation"),
        "coverage_bootstrap_se": primary("bootstrap"),
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


def _significant(rows: pd.DataFrame) -> np.ndarray:
    """Whether each interval excludes zero.

    Read off the interval rather than the p-value, matching
    `EffectEstimate.is_significant`, so estimators whose inference is
    permutation- or posterior-based and carry no p-value are still counted.
    """
    lower = rows["ci_lower"].to_numpy(dtype=float)
    upper = rows["ci_upper"].to_numpy(dtype=float)
    usable = np.isfinite(lower) & np.isfinite(upper)
    excludes = ~((lower <= 0.0) & (upper >= 0.0))
    return (usable & excludes).astype(float)[usable]


def sign_and_magnitude(frame: pd.DataFrame) -> dict[str, MetricValue]:
    """Type S (sign error) rate and Type M (exaggeration) ratio.

    Both are conditional on significance, which is what makes them the
    interesting pair in small-sample cells: an estimator that only reaches
    significance when it has overshot will show a Type M well above 1.

    Both are undefined when the truth is zero. There is no sign to get wrong
    and no magnitude to exaggerate, and `|point| / 0` would be infinite rather
    than meaningful -- so the null cells report nan here and their false
    positive rate in `detection_metrics` instead.
    """
    rows = _fitted(frame)
    lower = rows["ci_lower"].to_numpy(dtype=float)
    upper = rows["ci_upper"].to_numpy(dtype=float)
    point = rows["point"].to_numpy(dtype=float)
    truth = rows["truth"].to_numpy(dtype=float)

    usable = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(point)
    significant = usable & ~((lower <= 0.0) & (upper >= 0.0))

    finite_truth = truth[np.isfinite(truth)]
    mean_truth = float(np.mean(finite_truth)) if finite_truth.size else float("nan")
    if not np.isfinite(mean_truth) or mean_truth == 0.0:
        return {"type_s": _EMPTY, "type_m": _EMPTY}

    chosen = point[significant]
    if chosen.size == 0:
        return {"type_s": _EMPTY, "type_m": _EMPTY}

    wrong_sign = (np.sign(chosen) != np.sign(mean_truth)).astype(float)
    type_s = _share_with_mcse(wrong_sign)
    type_m = _mean_with_mcse(np.abs(chosen) / abs(mean_truth))
    return {"type_s": type_s, "type_m": type_m}


def detection_metrics(frame: pd.DataFrame) -> dict[str, MetricValue]:
    """Power in non-null cells, false-positive rate in null ones.

    The same underlying share, split by whether the truth is zero, because the
    two answer different questions and only one of them is meaningful per cell.
    The design document calls the false-positive rate under the null scenarios
    the falsifiability test of the whole system: it must come out near 0.05.
    """
    rows = _fitted(frame)
    hits = _significant(rows)

    truth = rows["truth"].to_numpy(dtype=float)
    finite_truth = truth[np.isfinite(truth)]
    mean_truth = float(np.mean(finite_truth)) if finite_truth.size else float("nan")
    is_null = np.isfinite(mean_truth) and mean_truth == 0.0

    share = _share_with_mcse(hits)
    if is_null:
        return {"power": _EMPTY, "false_positive_rate": share}
    return {"power": share, "false_positive_rate": _EMPTY}


# A cell's identity, which is what a metric is computed *within*. Every axis
# the grid can vary has to appear here: two cells that differ on an axis are
# different cells, and pooling them averages the axis away.
#
# The axis names come from `AXIS_NAMES` rather than being restated, so an axis
# added to `ScenarioParams` cannot arrive in the results frame (via
# `run.ROW_COLUMNS`) while silently staying absent from the grouping -- which
# is how Phase 18's four new axes would otherwise have been averaged into
# their own baseline. That failure is invisible in the output: the summary
# still has one row per (method, cell), it just quietly describes a pool of
# cells that are not the same cell. `confounding_strength` leads `AXIS_NAMES`,
# so this keeps the exact column order the tuple had before those axes
# existed.
#
# `realised_treated_share` is deliberately NOT here. It is measured from each
# cell's own generated data (`run._realised_treated_share`) and so differs
# between replications of the *same* cell; grouping on it would put every
# replication in a group of its own and leave every metric with n_used=1.
GROUP_COLUMNS = ("shape", "scenario", "n_units", "n_periods", *AXIS_NAMES)


def _seed_component(value: object) -> object:
    """One group-key value, made JSON-total for hashing.

    Group keys legitimately contain `float('nan')` -- the grid puts it in
    `n_periods` for cross-sectional cells and in `confounding_strength` for
    panel cells, since those axes don't apply to that shape. `json.dumps`
    would otherwise fall back to emitting the bare token `NaN`, which is not
    valid JSON text, and every NaN would collapse onto that one token
    regardless of *which* column it came from. Swapping it for a sentinel
    tagged with its own type keeps the JSON strict and keeps a NaN in one
    column from hashing identically to a NaN in another, or to a real string
    that happened to read "nan".
    """
    if isinstance(value, float) and math.isnan(value):
        return "__nan__"
    return value


def _group_seed(root_seed: int, group_columns: list[str], key: tuple[object, ...]) -> int:
    """A per-group bootstrap seed, derived from the group's own identity.

    Without this, `se_calibration`'s default `seed=0` is shared by every
    group `summarise` computes in one call, so any two groups with the same
    number of fitted rows draw the literal same bootstrap resample indices --
    an artificial correlation between their MCSEs that has nothing to do with
    the data. Deriving the seed from the group's own key gives each group an
    independent stream while keeping the same group's seed stable run to run.

    This follows `cp_evaluation.grid.cell_rng`'s precedent for the same
    problem: sha256 over sorted JSON, not `hash()`, because `hash()` is
    salted per process and anything built from it fails to reproduce across
    runs (see `cp_simulation._hashing`). Deriving from identity rather than
    position in the table means adding or removing a method or cell leaves
    every *other* group's seed untouched -- but only once `group_columns`
    itself is stable. `group_columns` is not part of a group's identity, and
    widening it (as Phase 18 did, adding the four new axis columns to every
    group's key) changes what gets hashed for every group at once, even
    groups whose own data and axis values did not change. That one-time
    reseed is recorded in
    `docs/specs/2026-09-13-causal-core-phase18-design.md` section 12.
    """
    payload: dict[str, object] = {"root_seed": root_seed}
    payload.update(
        {
            column: _seed_component(value)
            for column, value in zip(group_columns, key, strict=True)
        }
    )
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False, default=str).encode()
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], "big")


def summarise(
    frame: pd.DataFrame, *, seed: int = 0, by: Sequence[str] | None = None
) -> pd.DataFrame:
    """Every metric family, per (method x cell identity), one row each.

    `by` replaces the cell-identity columns the rows are grouped by. Grouping by
    fewer columns pools the replication rows of every cell that shares the
    remaining ones, and each metric and its MCSE is computed over those pooled
    rows -- never by combining per-cell summaries, which would average averages
    of unequal groups and could not give the pooled MCSE at all. `by=[]` pools
    across everything a method fitted.

    Status counts travel with the metrics rather than being computed
    separately, so a reader can never see a bias without also seeing how many
    of that group's fits were refused or ineligible.

    `seed` roots `se_calibration`'s per-group bootstrap seeds (see
    `_group_seed`); it does not otherwise affect any metric.

    The output also carries `realised_treated_share` (plus its `_mcse` and
    `_n_used`), the per-group mean of `run._realised_treated_share` (see
    `docs/specs/2026-09-13-causal-core-phase18-design.md` section 12,
    "Findings from the whole-branch review"): `treatment_intercept` is a
    logit-scale parameter, not a quantity a reader of a summary table thinks
    in, and before this the
    measured share -- spec section 4.1's mitigation for exactly that -- never
    reached `summarise`'s output at all, so a `bench report` table showed the
    intercept axis with no share to read it against. It is computed over
    every row in the group, not just `_fitted` ones: it is a property of the
    cell's generated data, not of whether an estimator's fit succeeded on it,
    and every row sharing a cell (fitted, ineligible or refused alike) has
    the identical measured value. NaN for a panel group, which carries no
    such axis.
    """
    if by is None:
        group_columns = ["method", *(c for c in GROUP_COLUMNS if c in frame.columns)]
    else:
        absent = [c for c in by if c not in frame.columns]
        if absent:
            raise KeyError(f"cannot group by {absent}: not columns of this frame")
        group_columns = ["method", *by]
    records: list[dict[str, object]] = []

    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        record: dict[str, object] = dict(zip(group_columns, key_tuple, strict=True))
        record["n_fitted"] = int((group["status"] == "fitted").sum())
        record["n_ineligible"] = int((group["status"] == "ineligible").sum())
        record["n_refused"] = int((group["status"] == "refused").sum())

        fitted_runtime = _fitted(group)["runtime_seconds"].to_numpy(dtype=float)
        if "realised_treated_share" in group.columns:
            share = group["realised_treated_share"].to_numpy(dtype=float)
            share_metric = _mean_with_mcse(share[np.isfinite(share)])
        else:
            share_metric = _EMPTY
        families: dict[str, MetricValue] = {
            **accuracy_metrics(group),
            **coverage_metrics(group),
            **sign_and_magnitude(group),
            **detection_metrics(group),
            "se_calibration": se_calibration(
                group, seed=_group_seed(seed, group_columns, key_tuple)
            ),
            "runtime_seconds": _mean_with_mcse(
                fitted_runtime[np.isfinite(fitted_runtime)]
            ),
            "realised_treated_share": share_metric,
        }
        for name, metric in families.items():
            record[name] = metric.value
            record[f"{name}_mcse"] = metric.mcse
            record[f"{name}_n_used"] = metric.n_used

        records.append(record)

    return pd.DataFrame(records)
