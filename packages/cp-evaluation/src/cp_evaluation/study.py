"""The benchmark study's tables: the baseline slice, pooled summaries, and coverage.

Everything here is a function of a finished run's rows, and nothing here fits
anything. The tables are `metrics.summarise` over the rows, grouped more coarsely
than a cell so that each figure and its MCSE come from every replication of a
scenario's baseline cells at once.

**The baseline slice.** A scenario's grid holds its baseline cells plus
one-at-a-time cells that move a single axis. Pooling across those would mix
different truths -- halving `effect_scale` halves the effect being estimated --
and `sign_and_magnitude` and `detection_metrics` decide null versus non-null from
the group's mean truth. So the headline tables pool over the baseline slice only
(every axis at `AXIS_BASELINES`, `n_units` free), where a scenario's data-generating
process is one process. What the one-at-a-time cells show is per-cell, in the
per-cell summary and the axis-profile figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cp_evaluation.grid import AXIS_BASELINES
from cp_evaluation.metrics import summarise
from cp_evaluation.store import RunStatus

STATUS_COLUMNS: tuple[str, ...] = ("n_fitted", "n_ineligible", "n_refused")

# Each headline table's metrics, in the parent design's section 8.2 order.
TABLE_METRICS: dict[str, tuple[str, ...]] = {
    "accuracy": ("bias", "relative_bias", "rmse", "mae"),
    "coverage": (
        "coverage_analytic",
        "coverage_bootstrap",
        "coverage_posterior",
        "coverage_permutation",
        "coverage_bootstrap_se",
        "mean_ci_width",
    ),
    "se_calibration": ("se_calibration",),
    "sign_and_magnitude": ("type_s", "type_m"),
    "detection": ("power", "false_positive_rate"),
    "runtime": ("runtime_seconds",),
}

# The two levels every table is reported at.
LEVELS: dict[str, tuple[str, ...]] = {
    "scenario": ("scenario",),
    "scenario_by_size": ("scenario", "n_units", "n_periods"),
}


def baseline_slice(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows whose every axis sits at its baseline; `n_units` is left free.

    A panel row carries no cross-sectional axes (they are NaN), so it passes:
    the panel grid has nothing to vary but sample size and length.
    """
    keep = pd.Series(True, index=frame.index)
    for axis, baseline in AXIS_BASELINES.items():
        if axis in frame.columns:
            column = frame[axis]
            keep &= column.isna() | (column == baseline)
    return frame[keep]


def headline_tables(frame: pd.DataFrame, *, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Every headline table, named `<metric family>__<level>`, pooled over the baseline slice."""
    baseline = baseline_slice(frame)
    tables: dict[str, pd.DataFrame] = {}
    for level, by in LEVELS.items():
        summary = summarise(baseline, by=list(by), seed=seed)
        identity = ["method", *by]
        for family, metrics in TABLE_METRICS.items():
            columns = [
                *identity,
                *STATUS_COLUMNS,
                *(f"{m}{suffix}" for m in metrics for suffix in ("", "_mcse", "_n_used")),
            ]
            tables[f"{family}__{level}"] = summary[columns].reset_index(drop=True)
    detection = tables["detection__scenario"]
    tables["false_positive_rate__scenario"] = detection[
        detection["false_positive_rate_n_used"] > 0
    ].reset_index(drop=True)
    return tables


def format_value(value: object) -> str:
    if isinstance(value, float | np.floating):
        return "" if not np.isfinite(value) else f"{float(value):.4g}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    """A GitHub-flavoured table, with no dependency beyond pandas."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(format_value(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body]) + "\n"


# What the study did not run, and why. The parent design's grid has arms the
# shipped `full` tier leaves out; the study reports them by name rather than
# letting a reader assume they were covered.
NOT_COVERED: tuple[tuple[str, str], ...] = (
    (
        "CATE metrics",
        "the heterogeneity metrics (CATE correlation, RMSE, calibration slope) have no arm yet",
    ),
    ("scenario P7", "it needs n_periods >= 16 and cannot share a panel axis with P1 to P6"),
)


def coverage_of_study_markdown(
    frame: pd.DataFrame, status: RunStatus, replications: int
) -> str:
    """What the study covered and did not, so a reader cannot assume the difference."""
    mcse = float(np.sqrt(0.95 * 0.05 / replications))
    statuses = (
        frame.groupby(["method", "status"]).size().unstack(fill_value=0).reset_index()
        if len(frame)
        else pd.DataFrame({"method": []})
    )
    lines = [
        "# Coverage of the study",
        "",
        f"- Replications per configuration: {replications}. At a coverage of 0.95 the "
        f"binomial MCSE of one configuration's coverage is {mcse:.3f}.",
        f"- Cells: {status.done} of {status.expected} done.",
        f"- Scenarios run: {', '.join(sorted(frame['scenario'].unique()))}.",
        f"- Estimators run: {', '.join(sorted(frame['method'].unique()))}.",
        "",
        "## Not covered",
        "",
        *(f"- **{name}**: {reason}." for name, reason in NOT_COVERED),
        "",
        "## Rows by status, per method",
        "",
        markdown_table(statuses),
    ]
    return "\n".join(lines)
