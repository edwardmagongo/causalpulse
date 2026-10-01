"""What a benchmark run becomes in MLflow. Pure: no MLflow import here.

One MLflow run per benchmark run. The params are the run's identity, so two
runs of the same grid can be told apart by what was varied. The metrics are
the headline slice only, per scenario, plus a run's CATE metrics when it has
scored CATE rows: every grid cell would be tens of thousands of keys and would
make MLflow's comparison view unusable. The full headline tables go in as
artifacts, so nothing is lost.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from cp_evaluation.cate import CATE_METRIC_COLUMNS
from cp_evaluation.metrics import summarise
from cp_evaluation.study import headline_tables

FINGERPRINT_TAG = "cp.fingerprint"
# family of headline table -> the metrics read from it
_FAMILIES: dict[str, tuple[str, ...]] = {
    "accuracy": ("bias", "rmse"),
    "coverage": ("coverage_analytic", "coverage_bootstrap"),
    "se_calibration": ("se_calibration",),
}
TRACKED_METRICS: tuple[str, ...] = tuple(m for ms in _FAMILIES.values() for m in ms)
# Per-unit CATE recovery, logged per scenario and method under the same rule
# as the ATE families. Read from `summarise` over a run's CATE rows: exactly
# the numbers the benchmark report's CATE section shows.
CATE_TRACKED_METRICS: tuple[str, ...] = CATE_METRIC_COLUMNS
_MAX_PARAM = 6000  # MLflow's limit on a param value
_TRUNCATED = "[truncated; see identity.json]"


@dataclass(frozen=True)
class TrackingPayload:
    run_name: str
    params: dict[str, str]
    tags: dict[str, str]
    metrics: dict[str, float]
    artifacts: dict[str, str]  # file name -> text content


def fingerprint(identity: Mapping[str, Any], n_rows: int, git_commit: str | None) -> str:
    """The same finished run, at the same commit, always fingerprints the same.

    A different commit, identity or row count gives a different one. The row
    values themselves are not hashed, so re-running the identical grid at the
    same commit counts as the same run.
    """
    canonical = json.dumps(
        {"identity": identity, "n_rows": n_rows, "git_commit": git_commit}, sort_keys=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _bound(text: str) -> str:
    """Every param value goes through here, so none can exceed MLflow's limit."""
    if len(text) > _MAX_PARAM:
        # Never dropped silently: the full value is in the identity artifact.
        text = text[: _MAX_PARAM - len(_TRUNCATED)] + _TRUNCATED
    return text


def _param(value: object) -> str:
    return _bound(json.dumps(value, sort_keys=True))


def _params(manifest: Mapping[str, Any]) -> dict[str, str]:
    identity = manifest["identity"]
    params = {
        "tier": _bound(str(identity["tier"])),
        "estimators": _bound(",".join(identity["estimators"])),
        "root_entropy": _bound(str(identity["root_entropy"])),
        "replications": _bound(str(manifest["replications"])),
    }
    for field, value in identity["fit_config"].items():
        params[f"fit.{field}"] = _param(value)
    for arm, axes in identity["grid"].items():
        if axes is None:
            continue
        for axis, value in axes.items():
            params[f"grid.{arm}.{axis}"] = _param(value)
    return params


def _tags(manifest: Mapping[str, Any], run_dir: Path, n_rows: int) -> dict[str, str]:
    provenance = manifest["provenance"]
    tags = {
        "cp.git_commit": provenance.get("git_commit") or "unknown",
        "cp.run_dir": str(run_dir.resolve()),
        FINGERPRINT_TAG: fingerprint(manifest["identity"], n_rows, provenance.get("git_commit")),
    }
    for name, version in provenance.get("package_versions", {}).items():
        tags[f"cp.package.{name}"] = str(version)
    return tags


def _metrics(tables: Mapping[str, pd.DataFrame]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for family, names in _FAMILIES.items():
        table = tables[f"{family}__scenario"]
        for record in table.to_dict("records"):
            for name in names:
                value, mcse = record[name], record[f"{name}_mcse"]
                # A pair is logged whole or not at all: a value with no MCSE
                # could not be compared honestly across runs.
                if not (math.isfinite(value) and math.isfinite(mcse)):
                    continue
                key = f"{record['scenario']}.{record['method']}.{name}"
                metrics[key] = float(value)
                metrics[f"{key}_mcse"] = float(mcse)
    return metrics


def _cate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per method and scenario with a scored CATE metric; empty if none."""
    rows = frame[frame["shape"] == "cate"]
    if rows.empty:
        return pd.DataFrame()
    summary = summarise(rows, by=["scenario"])
    counts = [f"{metric}_n_used" for metric in CATE_TRACKED_METRICS]
    scored = summary[summary[counts].sum(axis=1) > 0]
    columns = [
        "method",
        "scenario",
        *(f"{m}{suffix}" for m in CATE_TRACKED_METRICS for suffix in ("", "_mcse", "_n_used")),
    ]
    return scored[columns].reset_index(drop=True)


def _cate_metrics(summary: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for record in summary.to_dict("records"):
        for name in CATE_TRACKED_METRICS:
            value, mcse = record[name], record[f"{name}_mcse"]
            # Same rule as `_metrics`: a value is logged with its MCSE or not at all,
            # so NULL's undefined correlation and slope never become NaN metrics.
            if not (math.isfinite(value) and math.isfinite(mcse)):
                continue
            key = f"{record['scenario']}.{record['method']}.{name}"
            metrics[key] = float(value)
            metrics[f"{key}_mcse"] = float(mcse)
    return metrics


def build_payload(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    run_dir: Path,
) -> TrackingPayload:
    tables = headline_tables(frame)
    artifacts = {f"{name}.csv": table.to_csv(index=False) for name, table in tables.items()}
    artifacts["identity.json"] = json.dumps(manifest["identity"], indent=2, sort_keys=True)
    artifacts["sidecar.json"] = json.dumps(dict(sidecar), indent=2, sort_keys=True, default=str)
    cate = _cate_summary(frame)
    if not cate.empty:
        artifacts["cate__scenario.csv"] = cate.to_csv(index=False)
    return TrackingPayload(
        run_name=run_dir.name,
        params=_params(manifest),
        tags=_tags(manifest, run_dir, len(frame)),
        # Cannot overwrite: ATE keys end in TRACKED_METRICS names and CATE keys in
        # `cate_*` names, which are disjoint.
        metrics={**_metrics(tables), **_cate_metrics(cate)},
        artifacts=artifacts,
    )
