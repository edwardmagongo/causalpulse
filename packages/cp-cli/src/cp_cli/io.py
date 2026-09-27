"""Reading and writing what the commands produce.

The column vocabulary is deliberately shared with `cp_evaluation.run`'s
`ROW_COLUMNS` wherever the two describe the same quantity. Two later phases --
the figures and the evidence report -- read both an analysis and a benchmark,
and giving them one set of names rather than two is cheaper here than a
translation layer there.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from cp_core.results import EffectEstimate, estimate_summary

ESTIMATE_COLUMNS: tuple[str, ...] = (
    "method",
    "estimand",
    "point",
    "se",
    "ci_lower",
    "ci_upper",
    "ci_basis",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "p_value",
    "n_treated",
    "n_control",
    "runtime_seconds",
)

_SIDECAR_NAME = "run.json"


def estimates_frame(estimates: Sequence[EffectEstimate]) -> pd.DataFrame:
    rows = []
    for estimate in estimates:
        summary = estimate_summary(estimate)
        rows.append(
            {
                column: np.nan if summary[column] is None else summary[column]
                for column in ESTIMATE_COLUMNS
            }
        )
    return pd.DataFrame(rows, columns=list(ESTIMATE_COLUMNS))


def write_estimates(estimates: Sequence[EffectEstimate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    estimates_frame(estimates).to_csv(path, index=False)


def write_benchmark(
    frame: pd.DataFrame, directory: Path, sidecar: Mapping[str, object]
) -> None:
    """Write a tidy grid frame as Parquet partitioned by scenario, plus its sidecar."""
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        directory / "results", engine="pyarrow", partition_cols=["scenario"], index=False
    )
    (directory / _SIDECAR_NAME).write_text(json.dumps(dict(sidecar), sort_keys=True, indent=2))


def read_benchmark(directory: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read back what `write_benchmark` wrote.

    `partition_cols` writes a directory tree keyed by `scenario`, and reading
    it back does not guarantee the plain string column that was written: on
    pyarrow 25.0.1 / pandas 3.0.5 (verified directly against this repo's
    pinned versions), the partition key comes back as a pandas categorical,
    which groups and compares differently from the in-memory frame. It is
    converted back here; the conversion is kept unconditional (an isinstance
    check, not an assumption) since this behaviour has varied across pyarrow
    releases and the check costs nothing when it turns out to be a no-op.

    Row order is also not preserved by a partitioned read -- rows come back
    grouped by partition, not in the order they were written. This function
    does not attempt to restore it; a caller that needs order-independent
    comparison should sort by a key before comparing.
    """
    sidecar_path = directory / _SIDECAR_NAME
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"{directory} has no {_SIDECAR_NAME}; it was not written by `bench run`"
        )
    # `engine="pyarrow"` is left implicit (falls back to the "auto" overload):
    # the "pyarrow" overload in this pandas-stubs release requires a mandatory
    # `to_pandas_kwargs` argument that has no bearing on this read, and pyarrow
    # is the only parquet engine this package depends on, so "auto" resolves
    # to it anyway.
    frame = pd.read_parquet(directory / "results")
    if isinstance(frame["scenario"].dtype, pd.CategoricalDtype):
        frame["scenario"] = frame["scenario"].astype(object)
    sidecar = json.loads(sidecar_path.read_text())
    assert isinstance(sidecar, dict)
    return frame, sidecar
