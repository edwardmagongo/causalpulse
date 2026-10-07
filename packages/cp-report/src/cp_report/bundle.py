"""The study bundle: the benchmark report's evidence as one JSON document.

A third rendering of `BenchmarkEvidence`, beside the HTML and Markdown
reports, for the dashboard's benchmark explorer. It is written in the same
run and from the same evidence object as the Markdown report, and records
that report's SHA-256, so the two cannot be regenerated apart.

What it adds to the report is the headline tables behind the report's two
curated ones, at both of their levels, with each metric's Monte Carlo
standard error and the number of fits behind it. It holds no image: the
explorer draws its own matrix from the tables.

A bundle is byte-identical whenever it is rebuilt from the same stores. It
carries no timestamp and no path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from cp_core.catalogue import ESTIMATOR_FAMILIES
from cp_evaluation.store import RunStore
from cp_evaluation.study import LEVELS, METRIC_DEFINITIONS, TABLE_METRICS

from cp_report.benchmark import BenchmarkEvidence, _section_data, document_prose_fragments
from cp_report.language import require_clean_prose

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunProvenance:
    """One run store a bundle was built from: the role it played, and what it holds.

    No path and no timestamp, so a bundle stays byte-identical across rebuilds.
    """

    role: str
    tier: str
    replications: int
    cells_done: int
    cells_expected: int
    git_commit: str | None


def run_provenance(role: str, directory: Path) -> RunProvenance:
    """`directory`'s provenance, read from its manifest and ledger; no result is loaded."""
    store = RunStore.open(directory)
    status = store.status()
    commit = store.provenance.get("git_commit")
    return RunProvenance(
        role=role,
        tier=store.identity.tier,
        replications=store.replications,
        cells_done=status.done,
        cells_expected=status.expected,
        git_commit=None if commit is None else str(commit),
    )


def _plain(value: object) -> object:
    """One table cell as JSON holds it: a missing number is `None`, never `NaN`."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int | str):
        return value
    raise TypeError(f"a bundle table cannot hold a {type(value).__name__}: {value!r}")


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    columns = [str(column) for column in frame.columns]
    return [
        dict(zip(columns, (_plain(value) for value in row), strict=True))
        for row in frame.itertuples(index=False, name=None)
    ]


def _families(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, list[dict[str, object]]]]:
    """`{family: {level: rows}}` from the headline tables' `{family}__{level}` names."""
    families: dict[str, dict[str, list[dict[str, object]]]] = {}
    for name in sorted(tables):
        family, separator, level = name.partition("__")
        if not separator or level not in LEVELS:
            raise ValueError(f"headline table {name!r} is not named {{family}}__{{level}}")
        families.setdefault(family, {})[level] = _records(tables[name])
    return families


def benchmark_bundle(
    evidence: BenchmarkEvidence,
    report_markdown: bytes,
    provenance: Sequence[RunProvenance] = (),
) -> dict[str, Any]:
    """The bundle as a JSON-ready dict. `report_markdown` is the Markdown report's bytes."""
    overview, findings, comparisons, limitations = _section_data(evidence)
    cate, p7 = evidence.cate_section, evidence.p7_section
    return {
        "schema_version": SCHEMA_VERSION,
        "report_sha256": hashlib.sha256(report_markdown).hexdigest(),
        "provenance": [asdict(run) for run in provenance],
        "overview": asdict(overview),
        "expectations": {
            "n_met": findings.n_met,
            "n_failed": findings.n_failed,
            "n_inconclusive": findings.n_inconclusive,
            "rows": [asdict(row) for row in findings.rows],
        },
        "estimators": [
            {"name": name, "family": ESTIMATOR_FAMILIES.get(name)}
            for name in evidence.estimators_run
        ],
        "scenarios": [
            {"name": name, "shape": (evidence.scenario_shapes or {}).get(name)}
            for name in evidence.scenarios_run
        ],
        "levels": {level: list(columns) for level, columns in LEVELS.items()},
        "family_metrics": {family: list(metrics) for family, metrics in TABLE_METRICS.items()},
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "families": _families(evidence.tables),
        "report_tables": {
            "tables": {title: _records(table) for title, table in comparisons.tables.items()},
            "coverage_note": comparisons.coverage_note,
            "exploratory_note": comparisons.exploratory_note,
        },
        "cate": None
        if cate is None
        else {
            "table": _records(cate.table),
            "replications": cate.replications,
            "cells_done": cate.cells_done,
            "cells_expected": cate.cells_expected,
            "note": cate.note,
        },
        "p7": None
        if p7 is None
        else {
            "table": _records(p7.table),
            "replications": p7.replications,
            "cells_done": p7.cells_done,
            "cells_expected": p7.cells_expected,
            "note": p7.note,
        },
        "limitations": {
            "not_covered": [
                {"name": name, "reason": reason} for name, reason in limitations.not_covered
            ],
            "statements": list(limitations.statements),
        },
    }


def render_benchmark_bundle(
    evidence: BenchmarkEvidence,
    report_markdown: bytes,
    provenance: Sequence[RunProvenance] = (),
) -> str:
    """The bundle's text: sorted keys, two-space indent, a trailing newline.

    `allow_nan=False` makes a stray `NaN` an error here instead of invalid JSON
    in a committed file.
    """
    bundle = benchmark_bundle(evidence, report_markdown, provenance)
    return json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_benchmark_bundle(
    evidence: BenchmarkEvidence,
    path: Path,
    report_path: Path,
    provenance: Sequence[RunProvenance] = (),
) -> Path:
    """Write the bundle to `path`, hashed against the Markdown report at `report_path`.

    Lints the same prose the report's renderers lint, before writing anything.
    The bundle's parent directory must already exist, for the report writers'
    reason: this never creates a path the caller may not have meant.
    """
    require_clean_prose(document_prose_fragments(evidence))
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"write_benchmark_bundle requires an existing directory, got {path.parent!r}"
        )
    text = render_benchmark_bundle(evidence, report_path.read_bytes(), provenance)
    path.write_text(text, encoding="utf-8")
    return path
