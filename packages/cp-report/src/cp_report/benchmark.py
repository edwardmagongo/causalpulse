"""The benchmark study's research report: data, sections and renderers.

Sub-project 2, part 2 (`docs/specs/2026-09-23-benchmark-study-report-design.md`).
Mirrors `cp_report.evidence`/`sections`/`html`/`markdown` for a different document
shape: aggregate performance across many estimators and scenarios rather than one
analysis's assumption checks. `BenchmarkEvidence` plays `Evidence`'s role -- the one
argument each entry point takes -- and is built by `benchmark_evidence_from_run`
from a finished study's stored results, the only function here that imports
`cp_evaluation`.

Four sections, each a frozen dataclass with a `prose_fragments()` method, exactly as
`cp_report.sections` does: `OverviewData`, `HeadlineFindingsData`,
`MethodComparisonsData`, `LimitationsData` (this module's own, distinct from
`cp_report.sections.LimitationsData`). No new figure-drawing code: every
`ReportFigure` wraps an existing `cp_figures.benchmark` draw function via `partial`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from cp_evaluation.expectations import ExpectationResult, Verdict, evaluate_expectations
from cp_evaluation.metrics import summarise
from cp_evaluation.store import RunStatus, RunStore
from cp_evaluation.study import NOT_COVERED, baseline_slice, headline_tables
from cp_figures.benchmark import axis_profile_data, draw_axis_profile, draw_heatmap, heatmap_data

from cp_report.evidence import ReportFigure

# One heatmap per metric over the baseline-slice pooled summary -- a subset of the
# metrics `cp_cli.commands.study` draws, so a reader comparing the report against
# `bench study`'s exhaustive figure set is comparing the same quantities. Which
# figures are worth a reader's attention is an editorial choice recorded in the
# design doc's "As built" section once real numbers exist; these are the starting set.
REPORT_HEATMAP_METRICS: tuple[str, ...] = ("bias", "coverage_analytic")
REPORT_PROFILE_SELECTIONS: tuple[tuple[str, str, str], ...] = (
    ("D", "confounding_strength", "bias"),
)

# Which of `headline_tables`'s tables the document embeds. `BenchmarkEvidence.tables`
# keeps the full set as a record; the document shows a curated few, not a raw dump.
REPORT_TABLES: tuple[str, ...] = ("accuracy__scenario", "coverage__scenario")

# `coverage_metrics` puts every estimator's `ci_lower`/`ci_upper` in `coverage_analytic`
# regardless of `ci_basis` (design doc 2026-09-21 §13.11), so for these two estimators
# the column name overclaims. The note is code, not narrative, so no author can forget
# it; re-read it if `coverage_metrics` is ever made basis-aware.
COVERAGE_COLUMN_NOTE = (
    "The coverage_analytic column reports how often each estimator's primary reported "
    "interval contains the true effect, whatever kind of interval that is: for "
    "hierarchical_did it is a posterior credible interval, and for callaway_santanna it "
    "is a Wald interval built from a cluster-bootstrap standard error."
)


@dataclass(frozen=True)
class OverviewData:
    """The grid's shape and what its replication count buys in precision.

    `coverage_mcse` is the binomial Monte Carlo standard error a coverage figure
    carries at this replication count, `sqrt(.95*.05/replications)` -- the arithmetic
    `cp_evaluation.study.coverage_of_study_markdown` already does.
    """

    n_estimators: int
    n_scenarios: int
    n_cells: int
    replications: int
    coverage_mcse: float

    def prose_fragments(self) -> tuple[str, ...]:
        """Nothing here is free text: every field is a count or a computed number."""
        return ()


def overview_data(
    n_estimators: int, n_scenarios: int, n_cells: int, replications: int
) -> OverviewData:
    return OverviewData(
        n_estimators=n_estimators,
        n_scenarios=n_scenarios,
        n_cells=n_cells,
        replications=replications,
        coverage_mcse=float(np.sqrt(0.95 * 0.05 / replications)),
    )


@dataclass(frozen=True)
class ExpectationRow:
    """One pre-registered expectation as a report reader sees it.

    `narrative` is the caller's reader-facing paragraph when one was supplied, else
    `result.note`, else `result.statement` -- `ExpectationResult.note` exists for
    exactly this, a reworded explanation standing in for the raw statement.
    """

    id: str
    statement: str
    verdict: Verdict
    narrative: str


@dataclass(frozen=True)
class HeadlineFindingsData:
    """One row per pre-registered expectation, plus the tally a reader might check."""

    rows: tuple[ExpectationRow, ...]
    n_met: int
    n_failed: int
    n_inconclusive: int

    def prose_fragments(self) -> tuple[str, ...]:
        """Included: each row's `statement` and, when it differs, its `narrative` --
        both free text, and the narrative is exactly the caller-supplied override the
        lint exists for. Excluded: `verdict`, from a fixed vocabulary."""
        fragments: list[str] = []
        for row in self.rows:
            fragments.append(row.statement)
            if row.narrative != row.statement:
                fragments.append(row.narrative)
        return tuple(fragments)


def headline_findings_data(
    results: Sequence[ExpectationResult], narratives: Mapping[str, str]
) -> HeadlineFindingsData:
    rows = tuple(
        ExpectationRow(
            id=result.id,
            statement=result.statement,
            verdict=result.verdict,
            narrative=narratives.get(result.id, result.note or result.statement),
        )
        for result in results
    )
    return HeadlineFindingsData(
        rows=rows,
        n_met=sum(1 for r in results if r.verdict == "met"),
        n_failed=sum(1 for r in results if r.verdict == "failed"),
        n_inconclusive=sum(1 for r in results if r.verdict == "inconclusive"),
    )


@dataclass(frozen=True, eq=False)
class MethodComparisonsData:
    """The curated tables, the coverage-column note, and the exploratory note.

    `eq=False`: holds DataFrames, whose comparison has an ambiguous truth value.
    `coverage_note` is `COVERAGE_COLUMN_NOTE` when a coverage table is embedded, else
    empty.
    """

    tables: dict[str, pd.DataFrame]
    exploratory_note: str
    coverage_note: str

    def prose_fragments(self) -> tuple[str, ...]:
        """Included: both notes. Excluded: `tables`, numeric data."""
        return (self.exploratory_note, *((self.coverage_note,) if self.coverage_note else ()))


def method_comparisons_data(
    tables: dict[str, pd.DataFrame], exploratory_note: str
) -> MethodComparisonsData:
    """Curate `REPORT_TABLES` out of the full `headline_tables()` set.

    A curated name missing from `tables` is skipped rather than raising: the document
    renders what it has.
    """
    selected = {name: tables[name] for name in REPORT_TABLES if name in tables}
    has_coverage = any(name.startswith("coverage") for name in selected)
    return MethodComparisonsData(
        tables=selected,
        exploratory_note=exploratory_note,
        coverage_note=COVERAGE_COLUMN_NOTE if has_coverage else "",
    )


@dataclass(frozen=True)
class LimitationsData:
    """What the study did not cover, plus whatever else the caller states.

    `not_covered` is `cp_evaluation.study.NOT_COVERED`, the structural source this
    section leads with; `statements` is the caller's tuple, carried through unmodified.
    """

    not_covered: tuple[tuple[str, str], ...]
    statements: tuple[str, ...]

    def prose_fragments(self) -> tuple[str, ...]:
        """Included: every `not_covered` reason and every caller statement -- the
        likeliest place for an unhedged claim, since a report's author wrote it.
        Excluded: the `not_covered` names, from a fixed set of identifiers."""
        return tuple(reason for _, reason in self.not_covered) + self.statements


def limitations_data(
    not_covered: tuple[tuple[str, str], ...], statements: tuple[str, ...]
) -> LimitationsData:
    return LimitationsData(not_covered=not_covered, statements=statements)


@dataclass(frozen=True, eq=False)
class BenchmarkEvidence:
    """Everything the two entry points need, gathered once by `benchmark_evidence_from_run`."""

    expectations: tuple[ExpectationResult, ...]
    tables: dict[str, pd.DataFrame]
    status: RunStatus
    replications: int
    scenarios_run: tuple[str, ...]
    estimators_run: tuple[str, ...]
    not_covered: tuple[tuple[str, str], ...]
    figures: tuple[ReportFigure, ...]
    limitations: tuple[str, ...]
    narratives: dict[str, str]
    exploratory_note: str


def _exploratory_note(by_size: pd.DataFrame) -> str:
    """Describe the analytic-versus-bootstrap comparison with no direction claimed --
    the study design's "deliberately not pre-registered" honoured literally: this
    states what is in the table, not a conclusion about it."""
    if by_size.empty:
        return (
            "The analytic-versus-bootstrap coverage comparison across sample sizes has no rows "
            "to report in this run."
        )
    return (
        "Analytic and bootstrap interval coverage are compared across the two cross-sectional "
        "sample sizes the grid varies. This comparison was not pre-registered as an expectation, "
        "and no direction is claimed here -- see the full table for the measured values."
    )


def _report_figures(frame: pd.DataFrame) -> tuple[ReportFigure, ...]:
    """The curated figure set, each wrapping an existing `cp_figures.benchmark` draw
    function. A profile selection whose scenario is absent from `frame` is skipped,
    like a missing curated table."""
    pooled = summarise(baseline_slice(frame), by=["scenario"])
    per_cell = summarise(frame)
    present = {str(s) for s in frame["scenario"].unique()}
    figures = [
        ReportFigure(
            caption=f"{metric} by method and scenario, pooled over each scenario's baseline slice.",
            draw=partial(draw_heatmap, heatmap_data(pooled, metric)),
        )
        for metric in REPORT_HEATMAP_METRICS
    ]
    for scenario, axis, metric in REPORT_PROFILE_SELECTIONS:
        if scenario not in present:
            continue
        figures.append(
            ReportFigure(
                caption=f"{metric} on scenario {scenario}, varying {axis} one value at a time.",
                draw=partial(
                    draw_axis_profile,
                    axis_profile_data(per_cell, metric, axis=axis, scenario=scenario),
                ),
            )
        )
    return tuple(figures)


def benchmark_evidence_from_run(
    directory: Path,
    *,
    limitations: tuple[str, ...] = (),
    narratives: Mapping[str, str] | None = None,
) -> BenchmarkEvidence:
    """Build `BenchmarkEvidence` from a finished, complete sharded study.

    Refuses an incomplete run by reusing `RunStore.read_frame`'s own refusal
    (`IncompleteRunError`). `limitations` and `narratives` are the two places a caller
    states something the stored data cannot: no text is baked in here, including the
    `hierarchical_did`/P6 exclusion, which the generating command states explicitly.
    A narrative keyed to an expectation id that does not exist is refused, not ignored.
    """
    store = RunStore.open(directory)
    frame = store.read_frame()
    status = store.status()
    results = evaluate_expectations(frame, status)

    narratives = dict(narratives or {})
    unknown = sorted(set(narratives) - {r.id for r in results})
    if unknown:
        raise ValueError(f"narratives name unknown expectation ids: {unknown}")

    tables = headline_tables(frame)
    return BenchmarkEvidence(
        expectations=results,
        tables=tables,
        status=status,
        replications=store.replications,
        scenarios_run=tuple(sorted(frame["scenario"].unique())),
        estimators_run=tuple(sorted(frame["method"].unique())),
        not_covered=NOT_COVERED,
        figures=_report_figures(frame),
        limitations=limitations,
        narratives=narratives,
        exploratory_note=_exploratory_note(
            tables.get("coverage__scenario_by_size", pd.DataFrame())
        ),
    )
