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
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from cp_evaluation.expectations import ExpectationResult, Verdict, evaluate_expectations
from cp_evaluation.metrics import summarise
from cp_evaluation.store import RunStatus, RunStore
from cp_evaluation.study import (
    NOT_COVERED,
    baseline_slice,
    format_value,
    headline_tables,
    markdown_table,
)
from cp_figures.benchmark import axis_profile_data, draw_axis_profile, draw_heatmap, heatmap_data

from cp_report.evidence import ReportFigure
from cp_report.figures import figure_to_file, figure_to_inline_svg
from cp_report.language import require_clean_prose

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

# `coverage_metrics` reports coverage per `ci_basis` (design doc 2026-09-21 §13.11 and
# 2026-09-23-coverage-basis-metrics-design.md): `coverage_analytic` is analytic intervals
# only, so it is blank for the two estimators whose primary interval is not analytic. The
# note is code, not narrative, so no author can forget it; re-read it against
# `coverage_metrics` if the columns change again.
COVERAGE_COLUMN_NOTE = (
    "Coverage is reported per kind of interval, because an estimator's primary interval "
    "is not always analytic. The coverage_analytic column covers analytic intervals only, "
    "so it is blank for hierarchical_did, whose interval is a posterior credible interval "
    "reported as coverage_posterior, and for callaway_santanna, whose interval is a Wald "
    "interval built from a cluster-bootstrap standard error reported as "
    "coverage_bootstrap_se. The coverage_bootstrap column is a separate percentile-bootstrap "
    "interval, where an estimator reports one."
)

# Appended to the coverage_analytic heatmap's caption: its blank cells are those two
# estimators, not missing data.
_BLANK_ANALYTIC_CELLS = (
    " Blank cells are estimators whose primary interval is not analytic; the coverage "
    "table names the column that covers them."
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


def _drop_empty_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop every metric with no value in any row, with its `_mcse` and `_n_used`.

    A metric column is empty when no estimator has that kind of interval yet:
    `coverage_permutation` is blank for every estimator until a donor-pool arm exists.
    A metric is recognised by having an `_n_used` sibling, so status and identity columns
    are never dropped. A frame with no rows is returned unchanged.
    """
    if frame.empty:
        return frame
    empty = [c for c in frame.columns if f"{c}_n_used" in frame.columns and frame[c].isna().all()]
    drop = [name for c in empty for name in (c, f"{c}_mcse", f"{c}_n_used")]
    return frame.drop(columns=drop)


def method_comparisons_data(
    tables: dict[str, pd.DataFrame], exploratory_note: str
) -> MethodComparisonsData:
    """Curate `REPORT_TABLES` out of the full `headline_tables()` set.

    A curated name missing from `tables` is skipped rather than raising: the document
    renders what it has. Metrics empty in every row are dropped from each table.
    """
    selected = {
        name: _drop_empty_metrics(tables[name]) for name in REPORT_TABLES if name in tables
    }
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
            caption=(
                f"{metric} by method and scenario, pooled over each scenario's baseline slice."
                + (_BLANK_ANALYTIC_CELLS if metric == "coverage_analytic" else "")
            ),
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


# -- prose assembly and the language lint -----------------------------------------


def _section_data(
    evidence: BenchmarkEvidence,
) -> tuple[OverviewData, HeadlineFindingsData, MethodComparisonsData, LimitationsData]:
    """The four sections, rebuilt from `evidence` in one place so a fifth section is a
    one-place edit and the prose the lint sees cannot drift from the prose rendered."""
    return (
        overview_data(
            len(evidence.estimators_run),
            len(evidence.scenarios_run),
            evidence.status.expected,
            evidence.replications,
        ),
        headline_findings_data(evidence.expectations, evidence.narratives),
        method_comparisons_data(evidence.tables, evidence.exploratory_note),
        limitations_data(evidence.not_covered, evidence.limitations),
    )


def document_prose_fragments(evidence: BenchmarkEvidence) -> list[tuple[str, str]]:
    """Every fragment of reader-facing prose a report built from `evidence` shows,
    tagged with its section, in rendering order. Mirrors
    `cp_report.sections.document_prose_fragments`; a figure caption is its own fragment
    because it reaches the reader as a `<figcaption>` or an image's alt text."""
    overview, findings, comparisons, limitations = _section_data(evidence)
    fragments = [("Overview", text) for text in overview.prose_fragments()]
    fragments += [("Headline Findings", text) for text in findings.prose_fragments()]
    fragments += [("Figures", figure.caption) for figure in evidence.figures]
    fragments += [("Method Comparisons", text) for text in comparisons.prose_fragments()]
    fragments += [("Limitations", text) for text in limitations.prose_fragments()]
    return fragments


# -- HTML -------------------------------------------------------------------------

_STYLE = (
    "body{font-family:sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;}"
    "table{border-collapse:collapse;width:100%;margin:1rem 0;}"
    "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;}"
    "figure{margin:1rem 0;}figcaption{font-style:italic;}"
)


def _html_cell(value: object) -> str:
    return escape(format_value(value)) if value is not None else "—"


def _dataframe_to_html(frame: pd.DataFrame) -> str:
    header = "".join(f"<th>{escape(str(c))}</th>" for c in frame.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_html_cell(v)}</td>" for v in row) + "</tr>"
        for row in frame.itertuples(index=False, name=None)
    )
    return f"<table><tr>{header}</tr>{rows}</table>"


def render_overview_html(data: OverviewData) -> str:
    return (
        "<section><h2>Overview</h2>"
        f"<p>{data.n_estimators} estimators, {data.n_scenarios} scenarios, "
        f"{data.n_cells} cells, {data.replications} replications per configuration "
        f"(coverage MCSE ~= {data.coverage_mcse:.3f}).</p></section>"
    )


def render_headline_findings_html(data: HeadlineFindingsData) -> str:
    rows = "".join(
        f"<tr><td>{escape(row.id)}</td><td>{escape(row.verdict)}</td>"
        f"<td>{escape(row.narrative)}</td></tr>"
        for row in data.rows
    )
    summary = f"Met: {data.n_met} · Failed: {data.n_failed} · Inconclusive: {data.n_inconclusive}"
    return (
        "<section><h2>Headline Findings</h2>"
        f"<p>{summary}</p>"
        "<table><tr><th>ID</th><th>Verdict</th><th>Explanation</th></tr>"
        f"{rows}</table></section>"
    )


def render_method_comparisons_html(data: MethodComparisonsData) -> str:
    tables = "".join(
        f"<h3>{escape(name)}</h3>{_dataframe_to_html(frame)}" for name, frame in data.tables.items()
    )
    coverage_note = f"<p>{escape(data.coverage_note)}</p>" if data.coverage_note else ""
    return (
        "<section><h2>Method Comparisons</h2>"
        f"{tables}{coverage_note}<p><em>{escape(data.exploratory_note)}</em></p></section>"
    )


def render_limitations_html(data: LimitationsData) -> str:
    not_covered = "".join(
        f"<li>{escape(name)}: {escape(reason)}</li>" for name, reason in data.not_covered
    )
    statements = "".join(f"<li>{escape(s)}</li>" for s in data.statements)
    return f"<section><h2>Limitations</h2><ul>{not_covered}{statements}</ul></section>"


def _render_figures_html(figures: tuple[ReportFigure, ...]) -> str:
    if not figures:
        return ""
    body = "".join(
        f"<figure>{figure_to_inline_svg(figure)}<figcaption>{escape(figure.caption)}"
        "</figcaption></figure>"
        for figure in figures
    )
    return f"<section><h2>Figures</h2>{body}</section>"


def render_html_benchmark_report(evidence: BenchmarkEvidence) -> str:
    """Render the report as one self-contained HTML document.

    Lints before rendering, and carries its own `require_clean_prose` call rather than
    sharing one with the Markdown entry point, so a guard removed from one format
    cannot mask a gap in the other.
    """
    overview, findings, comparisons, limitations = _section_data(evidence)
    require_clean_prose(document_prose_fragments(evidence))

    body = "".join(
        [
            render_overview_html(overview),
            render_headline_findings_html(findings),
            _render_figures_html(evidence.figures),
            render_method_comparisons_html(comparisons),
            render_limitations_html(limitations),
        ]
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>CausalPulse Benchmark Study Report</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>CausalPulse Benchmark Study Report</h1>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


# -- Markdown ---------------------------------------------------------------------

_DOCUMENT_NAME = "benchmark-report.md"


def _md_cell(text: str) -> str:
    """A table cell: a pipe or a newline would end the row early."""
    return text.replace("|", "\\|").replace("\n", " ")


def render_overview_markdown(data: OverviewData) -> str:
    return (
        "## Overview\n\n"
        f"{data.n_estimators} estimators, {data.n_scenarios} scenarios, "
        f"{data.n_cells} cells, {data.replications} replications per configuration "
        f"(coverage MCSE ~= {data.coverage_mcse:.3f})."
    )


def render_headline_findings_markdown(data: HeadlineFindingsData) -> str:
    summary = f"Met: {data.n_met} · Failed: {data.n_failed} · Inconclusive: {data.n_inconclusive}"
    rows = [f"| {row.id} | {row.verdict} | {_md_cell(row.narrative)} |" for row in data.rows]
    table = "\n".join(["| ID | Verdict | Explanation |", "| --- | --- | --- |", *rows])
    return f"## Headline Findings\n\n{summary}\n\n{table}"


def render_method_comparisons_markdown(data: MethodComparisonsData) -> str:
    parts = ["## Method Comparisons"]
    for name, frame in data.tables.items():
        parts.extend([f"### {name}", markdown_table(frame).rstrip("\n")])
    if data.coverage_note:
        parts.append(data.coverage_note)
    parts.append(f"*{data.exploratory_note}*")
    return "\n\n".join(parts)


def render_limitations_markdown(data: LimitationsData) -> str:
    lines = [f"- {name}: {reason}" for name, reason in data.not_covered]
    lines.extend(f"- {statement}" for statement in data.statements)
    return "\n".join(["## Limitations", "", *lines])


def _render_figures_markdown(figures: tuple[ReportFigure, ...], paths: list[Path]) -> str:
    if not figures:
        return ""
    links = "\n".join(
        f"![{figure.caption}]({path.name})" for figure, path in zip(figures, paths, strict=True)
    )
    return f"## Figures\n\n{links}"


def write_markdown_benchmark_report(evidence: BenchmarkEvidence, directory: Path) -> Path:
    """Write the report as one Markdown document plus its SVG figures under `directory`.

    Lints before writing anything, and `directory` must already exist -- the two
    guarantees of `cp_report.markdown.write_markdown_report`, for the same reasons: a
    refusal must never leave a half-written directory, and this never silently `mkdir`s
    a path a caller may not have meant to create.
    """
    overview, findings, comparisons, limitations = _section_data(evidence)
    require_clean_prose(document_prose_fragments(evidence))

    if not directory.is_dir():
        raise FileNotFoundError(
            f"write_markdown_benchmark_report requires an existing directory, got {directory!r}"
        )

    figure_paths = [figure_to_file(figure, directory) for figure in evidence.figures]
    document = "\n\n".join(
        part
        for part in [
            "# CausalPulse Benchmark Study Report",
            render_overview_markdown(overview),
            render_headline_findings_markdown(findings),
            _render_figures_markdown(evidence.figures, figure_paths),
            render_method_comparisons_markdown(comparisons),
            render_limitations_markdown(limitations),
        ]
        if part
    )
    path = directory / _DOCUMENT_NAME
    path.write_text(document, encoding="utf-8")
    return path
