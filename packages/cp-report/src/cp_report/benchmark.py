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

from collections.abc import Callable, Mapping, Sequence
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
    baseline_slice,
    format_value,
    headline_tables,
    markdown_table,
    not_covered_for,
)
from cp_figures.benchmark import axis_profile_data, draw_axis_profile, draw_heatmap, heatmap_data
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cp_report.evidence import ReportFigure
from cp_report.figures import figure_to_file, figure_to_inline_svg
from cp_report.language import require_clean_prose

# One heatmap: each estimator's primary-interval coverage over the baseline-slice pooled
# summary, in a synthetic `coverage_primary` column (see `_primary_values`). A bias heatmap
# was tried and dropped: two scenarios built to break the DiD family (P4, P5) have biases
# hundreds of times the rest, so the colour scale left every other cell one colour, and the
# bias table already gives the numbers. The choice is recorded in the design doc's
# "As built" section.
REPORT_HEATMAP_METRICS: tuple[str, ...] = ("coverage_primary",)
REPORT_PROFILE_SELECTIONS: tuple[tuple[str, str, str], ...] = (
    ("D", "confounding_strength", "bias"),
)

# The two tables the document embeds, each a compact method-by-scenario view built from a
# `headline_tables` source table (`BenchmarkEvidence.tables` keeps the full set as a record;
# a 130-row, 18-column source table is not something a reader can read). Bias comes from
# `accuracy__scenario`, coverage from `coverage__scenario`.
BIAS_TABLE = "Bias by method and scenario"
COVERAGE_TABLE = "Coverage of each estimator's primary interval, by scenario"

# `coverage_metrics` reports coverage per `ci_basis`
# (2026-09-23-coverage-basis-metrics-design.md). Each estimator's primary interval has
# exactly one basis, so the coverage table collapses the basis columns into one value per
# cell and an `interval` column naming the basis. The note is code, not narrative, so no
# author can forget what that column means; re-read it against `coverage_metrics` if the
# basis columns change.
COVERAGE_COLUMN_NOTE = (
    "Each coverage value is how often the interval an estimator reports as its primary one "
    "contains the true effect, and the interval column names that interval's kind: analytic "
    "for standard-error intervals, posterior for the credible interval hierarchical_did "
    "reports, bootstrap_se for the Wald interval callaway_santanna builds from a "
    "cluster-bootstrap standard error, and permutation for the interval synthetic_control "
    "builds from its placebo distribution."
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


# The `coverage_metrics` columns that hold a primary interval's coverage, by the `ci_basis`
# each reads. An estimator's primary interval has exactly one basis, so at most one of these
# is populated for any of its rows.
_PRIMARY_COVERAGE: tuple[tuple[str, str], ...] = (
    ("analytic", "coverage_analytic"),
    ("posterior", "coverage_posterior"),
    ("bootstrap_se", "coverage_bootstrap_se"),
    ("permutation", "coverage_permutation"),
)


def _primary_values(frame: pd.DataFrame) -> pd.Series:
    """Each row's primary-interval coverage: the first populated `_PRIMARY_COVERAGE` column.

    Empty when the frame carries none of those columns.
    """
    columns = [column for _, column in _PRIMARY_COVERAGE if column in frame.columns]
    if not columns:
        return pd.Series(np.nan, index=frame.index)
    return frame[columns].bfill(axis=1).iloc[:, 0]


def _bias_table(accuracy: pd.DataFrame) -> pd.DataFrame:
    """Bias, one row per method and one column per scenario; blank where not fitted."""
    wide = accuracy.pivot(index="method", columns="scenario", values="bias")
    wide.columns.name = None
    return wide.reset_index()


def _coverage_table(coverage: pd.DataFrame) -> pd.DataFrame:
    """Each estimator's primary-interval coverage per scenario, with the interval's kind.

    Whichever `_PRIMARY_COVERAGE` column an estimator populates supplies its values; the
    `interval` column names that column's basis, and is empty for an estimator with no
    fitted row anywhere.
    """
    present = [(kind, column) for kind, column in _PRIMARY_COVERAGE if column in coverage.columns]
    values = _primary_values(coverage)
    kinds = {
        str(method): next((kind for kind, column in present if rows[column].notna().any()), "")
        for method, rows in coverage.groupby("method")
    }
    wide = coverage.assign(_value=values).pivot(index="method", columns="scenario", values="_value")
    wide.columns.name = None
    wide.insert(0, "interval", [kinds[str(method)] for method in wide.index])
    return wide.reset_index()


def method_comparisons_data(
    tables: dict[str, pd.DataFrame], exploratory_note: str
) -> MethodComparisonsData:
    """Build the curated bias and coverage tables from the full `headline_tables()` set.

    A source table missing from `tables` is skipped rather than raising: the document
    renders what it has. `coverage_note` accompanies the coverage table and only it.
    """
    curated: dict[str, pd.DataFrame] = {}
    if "accuracy__scenario" in tables:
        curated[BIAS_TABLE] = _bias_table(tables["accuracy__scenario"])
    if "coverage__scenario" in tables:
        curated[COVERAGE_TABLE] = _coverage_table(tables["coverage__scenario"])
    return MethodComparisonsData(
        tables=curated,
        exploratory_note=exploratory_note,
        coverage_note=COVERAGE_COLUMN_NOTE if COVERAGE_TABLE in curated else "",
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
        "Analytic and bootstrap interval coverage were also compared across the two "
        "cross-sectional sample sizes the grid varies; that comparison is the "
        "coverage__scenario_by_size table in the study's own output and is not reproduced "
        "here. It was not pre-registered as an expectation, and no direction is claimed."
    )


_HEATMAP_CAPTIONS = {
    "coverage_primary": (
        "Coverage of each estimator's primary interval, by method and scenario, pooled over "
        "each scenario's baseline slice. Lighter cells have higher coverage; grey cells have "
        "no fitted result."
    )
}


def _laid_out(draw: Callable[[Axes], Axes]) -> Callable[[Axes], Axes]:
    """Wrap a draw function so its figure lays itself out once drawn.

    `figure_to_inline_svg`/`figure_to_file` save a default-margin figure, which clips a
    long tick label such as `callaway_santanna`; `bench study` avoids that by saving with
    `bbox_inches="tight"`, which the shared renderers do not do.
    """

    def drawn(ax: Axes) -> Axes:
        result = draw(ax)
        figure = ax.get_figure()
        if isinstance(figure, Figure):
            figure.tight_layout()
        return result

    return drawn


def _report_figures(frame: pd.DataFrame) -> tuple[ReportFigure, ...]:
    """The curated figure set, each wrapping an existing `cp_figures.benchmark` draw
    function. A profile selection whose scenario is absent from `frame` is skipped,
    like a missing curated table."""
    pooled = summarise(baseline_slice(frame), by=["scenario"])
    pooled["coverage_primary"] = _primary_values(pooled)
    per_cell = summarise(frame)
    present = {str(s) for s in frame["scenario"].unique()}
    figures = [
        ReportFigure(
            caption=_HEATMAP_CAPTIONS[metric],
            draw=_laid_out(partial(draw_heatmap, heatmap_data(pooled, metric))),
        )
        for metric in REPORT_HEATMAP_METRICS
    ]
    for scenario, axis, metric in REPORT_PROFILE_SELECTIONS:
        if scenario not in present:
            continue
        figures.append(
            ReportFigure(
                caption=f"{metric} on scenario {scenario}, varying {axis} one value at a time.",
                draw=_laid_out(
                    partial(
                        draw_axis_profile,
                        axis_profile_data(per_cell, metric, axis=axis, scenario=scenario),
                    )
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
        not_covered=not_covered_for(frame),
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
