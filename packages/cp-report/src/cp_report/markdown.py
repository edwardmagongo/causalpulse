"""GitHub-flavoured Markdown renderers for the report's four sections.

Each renderer takes exactly one of `cp_report.sections`'s four frozen
dataclasses and returns a Markdown fragment -- a heading, a table or prose,
nothing else. Task 8 assembles the four fragments into a document; nothing
here writes a file or knows about the other sections.

**A renderer reads, it does not compute.** Every count, rank, bound or rate
a reader might want to check already lives on the dataclass a data function
built -- `n_passed`, `n_not_testable`, the per-row `lower`/`upper` bounds --
so a renderer only ever formats a field, never derives one. A number
computed inside a renderer would be testable only through a string
comparison, the brittlest test there is.

`statistic`, `threshold` and `remedy` on `AssumptionTableData` are `None`
exactly when the underlying check never carried a value there (see that
class's docstring). This module renders a `None` in any of those three
positions as an em dash (`—`), distinct from both a real zero and the
literal word "None", so a reader cannot mistake "not computed" for
"computed as zero" -- the confusion this project already shipped once in a
benchmark heatmap.
"""

from __future__ import annotations

from pathlib import Path

from cp_report.evidence import Evidence, ReportFigure
from cp_report.figures import figure_to_file
from cp_report.language import require_clean_prose
from cp_report.sections import (
    AssumptionTableData,
    ComparisonData,
    LimitationsData,
    ValidityBandData,
    assumption_table_data,
    comparison_data,
    limitations_data,
    validity_band_data,
)

_ABSENT = "—"  # em dash: a `None` statistic, threshold or remedy.
_DOCUMENT_NAME = "report.md"


def _cell(value: object) -> str:
    """Render one table cell, keeping `None` visibly distinct from `0.00`."""
    if value is None:
        return _ABSENT
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_assumption_table_markdown(data: AssumptionTableData) -> str:
    """Render the assumption checks as a heading, a summary line and a table."""
    header = "| Check | Kind | Status | Statistic | Threshold | Interpretation | Remedy |"
    separator = "| --- | --- | --- | --- | --- | --- | --- |"
    rows = [
        f"| {name} | {kind} | {status} | {_cell(statistic)} | {_cell(threshold)} | "
        f"{interpretation} | {_cell(remedy)} |"
        for name, kind, status, statistic, threshold, interpretation, remedy in zip(
            data.names,
            data.kinds,
            data.statuses,
            data.statistics,
            data.thresholds,
            data.interpretations,
            data.remedies,
            strict=True,
        )
    ]
    summary = (
        f"Passed: {data.n_passed} · Warned: {data.n_warned} · "
        f"Failed: {data.n_failed} · Not testable: {data.n_not_testable}"
    )
    return "\n".join(["## Assumption Checks", "", summary, "", header, separator, *rows])


def render_validity_band_markdown(data: ValidityBandData) -> str:
    """Render the validity band as a heading, band line, rationale and limits."""
    rationale = "\n".join(f"- {reason}" for reason in data.rationale)
    return "\n".join(
        [
            "## Validity Band",
            "",
            f"**Band:** {data.band} (ceiling: {data.ceiling})",
            "",
            rationale,
            "",
            data.limits,
        ]
    )


def render_comparison_markdown(data: ComparisonData) -> str:
    """Render the cross-method comparison as a heading and a table, one row per estimate."""
    header = "| Method | Estimand | Point | Lower | Upper | Basis |"
    separator = "| --- | --- | --- | --- | --- | --- |"
    rows = [
        f"| {method} | {estimand} | {point:.2f} | {lower:.2f} | {upper:.2f} | {basis} |"
        for method, estimand, point, lower, upper, basis in zip(
            data.methods,
            data.estimands,
            data.points,
            data.lower,
            data.upper,
            data.bases,
            strict=True,
        )
    ]
    return "\n".join(["## Cross-Method Comparison", "", header, separator, *rows])


def render_limitations_markdown(data: LimitationsData) -> str:
    """Render the limitations section, leading with the score's own limits."""
    statements = "\n".join(f"- {statement}" for statement in data.statements)
    parts = ["## Limitations", "", data.limits]
    if statements:
        parts.extend(["", statements])
    return "\n".join(parts)


def _render_figures_markdown(figures: tuple[ReportFigure, ...], paths: list[Path]) -> str:
    """Render the figures section: a heading plus one image link per figure.

    Each link is `![caption](filename)` with a bare filename, not a path --
    `write_markdown_report` writes both the figure files and the document
    itself into the same directory, so a bare filename resolves relative to
    the document regardless of the caller's own working directory.
    """
    links = "\n".join(
        f"![{figure.caption}]({path.name})"
        for figure, path in zip(figures, paths, strict=True)
    )
    parts = ["## Figures"]
    if links:
        parts.extend(["", links])
    return "\n".join(parts)


def write_markdown_report(evidence: Evidence, directory: Path) -> Path:
    """Write the report as a directory: one Markdown document plus its figures.

    Lints before writing anything. The four section data objects are built
    and their `prose_fragments()` are concatenated (with section names) and
    handed to `require_clean_prose` before this function touches the
    filesystem at all -- a refusal must never leave a half-written directory
    behind, which is what `test_a_refused_report_writes_nothing` pins. Only
    after that call returns does this function write the figure files (via
    `figure_to_file`) and then the document, in that order.
    """
    assumption = assumption_table_data(evidence.checks)
    validity = validity_band_data(evidence.score)
    comparison = comparison_data(evidence.estimates)
    limitations = limitations_data(evidence.score, evidence.limitations)

    fragments: list[tuple[str, str]] = []
    for section_name, data in (
        ("Assumption Checks", assumption),
        ("Validity Band", validity),
        ("Cross-Method Comparison", comparison),
        ("Limitations", limitations),
    ):
        fragments.extend((section_name, text) for text in data.prose_fragments())
    require_clean_prose(fragments)

    figure_paths = [figure_to_file(figure, directory) for figure in evidence.figures]

    document = "\n\n".join(
        [
            "# Causal Pulse Report",
            render_assumption_table_markdown(assumption),
            render_validity_band_markdown(validity),
            _render_figures_markdown(evidence.figures, figure_paths),
            render_comparison_markdown(comparison),
            render_limitations_markdown(limitations),
        ]
    )
    path = directory / _DOCUMENT_NAME
    path.write_text(document)
    return path
