"""HTML renderers for the report's four sections.

Each renderer takes exactly one of `cp_report.sections`'s four frozen
dataclasses and returns an HTML fragment -- a `<section>` with a heading, a
`<table>` for a tabular section, `<p>`/`<ul>` for prose -- nothing else.
Task 8 wraps the four fragments in a document; nothing here emits
`<!DOCTYPE>` or `<html>`, writes a file, or knows about the other sections.
Mirrors `cp_report.markdown`'s structure so Task 8 can treat the two
modules interchangeably.

**A renderer reads, it does not compute.** Every count, rank, bound or rate
a reader might want to check already lives on the dataclass a data function
built -- `n_passed`, `n_not_testable`, the per-row `lower`/`upper` bounds --
so a renderer only ever formats a field, never derives one.

**Escaping.** Every interpolated value can originate in a dataset or a
caller -- a covariate name, a free-text interpretation, a remedy, a
limitation statement. Unlike GitHub-flavoured Markdown, HTML has no excuse
for leaving this unescaped: the standard library does it correctly. Every
value substituted into markup below goes through `html.escape` first,
including values that are normally well-behaved (a method name, a band
label) -- there is no field on these dataclasses guaranteed markup-free by
construction.

`statistic`, `threshold` and `remedy` on `AssumptionTableData` are `None`
exactly when the underlying check never carried a value there (see that
class's docstring). This module renders a `None` in any of those three
positions as an em dash (`—`), distinct from both a real zero and the
literal word "None", matching `cp_report.markdown`'s choice.
"""

from __future__ import annotations

from html import escape

from cp_report.evidence import Evidence, ReportFigure
from cp_report.figures import figure_to_inline_svg
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

# The four `prose_fragments()`-bearing dataclasses `render_html_report` lints
# around the figures section -- named so the two grouped tuples below share
# one annotation instead of mypy narrowing each to just the two variants its
# own literal happens to contain.
_SectionData = AssumptionTableData | ValidityBandData | ComparisonData | LimitationsData

_ABSENT = "—"  # em dash: a `None` statistic, threshold or remedy.

_STYLE = (
    "body{font-family:sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;}"
    "table{border-collapse:collapse;width:100%;margin:1rem 0;}"
    "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;}"
    "figure{margin:1rem 0;}figcaption{font-style:italic;}"
)


def _cell(value: object) -> str:
    """Render one escaped table cell, keeping `None` visibly distinct from `0.00`."""
    if value is None:
        return _ABSENT
    if isinstance(value, float):
        return f"{value:.2f}"
    return escape(str(value))


def render_assumption_table_html(data: AssumptionTableData) -> str:
    """Render the assumption checks as a heading, a summary paragraph and a table."""
    header = (
        "<tr><th>Check</th><th>Kind</th><th>Status</th><th>Statistic</th>"
        "<th>Threshold</th><th>Interpretation</th><th>Remedy</th></tr>"
    )
    rows = [
        (
            "<tr>"
            f"<td>{escape(name)}</td><td>{escape(kind)}</td><td>{escape(status)}</td>"
            f"<td>{_cell(statistic)}</td><td>{_cell(threshold)}</td>"
            f"<td>{escape(interpretation)}</td><td>{_cell(remedy)}</td>"
            "</tr>"
        )
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
    table = "".join(["<table>", header, *rows, "</table>"])
    return "".join(
        [
            "<section>",
            "<h2>Assumption Checks</h2>",
            f"<p>{summary}</p>",
            table,
            "</section>",
        ]
    )


def render_validity_band_html(data: ValidityBandData) -> str:
    """Render the validity band as a heading, band line, rationale list and limits."""
    rationale = "".join(f"<li>{escape(reason)}</li>" for reason in data.rationale)
    return "".join(
        [
            "<section>",
            "<h2>Validity Band</h2>",
            f"<p><strong>Band:</strong> {escape(data.band)} (ceiling: {escape(data.ceiling)})</p>",
            f"<ul>{rationale}</ul>",
            f"<p>{escape(data.limits)}</p>",
            "</section>",
        ]
    )


def render_comparison_html(data: ComparisonData) -> str:
    """Render the cross-method comparison as a heading and a table, one row per estimate."""
    header = (
        "<tr><th>Method</th><th>Estimand</th><th>Point</th><th>Lower</th>"
        "<th>Upper</th><th>Basis</th></tr>"
    )
    rows = [
        (
            "<tr>"
            f"<td>{escape(method)}</td><td>{escape(estimand)}</td>"
            f"<td>{point:.2f}</td><td>{lower:.2f}</td><td>{upper:.2f}</td>"
            f"<td>{escape(basis)}</td>"
            "</tr>"
        )
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
    table = "".join(["<table>", header, *rows, "</table>"])
    return "".join(["<section>", "<h2>Cross-Method Comparison</h2>", table, "</section>"])


def render_limitations_html(data: LimitationsData) -> str:
    """Render the limitations section, leading with the score's own limits."""
    parts = ["<section>", "<h2>Limitations</h2>", f"<p>{escape(data.limits)}</p>"]
    if data.statements:
        items = "".join(f"<li>{escape(statement)}</li>" for statement in data.statements)
        parts.append(f"<ul>{items}</ul>")
    parts.append("</section>")
    return "".join(parts)


def _render_figures_html(figures: tuple[ReportFigure, ...]) -> str:
    """Render the figures section: a heading plus one inlined `<svg>` per figure.

    Each figure is rendered via `figure_to_inline_svg`, never `figure_to_file`
    plus an `<img>` reference -- an HTML report is one self-contained
    document, and a caller who saves only the string must still get every
    figure with it.

    Returns the empty string when there are no figures, rather than an empty
    `<section><h2>Figures</h2></section>` -- a figure-less report is a legal
    input `Evidence` anticipates, and an empty section is a blemish a
    publication-quality document should not carry. Checking `len(figures)`
    is a presence check on data the caller already has, not a computation.
    """
    if not figures:
        return ""
    figs = "".join(
        f"<figure>{figure_to_inline_svg(figure)}<figcaption>{escape(figure.caption)}"
        "</figcaption></figure>"
        for figure in figures
    )
    return f"<section><h2>Figures</h2>{figs}</section>"


def render_html_report(evidence: Evidence) -> str:
    """Render the report as one self-contained HTML document.

    Lints before rendering: the four section data objects' `prose_fragments()`
    plus every figure's caption are concatenated (with section names, in the
    same order the body below renders them) and handed to
    `require_clean_prose` before any markup is produced. A caption is
    caller-supplied prose exactly like an interpretation or a limitation
    statement -- it reaches the reader as a `<figcaption>` -- so it is linted
    as its own fragment per figure, tagged with the "Figures" section name,
    rather than folded into one of the other four sections or skipped.
    """
    assumption = assumption_table_data(evidence.checks)
    validity = validity_band_data(evidence.score)
    comparison = comparison_data(evidence.estimates)
    limitations = limitations_data(evidence.score, evidence.limitations)

    fragments: list[tuple[str, str]] = []
    leading_sections: tuple[tuple[str, _SectionData], ...] = (
        ("Assumption Checks", assumption),
        ("Validity Band", validity),
    )
    for section_name, data in leading_sections:
        fragments.extend((section_name, text) for text in data.prose_fragments())
    fragments.extend(("Figures", figure.caption) for figure in evidence.figures)
    trailing_sections: tuple[tuple[str, _SectionData], ...] = (
        ("Cross-Method Comparison", comparison),
        ("Limitations", limitations),
    )
    for section_name, data in trailing_sections:
        fragments.extend((section_name, text) for text in data.prose_fragments())
    require_clean_prose(fragments)

    body = "".join(
        [
            render_assumption_table_html(assumption),
            render_validity_band_html(validity),
            _render_figures_html(evidence.figures),
            render_comparison_html(comparison),
            render_limitations_html(limitations),
        ]
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>Causal Pulse Report</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>Causal Pulse Report</h1>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )
