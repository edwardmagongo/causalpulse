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

from cp_report.sections import (
    AssumptionTableData,
    ComparisonData,
    LimitationsData,
    ValidityBandData,
)

_ABSENT = "—"  # em dash: a `None` statistic, threshold or remedy.


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
