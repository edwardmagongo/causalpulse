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

from cp_report.sections import (
    AssumptionTableData,
    ComparisonData,
    LimitationsData,
    ValidityBandData,
)

_ABSENT = "—"  # em dash: a `None` statistic, threshold or remedy.


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
