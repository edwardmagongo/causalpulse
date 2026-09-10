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

# The four `prose_fragments()`-bearing dataclasses `write_markdown_report`
# lints around the figures section -- named so the two grouped tuples below
# share one annotation instead of mypy narrowing each to just the two
# variants its own literal happens to contain.
_SectionData = AssumptionTableData | ValidityBandData | ComparisonData | LimitationsData

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


def _escape_markdown_caption(caption: str) -> str:
    """Neutralise every bracket that would break `![caption](file)`.

    GitHub-flavoured Markdown has no general escaping mechanism inside link
    text the way HTML has `html.escape` -- but a backslash defangs a single
    character, and CommonMark (which GitHub Markdown follows) honours that
    inside link text.

    All four brackets need it, and `[` is the one that is easy to miss.
    A literal `]` closes the `[caption]` bracket early, truncating the
    caption; a literal `)` risks the same for `(file)`. But a literal `[`
    is worse than either, because the renderer always appends one genuine
    unescaped `]` right before `(file)`: CommonMark's bracket matching
    pairs the caption's stray `[` with *that* `]`, turning the tail of the
    caption into a nested link and leaving the outer `![` as literal text.
    A caption reading "CI [0.05, 0.45] for the ATE." -- ordinary in this
    domain -- renders as stray `![CI` followed by an unrelated hyperlink,
    with no image at all. Verified against a CommonMark parser rather than
    reasoned about.

    Backslash-escaping is chosen over a reference-style link
    (`[caption][ref]` plus a `[ref]: file` definition elsewhere) because it
    keeps this renderer a one-line format like every other line in this
    module, with no second, distant piece of syntax to keep in sync.
    """
    for bracket in "[]()":
        caption = caption.replace(bracket, "\\" + bracket)
    return caption


def _render_figures_markdown(figures: tuple[ReportFigure, ...], paths: list[Path]) -> str:
    """Render the figures section: a heading plus one image link per figure.

    Each link is `![caption](filename)` with a bare filename, not a path --
    `write_markdown_report` writes both the figure files and the document
    itself into the same directory, so a bare filename resolves relative to
    the document regardless of the caller's own working directory.

    Returns the empty string when there are no figures, rather than a bare
    "## Figures" heading with nothing under it -- a figure-less report is a
    legal input `Evidence` anticipates, and an empty section is a blemish a
    publication-quality document should not carry. Checking `len(figures)`
    is a presence check on data the caller already has, not a computation.
    """
    if not figures:
        return ""
    links = "\n".join(
        f"![{_escape_markdown_caption(figure.caption)}]({path.name})"
        for figure, path in zip(figures, paths, strict=True)
    )
    return "\n".join(["## Figures", "", links])


def write_markdown_report(evidence: Evidence, directory: Path) -> Path:
    """Write the report as a directory: one Markdown document plus its figures.

    Lints before writing anything. The four section data objects'
    `prose_fragments()` plus every figure's caption are concatenated (with
    section names, in the same order the document below assembles them) and
    handed to `require_clean_prose` before this function touches the
    filesystem at all -- a refusal must never leave a half-written directory
    behind, which is what `test_a_refused_report_writes_nothing` pins. A
    caption is caller-supplied prose exactly like an interpretation or a
    limitation statement -- it reaches the reader as the alt text of an
    embedded image -- so it is linted as its own fragment per figure, tagged
    with the "Figures" section name, before any figure file or the document
    itself is written. Only after the lint call returns does this function
    write the figure files (via `figure_to_file`) and then the document, in
    that order.

    `directory` must already exist. This function never creates it --
    silently `mkdir`-ing would change behaviour a caller may be relying on
    (e.g. a permissions boundary, or an intentional refusal to write into a
    path that was never set up). Given a missing directory it raises
    `FileNotFoundError` naming that directory, rather than letting the
    confusing error `figure_to_file`'s `savefig` would otherwise raise --
    one naming a `.svg` file, never mentioning that the real problem is the
    directory above it.
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

    if not directory.is_dir():
        raise FileNotFoundError(
            f"write_markdown_report requires an existing directory, got {directory!r}"
        )

    figure_paths = [figure_to_file(figure, directory) for figure in evidence.figures]

    document = "\n\n".join(
        part
        for part in [
            "# Causal Pulse Report",
            render_assumption_table_markdown(assumption),
            render_validity_band_markdown(validity),
            _render_figures_markdown(evidence.figures, figure_paths),
            render_comparison_markdown(comparison),
            render_limitations_markdown(limitations),
        ]
        if part
    )
    path = directory / _DOCUMENT_NAME
    path.write_text(document)
    return path
