r"""GitHub-flavoured Markdown renderers for the report's four sections.

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

**A literal `|` in a table cell.** `AssumptionCheck.name`, `.interpretation`
and `.remedy`, and `EffectEstimate.method`, all arrive from a caller's
dataset or estimator configuration -- a covariate name, free text, a
registry name -- and none of them is guaranteed pipe-free. An unescaped `|`
inside a GFM table cell is read as a new column delimiter, not as a literal
character, so a single stray pipe in a covariate name silently shifts every
cell after it into the wrong column for the rest of that row. `_escape_pipe`
backslash-escapes it, the same choice `_escape_markdown_caption` makes for
brackets below: CommonMark's table extension (which GitHub Markdown
follows) honours a backslash-escaped pipe inside a cell as a literal
character rather than a delimiter, so this needs no second syntax (a
reference-style cell, an HTML entity) to keep in sync with the rest of this
module's one-line-per-row style. `kind`, `status`, `estimand` and `basis`
are excluded -- each is drawn from a fixed, already-linted vocabulary that
cannot contain a pipe, so escaping them would be a no-op dressed up as a
guard.

**A literal newline in the same fields.** A raw `\n` is the same hazard in a
worse shape: GFM ends a table row at the newline, so a multi-line
`interpretation` or `remedy` does not shift cells like a stray pipe does --
it silently terminates the row early and drops every cell after the break
(the Remedy column, most often), with no error and no visible sign in the
source. `_escape_pipe` collapses all whitespace runs (newlines included) to
a single space before escaping the pipe, via `" ".join(text.split())` --
chosen over a `<br>` or a backslash-newline escape because a table cell has
no room for a second visual line in this renderer's one-line-per-row style,
and collapsing is the same fix regardless of which whitespace character
caused it (`\n`, `\r\n`, a stray tab), so there is nothing further to special
-case.

**Escaping order.** `_escape_pipe` and `_escape_markdown_caption` both
backslash-escape a fixed set of characters by replacing each in turn. That
is only correct if the backslash itself is escaped *first*: escaping `|`
after inserting a backslash the escaping itself introduced can turn one
inserted backslash plus a pre-existing backslash into a pair GFM reads as an
escaped-backslash-then-live-delimiter, undoing the very escape being
performed. A source value containing `\|` becomes `\\|` under a naive
single-pass substitution -- one escaped (literal) backslash followed by an
unescaped, still-live pipe. Escaping `\` to `\\` before escaping `|` (or a
bracket) avoids this: the same input then becomes `\\\|`, which GFM reads as
an escaped backslash followed by an escaped pipe, both literal.
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
    document_prose_fragments,
    limitations_data,
    validity_band_data,
)

_ABSENT = "—"  # em dash: a `None` statistic, threshold or remedy.
_DOCUMENT_NAME = "report.md"


def _escape_pipe(text: str) -> str:
    """Neutralise everything in `text` that could corrupt a GFM table row.

    Three steps, in the order the module docstring's "A literal `|`",
    "literal newline" and "Escaping order" sections explain: collapse every
    whitespace run (a `\\n` included) to a single space, so a multi-line
    value cannot terminate the row early and drop the cells after it;
    backslash-escape a literal `\\` first, so an already-escaped source value
    is not silently unescaped by the next step; then backslash-escape a
    literal `|` so it cannot be read as a column delimiter.
    """
    collapsed = " ".join(text.split())
    escaped_backslash = collapsed.replace("\\", "\\\\")
    return escaped_backslash.replace("|", "\\|")


def _cell(value: object) -> str:
    """Render one table cell, keeping `None` visibly distinct from `0.00`.

    The generic branch also escapes a literal `|` (via `_escape_pipe`) --
    every non-float, non-`None` value this function ever receives is a
    string (`remedy`), and a `remedy` is exactly the kind of caller-supplied
    free text the pipe hazard exists for. A formatted float can never
    contain `|`, so escaping only the generic branch is not a coincidence
    that could bit-rot: it is the only branch a string ever reaches.
    """
    if value is None:
        return _ABSENT
    if isinstance(value, float):
        return f"{value:.2f}"
    return _escape_pipe(str(value))


def render_assumption_table_markdown(data: AssumptionTableData) -> str:
    """Render the assumption checks as a heading, a summary line and a table.

    An input with no checks at all (`assumption_table_data(())`) renders an
    explicit "No assumption checks were run." sentence instead of a
    0/0/0/0 summary sitting over a table with a header row and no data
    rows -- a reader must be able to tell "nothing to report" apart from "the
    section silently lost every row", and an all-zero tally under a table
    that still looks like a table is not that unambiguous on its own.
    """
    if not data.names:
        return "\n".join(["## Assumption Checks", "", "No assumption checks were run."])

    header = "| Check | Kind | Status | Statistic | Threshold | Interpretation | Remedy |"
    separator = "| --- | --- | --- | --- | --- | --- | --- |"
    rows = [
        f"| {_escape_pipe(name)} | {kind} | {status} | {_cell(statistic)} | {_cell(threshold)} | "
        f"{_escape_pipe(interpretation)} | {_cell(remedy)} |"
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
    """Render the cross-method comparison as a heading and a table, one row per estimate.

    An input with no estimates at all (`comparison_data([])`) renders an
    explicit "No estimates were available for comparison." sentence instead
    of a table with a header row and no data rows -- unlike the assumption
    table, this section carries no summary counts to fall back on, so a bare
    empty table here has nothing at all telling a reader the row count was
    ever checked, rather than silently lost.
    """
    if not data.methods:
        return "\n".join(
            ["## Cross-Method Comparison", "", "No estimates were available for comparison."]
        )

    header = "| Method | Estimand | Point | Lower | Upper | Basis |"
    separator = "| --- | --- | --- | --- | --- | --- |"
    rows = [
        f"| {_escape_pipe(method)} | {estimand} | {point:.2f} | {lower:.2f} | {upper:.2f} | "
        f"{basis} |"
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

    The backslash itself is escaped first, before any bracket -- see the
    module docstring's "Escaping order" section for why looping over the
    brackets alone, without that first pass, can unescape a caption that
    already contains a literal `\\` immediately before one.
    """
    caption = caption.replace("\\", "\\\\")
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

    Lints before writing anything. `require_clean_prose` is called on
    `cp_report.sections.document_prose_fragments(evidence)` -- every prose
    fragment a report built from `evidence` would show, tagged with the
    section it came from, in the same order the document below assembles
    them -- before this function touches the filesystem at all: a refusal
    must never leave a half-written directory behind, which is what
    `test_a_refused_report_writes_nothing` pins. See
    `document_prose_fragments`'s docstring for what it includes and why a
    caption is linted as its own per-figure fragment. This module and
    `cp_report.html` each carry their own call to `require_clean_prose`
    rather than sharing one, so a guard removed from one format cannot mask
    a gap in the other. Only after the lint call returns does this function
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

    require_clean_prose(document_prose_fragments(evidence))

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
