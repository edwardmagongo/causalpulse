"""The report's figure carrier.

`ReportFigure` holds a caption and a draw callable, not a pre-rendered
`Axes`, so `cp_report.figures` controls figure size and output format
rather than screen-scraping an `Axes` someone else already built. Phase
16 section 10 anticipated exactly this split: every `cp_figures` figure is
a pure data function plus a drawing function that takes an `Axes` and
returns it, never creating a `Figure` or touching global pyplot state --
which is what lets `cp_report.figures` reuse those data functions directly
instead of reconstructing what they already computed.

`Evidence` is the container `cp_report.html.render_html_report` and
`cp_report.markdown.write_markdown_report` each consume as their one
argument. It carries exactly what those two entry points need to build all
four sections plus the figures: `estimates` for the comparison table,
`score` for the validity band (and, read as `score.limits`, for the
limitations section's leading sentence), `checks` for the assumption table,
`limitations` for the caller's own limitation statements, and `figures` for
the `ReportFigure`s to embed. It is frozen for the same reason every other
dataclass in this package is: a report entry point is not a place a caller
should be able to mutate evidence into after it has been assembled.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cp_core.results import AssumptionCheck, CausalValidityScore, EffectEstimate
from matplotlib.axes import Axes


@dataclass(frozen=True)
class ReportFigure:
    """A caption plus a callable that draws the figure onto a caller's `Axes`.

    `draw` matches every `cp_figures` drawing function's signature -- it
    takes an `Axes`, draws onto it, and returns it -- so a `ReportFigure` is
    typically built as `ReportFigure(caption="...", draw=partial(draw_x,
    data))`, reusing a `cp_figures` function rather than wrapping it.
    """

    caption: str
    draw: Callable[[Axes], Axes]


@dataclass(frozen=True)
class Evidence:
    """Everything the report's two entry points need, gathered in one place.

    `estimates` feeds `cp_report.sections.comparison_data`, `checks` feeds
    `assumption_table_data`, `score` feeds `validity_band_data` and (via
    `score.limits`) the limitations section, and `limitations` supplies the
    caller's own statements to `limitations_data`. `figures` is rendered
    directly by each entry point (inlined as SVG for HTML, written as files
    and linked for Markdown) rather than through a `sections.py` data
    function, since figures are not one of that module's four table/prose
    sections.

    **`checks` and `score.checks` are two independent fields, and nothing in
    this package reconciles them.** The assumption table renders from
    `checks` -- the argument on this dataclass. The validity band's rationale,
    ceiling and band were computed upstream from `score.checks` -- the tuple
    carried on the `CausalValidityScore` itself ("the table behind" the band,
    per that class's own docstring). A caller who passes a `checks` tuple
    that does not match the checks `score` was actually computed from gets a
    report whose two sections describe different check sets with no error
    and no warning: an assumption table with, say, two rows sitting directly
    above a validity band whose band and rationale were derived from a
    different, one-check table. Keeping the two in agreement is the caller's
    responsibility -- typically by passing the same `checks` tuple used to
    build `score` in the first place.
    """

    estimates: tuple[EffectEstimate, ...]
    score: CausalValidityScore
    checks: tuple[AssumptionCheck, ...]
    limitations: tuple[str, ...]
    figures: tuple[ReportFigure, ...]
