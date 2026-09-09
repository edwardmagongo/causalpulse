"""The report's figure carrier.

`ReportFigure` holds a caption and a draw callable, not a pre-rendered
`Axes`, so `cp_report.figures` controls figure size and output format
rather than screen-scraping an `Axes` someone else already built. Phase
16 section 10 anticipated exactly this split: every `cp_figures` figure is
a pure data function plus a drawing function that takes an `Axes` and
returns it, never creating a `Figure` or touching global pyplot state --
which is what lets `cp_report.figures` reuse those data functions directly
instead of reconstructing what they already computed.

`Evidence`, the container Task 8 assembles from a section's data and its
figures, is added there -- it has no reader here yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
