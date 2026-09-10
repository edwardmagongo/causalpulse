"""Render a `ReportFigure` as inline SVG or as a file on disk.

**The backend.** `cp_figures/__init__.py` selects the `Agg` backend at
import, and `cp_report/__init__.py` imports nothing that pulls it in --
it is a bare docstring plus `__version__`. Task 1's review flagged this as
harmless only because `cp_report` happened to always import `cp_figures`
first. That stopped being true here: this module is the first in
`cp_report` to call `savefig` directly, and it can be imported without
`cp_figures` ever loading -- `tests/report/test_figures.py` imports only
`cp_report.evidence` and `cp_report.figures`, and under `pytest -n auto`
that test module can land in a worker that has imported neither
`cp_report.sections` nor `cp_figures` before it runs. Measured directly:
importing `cp_report` alone leaves the interactive `macosx` backend
selected on this machine, and only importing something that reaches
`cp_figures` (e.g. `cp_report.sections`, which imports
`cp_figures.guards`) switches it to `Agg`. Relying on that transitive path
would make this module's behaviour depend on which other test happened to
run first in the same worker. So this module selects `Agg` itself, exactly
as `cp_figures` does, before `pyplot` is imported.

**The two hazards this module exists to close, both measured on
matplotlib 3.11.1 under `Agg`:**

1. `fig.savefig(buf, format="svg")` is warning-free under
   `filterwarnings = ["error"]`, but its output begins with an XML
   declaration and a DOCTYPE before the `<svg>` tag. Both are valid on a
   standalone `.svg` file -- `figure_to_file` keeps them -- but invalid
   mid-document, so `figure_to_inline_svg` strips everything before the
   first `<svg`.
2. matplotlib warns "More than 20 figures have been opened" once a leak
   accumulates, and the suite turns that warning into a failure far from
   whichever test leaked it. Both rendering functions below create exactly
   one `Figure` via `pyplot` (so it is tracked in `pyplot`'s figure
   registry) and then immediately enter a `try` whose `finally` closes it.
   Everything that could raise -- adding the axes, the caller's own `draw`,
   and `savefig` -- runs inside that `try`, so none of them can leave the
   figure open. The `plt.figure()` call itself sits just outside, because a
   bare `plt.figure()` has nothing to raise and putting it inside would
   leave `finally` with no figure to close.

**Filenames.** `figure_to_file` derives a filename from the caption: a
lowercase slug for readability, plus a short hex digest of the caption
text for uniqueness. The digest is necessary, not decorative -- a slug
alone collapses "Effect estimates." and "Effect estimates!" onto the same
"effect-estimates" stem, and the second write would silently overwrite the
first. Two `ReportFigure`s sharing one literal caption still collide; that
is accepted as a caller's naming choice rather than a case this module
guards against.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from cp_report.evidence import ReportFigure  # noqa: E402

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(caption: str) -> str:
    slug = _SLUG_RE.sub("-", caption.lower()).strip("-")
    return slug or "figure"


def _filename(caption: str) -> str:
    digest = hashlib.sha256(caption.encode("utf-8")).hexdigest()[:10]
    return f"{_slugify(caption)}-{digest}.svg"


def _strip_xml_preamble(svg: str) -> str:
    """Drop everything before the opening `<svg` tag.

    savefig's DOCTYPE line never itself contains the four characters
    `<svg` (its content is `-//W3C//DTD SVG 1.1//EN` and a `.dtd` URL), so
    the first occurrence in the document is always the real opening tag.
    """
    return svg[svg.index("<svg") :]


def _draw(figure: ReportFigure, fig: Figure) -> None:
    ax = fig.add_subplot()
    figure.draw(ax)


def figure_to_inline_svg(figure: ReportFigure) -> str:
    """Render `figure` to an SVG fragment safe to inline into an HTML body."""
    fig = plt.figure()  # a bare plt.figure() essentially cannot raise
    try:
        _draw(figure, fig)
        buffer = io.StringIO()
        fig.savefig(buffer, format="svg")
        return _strip_xml_preamble(buffer.getvalue())
    finally:
        plt.close(fig)


def figure_to_file(figure: ReportFigure, directory: Path) -> Path:
    """Render `figure` to a standalone `.svg` file under `directory`.

    The file keeps its XML declaration and DOCTYPE -- unlike the inline
    fragment, it is a whole document, not something embedded into another
    one.
    """
    fig = plt.figure()  # a bare plt.figure() essentially cannot raise
    try:
        _draw(figure, fig)
        path = directory / _filename(figure.caption)
        fig.savefig(path, format="svg")
        return path
    finally:
        plt.close(fig)
