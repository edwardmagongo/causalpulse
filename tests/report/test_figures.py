"""Tests for rendering a `ReportFigure` as inline SVG or as a file.

`figure_to_inline_svg` and `figure_to_file` both create their own `Figure`,
call the caller's `draw`, save, and close the figure in a `finally` -- see
`figures.py`'s docstring for why both hazards (an unstripped XML preamble
mid-document, a leaked open figure) are load-bearing rather than defensive
gold-plating.

Filenames are derived from the caption: a slug for readability plus a short
hash of the caption text for uniqueness. A naive slug alone would collapse
two captions that differ only in punctuation onto the same filename and
silently overwrite the first file with the second --
`test_captions_differing_only_in_punctuation_do_not_collide` pins that this
does not happen, since the given `test_two_figures_do_not_collide_on_one_filename`
uses two captions ("One."/"Two.") whose slugs already differ and so would
pass even with a naive implementation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from cp_report.evidence import ReportFigure
from cp_report.figures import figure_to_file, figure_to_inline_svg


def _figure():
    def draw(ax):
        ax.plot([0, 1], [0, 1])
        return ax

    return ReportFigure(caption="A line.", draw=draw)


def test_the_inline_svg_has_no_xml_declaration_or_doctype():
    # An unstripped <?xml ...?> mid-body is invalid HTML. Measured on
    # matplotlib 3.11.1: savefig emits both by default.
    svg = figure_to_inline_svg(_figure())

    assert not svg.lstrip().startswith("<?xml")
    assert "<!DOCTYPE" not in svg
    assert svg.lstrip().startswith("<svg")


def test_the_inline_svg_carries_the_drawing():
    # A blank axes -- no line, no label, nothing drawn -- still serialises to
    # 11,630 bytes on matplotlib 3.11.1 (ticks, spines, styling all emit
    # markup), so neither "<svg" presence nor a length floor can tell a real
    # drawing apart from an empty one. Measured directly by mutating `_draw`
    # to run the caller's `draw` onto a detached axes: every assertion above
    # this comment still passed.
    #
    # A distinctive `ylabel` is a real lower bound instead. matplotlib 3.11.1
    # emits label text verbatim inside an XML comment (`<!-- cp-report-probe
    # -->`) immediately before the glyph paths that render it, even though
    # `svg.fonttype` defaults to "path" (each character drawn as a path, not
    # written out as an SVG <text> node) -- confirmed directly against this
    # module's real output, not assumed from the rcParam's name. So the probe
    # string can only appear in the saved SVG if `draw` actually ran on the
    # axes that got saved.
    def draw(ax):
        ax.plot([0, 1], [0, 1])
        ax.set_ylabel("cp-report-probe-label")
        return ax

    svg = figure_to_inline_svg(ReportFigure(caption="A line.", draw=draw))

    assert "<svg" in svg and "</svg>" in svg
    assert "cp-report-probe-label" in svg


def test_rendering_a_figure_leaves_no_open_figures():
    # matplotlib warns about too many open figures and the suite treats a
    # warning as a failure, so a generator that leaks one breaks a later test.
    before = plt.get_fignums()

    figure_to_inline_svg(_figure())

    assert plt.get_fignums() == before


def test_a_figure_file_is_written_and_its_path_returned(tmp_path):
    path = figure_to_file(_figure(), tmp_path)

    assert path.exists()
    assert path.parent == tmp_path
    assert path.stat().st_size > 0


def test_writing_a_figure_file_also_leaves_no_open_figures(tmp_path):
    before = plt.get_fignums()

    figure_to_file(_figure(), tmp_path)

    assert plt.get_fignums() == before


def test_a_figure_file_keeps_its_xml_declaration_and_doctype(tmp_path):
    # Unlike the inline fragment, a standalone .svg file is a whole document
    # and should stay a valid one -- stripping is an inline-embedding concern
    # only, not something figure_to_file needs to (or should) do.
    path = figure_to_file(_figure(), tmp_path)

    written = path.read_text()
    assert written.lstrip().startswith("<?xml")
    assert "<!DOCTYPE" in written


def test_two_figures_do_not_collide_on_one_filename(tmp_path):
    first = figure_to_file(ReportFigure(caption="One.", draw=_figure().draw), tmp_path)
    second = figure_to_file(ReportFigure(caption="Two.", draw=_figure().draw), tmp_path)

    assert first != second


def _raising_figure():
    def draw(ax):
        raise ValueError("drawing failed")

    return ReportFigure(caption="Broken.", draw=draw)


def test_a_draw_that_raises_still_propagates_and_closes_for_inline_svg():
    before = plt.get_fignums()

    with pytest.raises(ValueError, match="drawing failed"):
        figure_to_inline_svg(_raising_figure())

    assert plt.get_fignums() == before


def test_a_draw_that_raises_still_propagates_and_closes_for_file(tmp_path):
    before = plt.get_fignums()

    with pytest.raises(ValueError, match="drawing failed"):
        figure_to_file(_raising_figure(), tmp_path)

    assert plt.get_fignums() == before


def test_captions_differing_only_in_punctuation_do_not_collide(tmp_path):
    # A naive slug (lowercase, strip punctuation) maps both of these captions
    # to "effect-estimates", which would make the second write silently
    # overwrite the first.
    first = figure_to_file(
        ReportFigure(caption="Effect estimates.", draw=_figure().draw), tmp_path
    )
    second = figure_to_file(
        ReportFigure(caption="Effect estimates!", draw=_figure().draw), tmp_path
    )

    assert first != second
    assert first.exists()
    assert second.exists()


def test_a_figure_renders_to_identical_bytes_every_time(tmp_path):
    # matplotlib stamps an SVG with the wall-clock time and derives element ids from a random
    # salt, so a report regenerated from the same data would differ in every figure and churn
    # a committed copy in git. Both are pinned; the report's own claim to be regeneratable
    # rests on this.
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    path_one = figure_to_file(_figure(), first)
    path_two = figure_to_file(_figure(), second)

    assert path_one.name == path_two.name
    assert path_one.read_bytes() == path_two.read_bytes()
    assert figure_to_inline_svg(_figure()) == figure_to_inline_svg(_figure())
