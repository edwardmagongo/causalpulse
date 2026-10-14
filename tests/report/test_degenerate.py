"""Degenerate inputs: the shapes a real analysis produces and a happy-path
test never sees.

One degenerate case is required per section, decided one of two
ways -- a clear refusal naming the problem, or a correct, non-empty
rendering -- never a section that quietly renders nothing while the rest of
the document reports success. Phase 16's equivalent task found three real
silent-success bugs by asking exactly this question of `cp_figures`; this
task asks it of `cp_report` and found two of the same shape in the
assumption table and the cross-method comparison table, both fixed in
`cp_report.markdown` and `cp_report.html` (see those modules' docstrings and
this file's own tests below).

Two cases were carried in from earlier reviews rather than drawn fresh from
the brief:

1. A literal `|` in `AssumptionCheck.name`/`.interpretation`/`.remedy` or
   `EffectEstimate.method` -- all caller- or dataset-supplied free text --
   used to reach a GFM table cell unescaped, where it reads as an extra
   column delimiter and shifts every cell after it out of place for the
   rest of that row. `test_a_pipe_in_a_check_name_does_not_corrupt_the_
   markdown_table` and its comparison-table sibling pin the fix
   (`cp_report.markdown._escape_pipe`) by splitting the rendered row on an
   *unescaped* pipe and checking the column count survives. HTML has no
   equivalent exposure -- `html.escape` never touches `|`, so a pipe simply
   passes through as one ordinary character inside a `<td>`, never a
   delimiter -- which `test_a_pipe_in_free_text_is_ordinary_text_in_html`
   confirms rather than assumes.

2. The three empty-list paths (`assumption_table_data(())`,
   `comparison_data([])`, `limitations_data(score, ())`) were untested.
   `limitations_data` already renders correctly with no source change --
   `LimitationsData.limits` is never the caller's to omit, so the section
   always leads with real content regardless of `statements`. The other two
   did not: both rendered a heading over an empty table (the assumption
   table at least had an honest 0/0/0/0 summary line; the comparison table
   had nothing at all distinguishing "zero rows on purpose" from "the
   renderer silently lost every row"). Both now render an explicit sentence
   in place of the empty table -- the fix this file exists to pin.
"""

from __future__ import annotations

import dataclasses
import re

from cp_core.results import VALIDITY_SCORE_LIMITS
from cp_report.evidence import Evidence
from cp_report.html import (
    render_assumption_table_html,
    render_comparison_html,
    render_html_report,
    render_limitations_html,
)
from cp_report.markdown import (
    render_assumption_table_markdown,
    render_comparison_markdown,
    render_limitations_markdown,
    write_markdown_report,
)
from cp_report.sections import assumption_table_data, comparison_data, limitations_data

from report.builders import _check, _estimate, _score

_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def _cells(row: str) -> list[str]:
    """Split a rendered table row the way a GFM parser would: on an
    unescaped `|` only, never on a backslash-escaped one."""
    return [cell.strip() for cell in _UNESCAPED_PIPE.split(row)]


# ---------------------------------------------------------------------------
# A single estimate, and both entry points, must still render an interval.
# ---------------------------------------------------------------------------


def test_a_single_estimate_still_renders_its_interval():
    data = comparison_data([_estimate(point=0.25, lo=0.05, hi=0.45)])

    rendered = render_comparison_markdown(data)

    assert "0.05" in rendered
    assert "0.45" in rendered
    # Exactly one data row -- a renderer that degenerated to a header-only
    # table for a one-element input would still pass a bare substring check.
    assert len(rendered.splitlines()) == 5  # heading, blank, header, separator, one row


def test_a_single_estimate_renders_through_both_entry_points(tmp_path):
    single = Evidence(
        estimates=(_estimate(point=0.25, lo=0.05, hi=0.45),),
        score=_score(),
        checks=(_check(),),
        limitations=(),
        figures=(),
    )

    path = write_markdown_report(single, tmp_path)
    markdown_text = path.read_text()
    html_text = render_html_report(single)

    for rendered in (markdown_text, html_text):
        assert "0.05" in rendered
        assert "0.45" in rendered


# ---------------------------------------------------------------------------
# An empty limitations tuple: the score's own limits are not the caller's to
# omit, so the section must still render real content.
# ---------------------------------------------------------------------------


def test_an_empty_limitations_tuple_still_renders_the_scores_own_limits():
    data = limitations_data(_score(), ())

    markdown_rendered = render_limitations_markdown(data)
    html_rendered = render_limitations_html(data)

    assert VALIDITY_SCORE_LIMITS in markdown_rendered
    assert VALIDITY_SCORE_LIMITS in html_rendered
    # No empty bullet list for the caller's own (absent) statements -- pin
    # the exact structure so a stray "<ul></ul>" or a dangling "- " line
    # cannot slip in unnoticed.
    assert markdown_rendered == "\n".join(["## Limitations", "", VALIDITY_SCORE_LIMITS])
    assert "<ul>" not in html_rendered


def test_an_empty_limitations_tuple_still_renders_through_both_entry_points(evidence, tmp_path):
    bare = dataclasses.replace(evidence, limitations=())

    path = write_markdown_report(bare, tmp_path)
    html_text = render_html_report(bare)

    assert VALIDITY_SCORE_LIMITS in path.read_text()
    assert VALIDITY_SCORE_LIMITS in html_text


# ---------------------------------------------------------------------------
# An assumption table entirely NOT_TESTABLE: n_passed is zero, and the
# section must not read as a failure.
# ---------------------------------------------------------------------------


def test_an_all_not_testable_assumption_table_does_not_read_as_a_failure():
    data = assumption_table_data(
        (
            _check("unconfoundedness", status="NOT_TESTABLE", statistic=None, threshold=None),
            _check("parallel_trends", status="NOT_TESTABLE", statistic=None, threshold=None),
        )
    )

    assert (data.n_passed, data.n_warned, data.n_failed, data.n_not_testable) == (0, 0, 0, 2)
    assert data.statuses == ("NOT_TESTABLE", "NOT_TESTABLE")

    rendered = render_assumption_table_markdown(data)

    assert "Passed: 0" in rendered
    assert "Failed: 0" in rendered
    assert "Not testable: 2" in rendered
    # Every row's own status column reads NOT_TESTABLE, never PASS or FAIL --
    # a table listing two NOT_TESTABLE rows must not carry either verdict.
    rows = [line for line in rendered.splitlines() if line.startswith("| unconfoundedness")]
    assert rows and " FAIL " not in rows[0] and " PASS " not in rows[0]


# ---------------------------------------------------------------------------
# An empty assumption table: source fix -- render an explicit sentence
# rather than an all-zero summary sitting over a header-only table.
# ---------------------------------------------------------------------------


def test_an_empty_assumption_table_states_explicitly_that_none_were_run():
    data = assumption_table_data(())

    assert (data.n_passed, data.n_warned, data.n_failed, data.n_not_testable) == (0, 0, 0, 0)

    markdown_rendered = render_assumption_table_markdown(data)
    html_rendered = render_assumption_table_html(data)

    assert markdown_rendered == "\n".join(
        ["## Assumption Checks", "", "No assumption checks were run."]
    )
    assert "No assumption checks were run." in html_rendered
    # No empty table shell left behind either -- a heading plus a lone
    # header row is exactly the silent-empty-section shape this fix removes.
    assert "| Check |" not in markdown_rendered
    assert "<table>" not in html_rendered


# ---------------------------------------------------------------------------
# comparison_data([]): the controller's second carried-in case. Same
# decision and same fix as the assumption table, for the same reason -- but
# this table has no summary counts to fall back on, so the pre-fix render
# had nothing at all telling a reader the row count was checked.
# ---------------------------------------------------------------------------


def test_an_empty_comparison_table_states_explicitly_that_none_were_compared():
    data = comparison_data([])

    assert data.methods == ()

    markdown_rendered = render_comparison_markdown(data)
    html_rendered = render_comparison_html(data)

    assert markdown_rendered == "\n".join(
        ["## Cross-Method Comparison", "", "No estimates were available for comparison."]
    )
    assert "No estimates were available for comparison." in html_rendered
    assert "| Method |" not in markdown_rendered
    assert "<table>" not in html_rendered


# ---------------------------------------------------------------------------
# A report with no figures: no figures heading, and Markdown writes no
# files beside the document itself.
# ---------------------------------------------------------------------------


def test_a_report_with_no_figures_writes_only_the_document(evidence, tmp_path):
    bare = dataclasses.replace(evidence, figures=())

    path = write_markdown_report(bare, tmp_path)

    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    assert "Figures" not in path.read_text()


def test_a_report_with_no_figures_renders_no_figures_heading_in_html(evidence):
    bare = dataclasses.replace(evidence, figures=())

    rendered = render_html_report(bare)

    assert "<h2>Figures</h2>" not in rendered


# ---------------------------------------------------------------------------
# A zero-width interval: a real, if unusual, statement about uncertainty
# that must render, matching how require_interval already treats it (it
# refuses only a missing interval, never a narrow one).
# ---------------------------------------------------------------------------


def test_a_zero_width_interval_renders_rather_than_refuses():
    # point (0.30) differs from the bounds (0.25) so the row's three numeric
    # columns are distinguishable -- a renderer that accidentally repeated
    # the point into the bound columns would still pass a test where all
    # three happened to share one value.
    data = comparison_data([_estimate(point=0.30, lo=0.25, hi=0.25)])

    assert data.lower == (0.25,)
    assert data.upper == (0.25,)

    markdown_rendered = render_comparison_markdown(data)
    html_rendered = render_comparison_html(data)

    assert "0.30 | 0.25 | 0.25" in markdown_rendered
    assert "<td>0.30</td><td>0.25</td><td>0.25</td>" in html_rendered


def test_a_zero_width_interval_renders_through_both_entry_points(evidence, tmp_path):
    narrow = dataclasses.replace(
        evidence, estimates=(_estimate(point=0.30, lo=0.25, hi=0.25),)
    )

    path = write_markdown_report(narrow, tmp_path)
    html_text = render_html_report(narrow)

    assert "0.30 | 0.25 | 0.25" in path.read_text()
    assert "<td>0.30</td><td>0.25</td><td>0.25</td>" in html_text


# ---------------------------------------------------------------------------
# The controller's first carried-in case: a "|" in free text must not
# corrupt a Markdown table. Markdown-only -- HTML escapes, so "|" is inert
# there (confirmed below rather than assumed).
# ---------------------------------------------------------------------------


def test_a_pipe_in_a_check_name_does_not_corrupt_the_markdown_table():
    check = _check(
        name="a|b",
        interpretation="income|region interacted",
        status="FAIL",
        remedy="Add income|region as a covariate.",
    )
    data = assumption_table_data((check,))

    rendered = render_assumption_table_markdown(data)
    row = next(line for line in rendered.splitlines() if line.startswith("| a"))
    cells = _cells(row)

    # 7 columns -> 8 delimiting pipes -> 9 pieces once split on the
    # *unescaped* ones only. An implementation that left the pipes raw would
    # instead split into more pieces here, one extra per raw "|".
    assert len(cells) == 9, cells
    assert cells[1] == "a\\|b"
    assert cells[6] == "income\\|region interacted"
    assert cells[7] == "Add income\\|region as a covariate."
    # And the literal pipe genuinely was the value's own content, not
    # coincidentally already escaped upstream -- the source field really did
    # contain a bare "|".
    assert "|" in check.name and "|" in check.interpretation and "|" in (check.remedy or "")


def test_a_pipe_in_a_comparison_method_name_does_not_corrupt_the_markdown_table():
    data = comparison_data([_estimate(method="method|variant")])

    rendered = render_comparison_markdown(data)
    row = next(line for line in rendered.splitlines() if line.startswith("| method"))
    cells = _cells(row)

    # 6 columns -> 7 delimiting pipes -> 8 pieces.
    assert len(cells) == 8, cells
    assert cells[1] == "method\\|variant"


def test_a_pipe_in_free_text_is_ordinary_text_in_html():
    # HTML has no table-delimiter syntax for "|" to hijack -- html.escape
    # never touches it, so it must pass through as one literal character
    # inside its own <td>, not split anything.
    check = _check(name="a|b")
    data = assumption_table_data((check,))

    rendered = render_assumption_table_html(data)

    assert "<td>a|b</td>" in rendered


# ---------------------------------------------------------------------------
# The other half of the same defect class: a raw newline in free text splits
# a GFM row exactly like an unescaped "|" does, and it is worse -- the row
# does not merely misalign, a trailing cell (here, Remedy) disappears
# entirely and the remainder becomes stray text outside the table.
# ---------------------------------------------------------------------------


def test_a_newline_in_free_text_does_not_drop_a_table_cell():
    check = _check(
        name="overlap",
        interpretation="Balance is adequate.\nSee the appendix for detail.",
        status="FAIL",
        remedy="Add\nan interaction term.",
    )
    data = assumption_table_data((check,))

    rendered = render_assumption_table_markdown(data)
    # If the newline were not neutralised, the row itself would split across
    # multiple lines and this lookup (and the row count below) would already
    # betray it.
    row = next(line for line in rendered.splitlines() if line.startswith("| overlap"))
    cells = _cells(row)

    # Same 7-column shape as the pipe test above: 9 pieces once split on
    # unescaped pipes. A dropped Remedy cell would show up here as 8.
    assert len(cells) == 9, cells
    assert cells[6] == "Balance is adequate. See the appendix for detail."
    assert cells[7] == "Add an interaction term."
    # The row itself is one line -- the document as a whole still has plenty
    # of newlines between its other lines, so the check is scoped to `row`.
    assert "\n" not in row


def test_a_backslash_immediately_before_a_pipe_does_not_unescape_it():
    # A naive single-pass "|" -> "\|" substitution turns a source text that
    # already contains "\|" into "\\|" -- read by GFM as an escaped literal
    # backslash followed by a *live*, unescaped pipe delimiter. Escaping the
    # backslash first closes this: "\|" becomes "\\\|", which GFM reads as an
    # escaped backslash plus an escaped pipe, both literal.
    check = _check(name="a\\|b")
    data = assumption_table_data((check,))

    rendered = render_assumption_table_markdown(data)
    row = next(line for line in rendered.splitlines() if line.startswith("| a"))
    cells = _cells(row)

    assert len(cells) == 9, cells
    assert cells[1] == "a\\\\\\|b"


# ---------------------------------------------------------------------------
# Everything degenerate at once, through both entry points: proves the two
# fixes above compose correctly inside write_markdown_report and
# render_html_report, not only in the standalone renderer functions.
# ---------------------------------------------------------------------------


def test_every_section_degenerate_at_once_still_produces_a_complete_report(tmp_path):
    empty_everywhere = Evidence(
        estimates=(),
        score=_score(checks=()),  # matches checks=() below -- see Evidence's docstring
        checks=(),
        limitations=(),
        figures=(),
    )

    path = write_markdown_report(empty_everywhere, tmp_path)
    markdown_text = path.read_text()
    html_text = render_html_report(empty_everywhere)

    for rendered in (markdown_text, html_text):
        assert "No assumption checks were run." in rendered
        assert "No estimates were available for comparison." in rendered
        assert VALIDITY_SCORE_LIMITS in rendered
        assert "Figures" not in rendered

    assert [p.name for p in tmp_path.iterdir()] == [path.name]
