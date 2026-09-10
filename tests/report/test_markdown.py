"""Tests for the Markdown renderers.

Each renderer is asserted on structure -- a row per estimate, a status
string appearing verbatim, the limits text being present -- never on a
golden blob of rendered text. A test asserting the whole document equals a
fixed string breaks on every wording change and teaches a maintainer to
regenerate it without reading it.

`test_a_missing_statistic_renders_as_absent_not_as_zero` is the one worth
the most care: this project has already shipped a benchmark heatmap that
rendered an uncomputed cell as `0.00`. The renderer here chooses an em dash
(`—`) for a `None` statistic, threshold or remedy, and the test below
pins that choice directly rather than only proving `0.00` is absent -- a
renderer that rendered `None` as the string `"None"` would also dodge the
`0.00` assertion, so both must be checked.
"""

from __future__ import annotations

import dataclasses
import re

import pytest
from cp_core.results import VALIDITY_SCORE_LIMITS
from cp_report.language import ReportLanguageError
from cp_report.markdown import (
    render_assumption_table_markdown,
    render_comparison_markdown,
    render_limitations_markdown,
    render_validity_band_markdown,
    write_markdown_report,
)
from cp_report.sections import (
    assumption_table_data,
    comparison_data,
    limitations_data,
    validity_band_data,
)

from report.builders import _check, _estimate, _score


def test_the_assumption_table_renders_a_row_per_check():
    data = assumption_table_data((_check("overlap"), _check("balance", kind="estimation")))

    rendered = render_assumption_table_markdown(data)

    for name in data.names:
        assert name in rendered


def test_a_not_testable_row_renders_its_status_verbatim():
    # Folding this into a pass rate would assert what the data cannot support.
    data = assumption_table_data(
        (_check("unconfoundedness", status="NOT_TESTABLE", statistic=None, threshold=None),)
    )

    assert "NOT_TESTABLE" in render_assumption_table_markdown(data)


def test_the_validity_band_never_renders_without_its_limits():
    rendered = render_validity_band_markdown(validity_band_data(_score()))

    assert VALIDITY_SCORE_LIMITS in rendered


def test_the_comparison_renders_the_estimand_beside_every_method():
    data = comparison_data([_estimate("ipw", "ATE"), _estimate("twfe", "ATT")])

    rendered = render_comparison_markdown(data)

    rows = [line for line in rendered.splitlines() if "ipw" in line or "twfe" in line]
    assert len(rows) == 2
    assert "ATE" in next(r for r in rows if "ipw" in r)
    assert "ATT" in next(r for r in rows if "twfe" in r)


def test_the_comparison_renders_an_interval_for_every_row():
    data = comparison_data([_estimate(point=0.25, lo=0.05, hi=0.45)])

    rendered = render_comparison_markdown(data)

    assert "0.05" in rendered
    assert "0.45" in rendered


def test_a_missing_statistic_renders_as_absent_not_as_zero():
    # A check with no statistic did not score zero. Rendering it as 0.00 would
    # be the heatmap's absent-versus-zero hazard in a table.
    data = assumption_table_data(
        (_check("unconfoundedness", status="NOT_TESTABLE", statistic=None, threshold=None),)
    )

    rendered = render_assumption_table_markdown(data)

    assert "0.00" not in rendered
    # The specific choice: a `None` statistic/threshold/remedy renders as an
    # em dash, not as the word "None" or a blank cell that could be misread
    # as a formatting accident.
    assert "—" in rendered


def test_a_present_statistic_renders_with_its_value_not_an_em_dash():
    # The em-dash choice must not swallow a real, computed value -- pin both
    # directions the way `test_a_not_testable_checks_statistic_and_threshold_
    # stay_none` pins None-preservation in test_sections.py.
    data = assumption_table_data((_check("overlap", statistic=0.04, threshold=0.1),))

    rendered = render_assumption_table_markdown(data)

    assert "0.04" in rendered
    assert "0.10" in rendered


def test_a_checks_remedy_renders_when_present():
    data = assumption_table_data(
        (_check(status="FAIL", remedy="Re-estimate propensity scores with more covariates."),)
    )

    rendered = render_assumption_table_markdown(data)

    assert "Re-estimate propensity scores with more covariates." in rendered


def test_the_assumption_table_renders_its_counts():
    # Distinct counts, and each assertion ties a label to its own number:
    # with every count equal to 1, or a bare `str(count) in rendered`, this
    # would pass even if the summary line were deleted outright, since a
    # digit like "1" also turns up in unrelated cells (e.g. the default
    # threshold 0.10).
    data = assumption_table_data(
        (
            _check("a", status="PASS"),
            _check("b", status="PASS"),
            _check("c", status="WARN"),
            _check("d", status="FAIL"),
            _check("e", status="FAIL"),
            _check("f", status="FAIL"),
            _check("g", status="NOT_TESTABLE", statistic=None, threshold=None),
        )
    )
    assert (data.n_passed, data.n_warned, data.n_failed, data.n_not_testable) == (2, 1, 3, 1)

    rendered = render_assumption_table_markdown(data)

    # Each assertion pairs the label with its own number in one substring, so
    # a stray digit elsewhere in the table (a threshold, a count from another
    # field) cannot satisfy it by accident.
    assert f"Passed: {data.n_passed}" in rendered
    assert f"Warned: {data.n_warned}" in rendered
    assert f"Failed: {data.n_failed}" in rendered
    assert f"Not testable: {data.n_not_testable}" in rendered


def test_the_validity_band_renders_its_band_and_ceiling():
    data = validity_band_data(_score(band="WEAK", ceiling="MODERATE"))

    rendered = render_validity_band_markdown(data)

    assert "WEAK" in rendered
    assert "MODERATE" in rendered


def test_the_validity_band_renders_its_rationale():
    data = validity_band_data(_score())

    rendered = render_validity_band_markdown(data)

    for reason in data.rationale:
        assert reason in rendered


def test_the_limitations_section_renders_its_limits():
    data = limitations_data(_score(), ("The donor pool is small.",))

    rendered = render_limitations_markdown(data)

    assert VALIDITY_SCORE_LIMITS in rendered


def test_the_limitations_section_renders_every_statement():
    data = limitations_data(_score(), ("The donor pool is small.", "The window is short."))

    rendered = render_limitations_markdown(data)

    assert "The donor pool is small." in rendered
    assert "The window is short." in rendered


def test_the_markdown_report_carries_all_four_sections(evidence, tmp_path):
    # Mirrors test_html.py's test_the_html_report_carries_all_four_sections.
    # Deleting a render_..._markdown call from write_markdown_report's
    # document body would pass every other test in this file -- the fragment
    # list that feeds the lint is built independently of the document body
    # -- so this is the one test that would catch a vanished section.
    path = write_markdown_report(evidence, tmp_path)

    text = path.read_text()
    for heading in ("Assumption", "Validity", "Comparison", "Limitations"):
        assert heading in text


def test_the_figures_section_is_omitted_when_there_are_no_figures(evidence, tmp_path):
    bare = dataclasses.replace(evidence, figures=())

    path = write_markdown_report(bare, tmp_path)

    assert "Figures" not in path.read_text()


def test_the_figures_section_appears_when_there_is_a_figure(evidence, tmp_path):
    path = write_markdown_report(evidence, tmp_path)

    assert "## Figures" in path.read_text()


def test_a_caption_needing_escape_still_produces_a_well_formed_link(evidence, tmp_path):
    # "]" would close the `[caption]` bracket early and ")" risks the same
    # for `(file)`; a renderer that left either raw would truncate the
    # caption or point the link at the wrong target.
    caption = "Effects [robust] and (unadjusted) estimates."
    figure = dataclasses.replace(evidence.figures[0], caption=caption)
    escaped = dataclasses.replace(evidence, figures=(figure,))

    path = write_markdown_report(escaped, tmp_path)
    text = path.read_text()

    # A parser that honours backslash escapes (CommonMark's own rule, which
    # GitHub Markdown follows) sees one link whose caption is the whole
    # original text and whose target is a real file -- not a caption
    # truncated at the first unescaped "]".
    match = re.search(r"!\[((?:\\.|[^\]])*)\]\(([^)]+)\)", text)
    assert match, "expected one well-formed image link"
    rendered_caption, target = match.groups()
    assert rendered_caption == caption.replace("]", "\\]").replace(")", "\\)")
    assert (path.parent / target).exists()


def test_the_markdown_report_writes_its_figures_and_links_them(evidence, tmp_path):
    path = write_markdown_report(evidence, tmp_path)

    text = path.read_text()
    written = [p.name for p in tmp_path.iterdir() if p != path]
    assert written, "a report with a figure must write that figure"
    for name in written:
        assert name in text, f"{name} was written but never linked"


def test_a_markdown_link_resolves_relative_to_the_document(evidence, tmp_path):
    # A link that only works from the caller's cwd is a broken link.
    path = write_markdown_report(evidence, tmp_path)

    targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", path.read_text())
    # The regex alone proves nothing if it matches zero links -- the fixture
    # carries one figure, so a genuinely linked report must produce at least
    # one match. Without this, a renderer using a different link syntax
    # (e.g. a bare "path" with no "![...]()" wrapper) would make the loop
    # below execute zero times and the test would pass vacuously.
    assert targets, "expected at least one markdown image link"
    for target in targets:
        assert (path.parent / target).exists(), target


def test_the_markdown_report_refuses_prose_that_fails_the_lint(evidence, tmp_path):
    bad = dataclasses.replace(evidence, limitations=("The rollout causes the lift.",))

    with pytest.raises(ReportLanguageError):
        write_markdown_report(bad, tmp_path)


def test_a_refused_report_writes_nothing(evidence, tmp_path):
    # A half-written directory after a refusal is worse than no directory.
    bad = dataclasses.replace(evidence, limitations=("The rollout causes the lift.",))

    with pytest.raises(ReportLanguageError):
        write_markdown_report(bad, tmp_path)

    assert list(tmp_path.iterdir()) == []
