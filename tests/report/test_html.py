"""Tests for the HTML renderers.

Mirrors `test_markdown.py`'s structural assertions -- a row per estimate, a
status string appearing verbatim, the limits text being present -- with two
additions specific to HTML.

First, escaping: every interpolated value can come from a dataset or a
caller, so `test_an_interpolated_value_is_escaped` pins that markup-bearing
input reaches the document only in its escaped form, never as raw markup
(and never simply dropped -- both directions are checked, the same shape
`test_markdown.py` uses for the absent-vs-zero em dash).

Second, the estimand/method pairing. Asserting all four of "ipw", "ATE",
"twfe", "ATT" appear somewhere in the rendered output cannot catch a
renderer that mispairs them -- e.g. printing "ipw (ATT)" and "twfe (ATE)",
the exact defect Phase 16 shipped in its forest plot. So
`test_the_comparison_renders_the_estimand_beside_every_method` parses the
rendered `<tr>` rows out of the table body and asserts, per row and in
order, that the row containing "ipw" also contains "ATE" and the row
containing "twfe" also contains "ATT" -- a mispairing fails this test even
though every individual string is still present in the document somewhere.
"""

from __future__ import annotations

import dataclasses
import re

import pytest
from cp_core.results import VALIDITY_SCORE_LIMITS
from cp_report.html import (
    render_assumption_table_html,
    render_comparison_html,
    render_html_report,
    render_limitations_html,
    render_validity_band_html,
)
from cp_report.language import ReportLanguageError
from cp_report.sections import (
    assumption_table_data,
    comparison_data,
    limitations_data,
    validity_band_data,
)

from report.builders import _check, _estimate, _score

_ROW_RE = re.compile(r"<tr>.*?</tr>", re.DOTALL)


def _body_rows(rendered: str) -> list[str]:
    """Return every `<tr>...</tr>` row after the header row."""
    return _ROW_RE.findall(rendered)[1:]


def test_the_assumption_table_renders_a_row_per_check():
    data = assumption_table_data((_check("overlap"), _check("balance", kind="estimation")))

    rendered = render_assumption_table_html(data)

    for name in data.names:
        assert name in rendered


def test_a_not_testable_row_renders_its_status_verbatim():
    data = assumption_table_data(
        (_check("unconfoundedness", status="NOT_TESTABLE", statistic=None, threshold=None),)
    )

    assert "NOT_TESTABLE" in render_assumption_table_html(data)


def test_an_interpolated_value_is_escaped():
    # Covariate names come from a dataset. One containing markup must not
    # reach the document as markup.
    data = assumption_table_data((_check(name="age<5 & over"),))

    rendered = render_assumption_table_html(data)

    assert "age&lt;5 &amp; over" in rendered
    assert "age<5 & over" not in rendered


def test_the_validity_band_never_renders_without_its_limits():
    rendered = render_validity_band_html(validity_band_data(_score()))

    assert VALIDITY_SCORE_LIMITS in rendered


def test_the_comparison_renders_the_estimand_beside_every_method():
    data = comparison_data([_estimate("ipw", "ATE"), _estimate("twfe", "ATT")])

    rendered = render_comparison_html(data)

    # Order-sensitive, per-row: string presence alone
    # cannot catch a mispairing such as "ipw (ATT)" / "twfe (ATE)" -- the
    # exact defect Phase 16 shipped in its forest plot. Parse the table
    # body rows and check the pairing directly.
    rows = _body_rows(rendered)
    assert len(rows) == 2
    ipw_row = next(row for row in rows if "ipw" in row)
    twfe_row = next(row for row in rows if "twfe" in row)
    assert "ATE" in ipw_row
    assert "ATT" not in ipw_row
    assert "ATT" in twfe_row
    assert "ATE" not in twfe_row


def test_the_comparison_renders_an_interval_for_every_row():
    data = comparison_data([_estimate(point=0.25, lo=0.05, hi=0.45)])

    rendered = render_comparison_html(data)

    assert "0.05" in rendered
    assert "0.45" in rendered


def test_a_missing_statistic_renders_as_absent_not_as_zero():
    data = assumption_table_data(
        (_check("unconfoundedness", status="NOT_TESTABLE", statistic=None, threshold=None),)
    )

    rendered = render_assumption_table_html(data)

    assert "0.00" not in rendered
    assert "—" in rendered


def test_a_present_statistic_renders_with_its_value_not_an_em_dash():
    data = assumption_table_data((_check("overlap", statistic=0.04, threshold=0.1),))

    rendered = render_assumption_table_html(data)

    assert "0.04" in rendered
    assert "0.10" in rendered


def test_a_checks_remedy_renders_when_present():
    data = assumption_table_data(
        (_check(status="FAIL", remedy="Re-estimate propensity scores with more covariates."),)
    )

    rendered = render_assumption_table_html(data)

    assert "Re-estimate propensity scores with more covariates." in rendered


def test_the_assumption_table_renders_its_counts():
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

    rendered = render_assumption_table_html(data)

    assert f"Passed: {data.n_passed}" in rendered
    assert f"Warned: {data.n_warned}" in rendered
    assert f"Failed: {data.n_failed}" in rendered
    assert f"Not testable: {data.n_not_testable}" in rendered


def test_the_validity_band_renders_its_band_and_ceiling():
    data = validity_band_data(_score(band="WEAK", ceiling="MODERATE"))

    rendered = render_validity_band_html(data)

    assert "WEAK" in rendered
    assert "MODERATE" in rendered


def test_the_validity_band_renders_its_rationale():
    data = validity_band_data(_score())

    rendered = render_validity_band_html(data)

    for reason in data.rationale:
        assert reason in rendered


def test_the_limitations_section_renders_its_limits():
    data = limitations_data(_score(), ("The donor pool is small.",))

    rendered = render_limitations_html(data)

    assert VALIDITY_SCORE_LIMITS in rendered


def test_the_limitations_section_renders_every_statement():
    data = limitations_data(_score(), ("The donor pool is small.", "The window is short."))

    rendered = render_limitations_html(data)

    assert "The donor pool is small." in rendered
    assert "The window is short." in rendered


def test_a_limitation_statement_is_escaped():
    # Limitation statements are free text a caller supplied; they can
    # contain markup that must not reach the document unescaped.
    data = limitations_data(_score(), ("Coverage is thin for age<5 & over.",))

    rendered = render_limitations_html(data)

    assert "age&lt;5 &amp; over" in rendered
    assert "age<5 & over" not in rendered


def test_the_html_report_is_one_self_contained_document(evidence):
    rendered = render_html_report(evidence)

    assert rendered.lstrip().startswith("<!DOCTYPE html>")
    assert "</html>" in rendered
    assert "<svg" in rendered  # the figure is inlined, not linked
    assert "<img" not in rendered


def test_the_html_report_carries_all_four_sections(evidence):
    rendered = render_html_report(evidence)

    for heading in ("Assumption", "Validity", "Comparison", "Limitations"):
        assert heading in rendered


def test_the_html_report_refuses_prose_that_fails_the_lint(evidence):
    bad = dataclasses.replace(evidence, limitations=("The rollout causes the lift.",))

    with pytest.raises(ReportLanguageError):
        render_html_report(bad)
