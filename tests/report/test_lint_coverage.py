"""Proves `require_clean_prose` is reached from every report entry point,
for every kind of prose that reaches the reader -- not merely defined and
imported somewhere in `cp_report`.

Task 2 built the refusal. A refusal enforced in one place only protects a
reader if every caller actually reaches it, and a caller that built the
right `Evidence`, discarded the lint's return value, and rendered anyway
would pass any test that only checks the string `require_clean_prose`
appears in a function's source -- Phase 16's equivalent task shipped exactly
that source-text form first and replaced it with a behavioural one, because
a discarded call satisfies a source check but never protects a reader. Every
test below instead builds an `Evidence` whose prose is known to fail the
lint and asserts the entry point raises `ReportLanguageError` -- provable
only by observing the entry point's actual behaviour, not its source.

`prose_fragments()` on `cp_report.sections`'s four dataclasses names six
distinct things that end up in front of a reader: a caller's own limitation
statement, an `AssumptionCheck.interpretation`, an `AssumptionCheck.remedy`,
a `CausalValidityScore.rationale` entry, an `AssumptionCheck.name` (often a
covariate name pulled from a dataset), and an `EffectEstimate.method` (a
name pulled from an estimator registry). The last two are identifiers
rather than authored sentences, but their own docstrings say they were
included deliberately because nothing had ever linted them before Phase 17
-- a caller's dataset column named for a settled result (`proven_uplift`)
reaches the table exactly as read as a caller's free-text limitation does.
A seventh source sits outside `cp_report.sections` entirely:
`ReportFigure.caption`, caller-supplied prose that reaches the reader as a
`<figcaption>` in HTML and as an image's alt text in Markdown. Task 9's
implementer shipped the other six and missed this one -- the entry points
built their fragment list from the four sections' `prose_fragments()` and
never looked at `evidence.figures` at all, so a caption could carry an
unhedged causal claim straight past the lint. A generator that linted only
the six `sections.py`-derived sources and skipped the caption would still
pass a test that only exercised those six, so all seven are covered here,
parametrised against both entry points, for a 2x7 matrix that makes the
coverage explicit rather than implied by a handful of hand-picked cases.

The final test is the belt-and-braces check spec Sections 5 and 7 both call
for: every prose source above is linted *before* rendering, as a fragment
list built independently of the document a renderer produces -- so a
renderer whose own template wording, or whose composition of otherwise-
clean fragments, introduced a fresh violation would never be caught by
that pre-render pass. Markdown is chosen deliberately over HTML for this
one: `causal_language_lint`'s clause splitter (`--|[,;:()–—]`)
treats HTML markup's own attribute punctuation as sentence and clause
boundaries, so linting markup would invent boundaries a plain-text lint
was never calibrated against. Markdown's headings and table pipes do not
collide with the tokeniser the same way.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest
from cp_core.language import causal_language_lint
from cp_report.evidence import Evidence
from cp_report.html import render_html_report
from cp_report.language import ReportLanguageError
from cp_report.markdown import write_markdown_report

# A single sentence, reused everywhere a source needs to fail the lint --
# it names no section-specific vocabulary, so any test using it is pinned
# to *this* prose source alone, not incidentally saved or sunk by wording
# picked for some other section. Taken verbatim from tests/report/
# test_language.py, which already establishes it fails on its own.
_UNHEDGED_CLAIM = "This proves unconfoundedness."

# A realistic stand-in for the identifier-shaped sources: a dataset column
# or estimator name that happens to collide with a banned word once split
# on its underscore, mirroring test_language.py's
# test_interpolated_text_is_linted_too.
_TAINTED_IDENTIFIER = "proven_uplift"


def _bad_limitation(evidence: Evidence) -> Evidence:
    """Corrupts the caller's own limitation statement."""
    return dataclasses.replace(evidence, limitations=(_UNHEDGED_CLAIM,))


def _bad_check_interpretation(evidence: Evidence) -> Evidence:
    """Corrupts one `AssumptionCheck.interpretation`."""
    bad_check = dataclasses.replace(evidence.checks[0], interpretation=_UNHEDGED_CLAIM)
    return dataclasses.replace(evidence, checks=(bad_check, *evidence.checks[1:]))


def _bad_check_remedy(evidence: Evidence) -> Evidence:
    """Corrupts one `AssumptionCheck.remedy`."""
    bad_check = dataclasses.replace(evidence.checks[0], status="FAIL", remedy=_UNHEDGED_CLAIM)
    return dataclasses.replace(evidence, checks=(bad_check, *evidence.checks[1:]))


def _bad_score_rationale(evidence: Evidence) -> Evidence:
    """Corrupts the `CausalValidityScore.rationale`."""
    bad_score = dataclasses.replace(evidence.score, rationale=(_UNHEDGED_CLAIM,))
    return dataclasses.replace(evidence, score=bad_score)


def _bad_check_name(evidence: Evidence) -> Evidence:
    """Corrupts one `AssumptionCheck.name`, as a dataset-derived covariate would."""
    bad_check = dataclasses.replace(evidence.checks[0], name=_TAINTED_IDENTIFIER)
    return dataclasses.replace(evidence, checks=(bad_check, *evidence.checks[1:]))


def _bad_estimate_method(evidence: Evidence) -> Evidence:
    """Corrupts one `EffectEstimate.method`, as a registry-derived name would."""
    bad_estimate = dataclasses.replace(evidence.estimates[0], method=_TAINTED_IDENTIFIER)
    return dataclasses.replace(evidence, estimates=(bad_estimate, *evidence.estimates[1:]))


def _bad_figure_caption(evidence: Evidence) -> Evidence:
    """Corrupts one `ReportFigure.caption`, as a careless caller's prose would."""
    bad_figure = dataclasses.replace(evidence.figures[0], caption=_UNHEDGED_CLAIM)
    return dataclasses.replace(evidence, figures=(bad_figure, *evidence.figures[1:]))


_PROSE_SOURCES: dict[str, Callable[[Evidence], Evidence]] = {
    "caller limitation": _bad_limitation,
    "check interpretation": _bad_check_interpretation,
    "check remedy": _bad_check_remedy,
    "score rationale": _bad_score_rationale,
    "check name": _bad_check_name,
    "estimate method": _bad_estimate_method,
    "figure caption": _bad_figure_caption,
}


def _call_html(evidence: Evidence, directory: Path) -> object:
    return render_html_report(evidence)


def _call_markdown(evidence: Evidence, directory: Path) -> object:
    return write_markdown_report(evidence, directory)


_ENTRY_POINTS: dict[str, Callable[[Evidence, Path], object]] = {
    "render_html_report": _call_html,
    "write_markdown_report": _call_markdown,
}


@pytest.mark.parametrize("prose_source", sorted(_PROSE_SOURCES))
@pytest.mark.parametrize("entry_point", sorted(_ENTRY_POINTS))
def test_every_entry_point_refuses_every_prose_source(
    entry_point, prose_source, evidence, tmp_path
):
    corrupt = _PROSE_SOURCES[prose_source]
    call = _ENTRY_POINTS[entry_point]
    bad_evidence = corrupt(evidence)

    with pytest.raises(ReportLanguageError):
        call(bad_evidence, tmp_path)


@pytest.mark.parametrize("prose_source", sorted(_PROSE_SOURCES))
def test_a_refused_markdown_report_writes_nothing(prose_source, evidence, tmp_path):
    # Mirrors test_markdown.py's `test_a_refused_report_writes_nothing`, but
    # parametrised across every prose source in `_PROSE_SOURCES` -- including
    # "figure caption" -- rather than one hand-picked limitation. The lint
    # must run, and the refusal must land, before `write_markdown_report`
    # makes any filesystem contact regardless of which fragment failed,
    # matching the ordering `write_markdown_report`'s own docstring commits
    # to: lint first, figure files and document only after.
    corrupt = _PROSE_SOURCES[prose_source]
    bad_evidence = corrupt(evidence)

    with pytest.raises(ReportLanguageError):
        write_markdown_report(bad_evidence, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_a_fully_rendered_markdown_report_passes_the_lint(evidence, tmp_path):
    # The generator lints its sections' prose, not its rendered output --
    # markup would feed tag soup to a sentence splitter tuned on plain text.
    # That leaves one gap: a violation introduced by a renderer's own
    # template wording, or by its composition of otherwise-clean fragments
    # into a document with a different adjacency than the pre-render
    # fragment list has. The corpus scan in tests/core/test_language.py
    # already covers the renderer's own string constants, since they are
    # string literals under packages/. This closes the gap from the other
    # side, on real rendered output, for content that already cleared the
    # pre-render lint above.
    path = write_markdown_report(evidence, tmp_path)

    assert causal_language_lint(path.read_text()) == ()
