"""Tests for `cp_report.benchmark`'s prose assembly, language-lint refusal and renderers.

The lint tests mirror `test_lint_coverage.py`: every source of caller prose, at both entry
points independently, because each entry point carries its own `require_clean_prose` call.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cp_core.language import causal_language_lint
from cp_report.benchmark import (
    CATE_NOTE,
    COVERAGE_COLUMN_NOTE,
    BenchmarkEvidence,
    CateSectionData,
    benchmark_evidence_from_run,
    document_prose_fragments,
    render_html_benchmark_report,
    write_markdown_benchmark_report,
)
from cp_report.language import ReportLanguageError

from report.benchmark_builders import make_benchmark_evidence

_UNHEDGED_CLAIM = "This proves unconfoundedness."


def test_document_prose_fragments_walk_every_section_and_every_caption(benchmark_evidence):
    fragments = document_prose_fragments(benchmark_evidence)

    assert {section for section, _ in fragments} == {
        "Headline Findings",
        "Figures",
        "Method Comparisons",
        "Limitations",
    }
    assert ("Figures", "bias by method and scenario.") in fragments
    assert ("Method Comparisons", COVERAGE_COLUMN_NOTE) in fragments
    assert ("Limitations", "hierarchical_did is excluded from scenario P6.") in fragments


def _bad_limitation(e: BenchmarkEvidence) -> BenchmarkEvidence:
    return dataclasses.replace(e, limitations=(_UNHEDGED_CLAIM,))


def _bad_statement(e: BenchmarkEvidence) -> BenchmarkEvidence:
    bad = dataclasses.replace(e.expectations[0], statement=_UNHEDGED_CLAIM)
    return dataclasses.replace(e, expectations=(bad, *e.expectations[1:]))


def _bad_note(e: BenchmarkEvidence) -> BenchmarkEvidence:
    bad = dataclasses.replace(e.expectations[1], note=_UNHEDGED_CLAIM)
    return dataclasses.replace(e, expectations=(e.expectations[0], bad, *e.expectations[2:]))


def _bad_narrative(e: BenchmarkEvidence) -> BenchmarkEvidence:
    return dataclasses.replace(e, narratives={"E1": _UNHEDGED_CLAIM})


def _bad_exploratory_note(e: BenchmarkEvidence) -> BenchmarkEvidence:
    return dataclasses.replace(e, exploratory_note=_UNHEDGED_CLAIM)


def _bad_caption(e: BenchmarkEvidence) -> BenchmarkEvidence:
    bad = dataclasses.replace(e.figures[0], caption=_UNHEDGED_CLAIM)
    return dataclasses.replace(e, figures=(bad, *e.figures[1:]))


_PROSE_SOURCES: dict[str, Callable[[BenchmarkEvidence], BenchmarkEvidence]] = {
    "caller limitation": _bad_limitation,
    "expectation statement": _bad_statement,
    "expectation note": _bad_note,
    "caller narrative": _bad_narrative,
    "exploratory note": _bad_exploratory_note,
    "figure caption": _bad_caption,
}

_ENTRY_POINTS: dict[str, Callable[[BenchmarkEvidence, Path], object]] = {
    "render_html_benchmark_report": lambda e, d: render_html_benchmark_report(e),
    "write_markdown_benchmark_report": write_markdown_benchmark_report,
}


@pytest.mark.parametrize("prose_source", sorted(_PROSE_SOURCES))
@pytest.mark.parametrize("entry_point", sorted(_ENTRY_POINTS))
def test_every_entry_point_refuses_every_prose_source(
    entry_point, prose_source, benchmark_evidence, tmp_path
):
    bad = _PROSE_SOURCES[prose_source](benchmark_evidence)

    with pytest.raises(ReportLanguageError):
        _ENTRY_POINTS[entry_point](bad, tmp_path)


@pytest.mark.parametrize("prose_source", sorted(_PROSE_SOURCES))
def test_a_refused_markdown_report_writes_nothing(prose_source, benchmark_evidence, tmp_path):
    bad = _PROSE_SOURCES[prose_source](benchmark_evidence)

    with pytest.raises(ReportLanguageError):
        write_markdown_benchmark_report(bad, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_markdown_refuses_a_directory_that_does_not_exist(benchmark_evidence, tmp_path):
    with pytest.raises(FileNotFoundError):
        write_markdown_benchmark_report(benchmark_evidence, tmp_path / "missing")


def test_a_fully_rendered_markdown_report_passes_the_lint(benchmark_evidence, tmp_path):
    path = write_markdown_benchmark_report(benchmark_evidence, tmp_path)

    assert causal_language_lint(path.read_text()) == ()


def test_markdown_report_has_every_section_the_note_and_one_svg_per_figure(
    benchmark_evidence, tmp_path
):
    path = write_markdown_benchmark_report(benchmark_evidence, tmp_path)
    text = path.read_text()

    assert text.startswith("# CausalPulse Benchmark Study Report")
    for heading in ("## Overview", "## Headline Findings", "## Figures", "## Method Comparisons",
                    "## Limitations"):
        assert heading in text
    assert COVERAGE_COLUMN_NOTE in text
    assert len(list(tmp_path.glob("*.svg"))) == len(benchmark_evidence.figures)


def test_a_pipe_or_newline_in_a_narrative_cannot_break_the_markdown_table(
    benchmark_evidence, tmp_path
):
    evidence = dataclasses.replace(
        benchmark_evidence, narratives={"E1": "Two | cells\nin one paragraph."}
    )

    text = write_markdown_benchmark_report(evidence, tmp_path).read_text()

    row = next(line for line in text.splitlines() if line.startswith("| E1 "))
    assert row.endswith("Two \\| cells in one paragraph. |")


def test_html_report_shows_the_tally_the_note_and_inline_svg(benchmark_evidence):
    html = render_html_benchmark_report(benchmark_evidence)

    assert "<h1>CausalPulse Benchmark Study Report</h1>" in html
    assert "Met: 1" in html and "Failed: 1" in html and "Inconclusive: 1" in html
    assert escape(COVERAGE_COLUMN_NOTE) in html  # the note has an apostrophe
    assert "<svg" in html


def test_html_report_escapes_caller_text(benchmark_evidence):
    evidence = dataclasses.replace(benchmark_evidence, limitations=("Compare <a> & <b>.",))

    html = render_html_benchmark_report(evidence)

    assert "Compare &lt;a&gt; &amp; &lt;b&gt;." in html


def test_both_formats_render_from_a_real_tiny_study(d_run, tmp_path):
    evidence = benchmark_evidence_from_run(
        d_run, limitations=("A stand-in limitation for this tiny fixture.",)
    )

    html = render_html_benchmark_report(evidence)
    out = tmp_path / "out"
    out.mkdir()
    path = write_markdown_benchmark_report(evidence, out)

    assert "<h1>CausalPulse Benchmark Study Report</h1>" in html
    assert path.read_text().startswith("# CausalPulse Benchmark Study Report")
    assert len(list(out.glob("*.svg"))) == len(evidence.figures) == 2


def _with_cate(evidence):
    table = pd.DataFrame(
        {
            "method": ["causal_forest", "causal_forest"],
            "scenario": ["NULL", "RICH"],
            "cate_correlation": [np.nan, 0.8123],
            "cate_correlation_mcse": [np.nan, 0.0101],
            "cate_correlation_n_used": [0, 50],
            "cate_rmse": [0.0391, 0.0842],
            "cate_rmse_mcse": [0.0012, 0.0015],
            "cate_rmse_n_used": [50, 50],
            "cate_calibration_slope": [np.nan, 1.7412],
            "cate_calibration_slope_mcse": [np.nan, 0.0333],
            "cate_calibration_slope_n_used": [0, 50],
        }
    )
    section = CateSectionData(
        table=table, replications=50, cells_done=150, cells_expected=150, note=CATE_NOTE
    )
    return dataclasses.replace(evidence, cate_section=section)


def test_markdown_carries_the_cate_section_between_comparisons_and_limitations(tmp_path):
    path = write_markdown_benchmark_report(_with_cate(make_benchmark_evidence()), tmp_path)
    text = path.read_text()

    assert "## Heterogeneous Effects (CATE)" in text
    assert text.index("## Method Comparisons") < text.index("## Heterogeneous Effects (CATE)")
    assert text.index("## Heterogeneous Effects (CATE)") < text.index("## Limitations")
    assert "1.741" in text
    assert "50 replications" in text
    assert CATE_NOTE in text


def test_html_carries_the_cate_section():
    html = render_html_benchmark_report(_with_cate(make_benchmark_evidence()))

    assert "<h2>Heterogeneous Effects (CATE)</h2>" in html
    assert html.index("Method Comparisons") < html.index("Heterogeneous Effects (CATE)")
    assert html.index("Heterogeneous Effects (CATE)") < html.index("Limitations")
    assert "1.741" in html


def test_no_cate_section_renders_nothing_extra(tmp_path):
    path = write_markdown_benchmark_report(make_benchmark_evidence(), tmp_path)

    assert "Heterogeneous Effects" not in path.read_text()
    assert "Heterogeneous Effects" not in render_html_benchmark_report(make_benchmark_evidence())


def test_the_cate_note_is_linted_with_the_document():
    evidence = _with_cate(make_benchmark_evidence())
    fragments = document_prose_fragments(evidence)

    assert ("Heterogeneous Effects (CATE)", CATE_NOTE) in fragments

    bad = dataclasses.replace(
        evidence,
        cate_section=dataclasses.replace(evidence.cate_section, note="This proves it works."),
    )
    with pytest.raises(ReportLanguageError):
        render_html_benchmark_report(bad)
