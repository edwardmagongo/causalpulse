import json

import pytest
from cp_cli.__main__ import main
from cp_evaluation.store import IncompleteRunError, RunStore

from cli._tiny_study import use_tiny_study_tier


@pytest.fixture
def run(tmp_path, monkeypatch):
    use_tiny_study_tier(monkeypatch)
    assert main(["bench", "run", "--tier", "study", "--out", str(tmp_path / "run")]) == 0
    return tmp_path / "run"


def _report(run, out, *extra):
    return main(["bench", "study-report", str(run), "--out", str(out), *extra])


def test_markdown_is_the_default_and_skips_a_profile_whose_scenario_the_run_lacks(
    run, tmp_path, capsys
):
    out = tmp_path / "out"
    out.mkdir()

    assert _report(run, out) == 0

    text = (out / "benchmark-report.md").read_text()
    assert text.startswith("# CausalPulse Benchmark Study Report")
    # The tiny grid has scenarios A and P1 only: the coverage heatmap, and no scenario-D
    # profile (skipped, not refused).
    assert len(list(out.glob("*.svg"))) == 1
    assert "wrote the benchmark study report to" in capsys.readouterr().out


def test_html_format_writes_one_self_contained_file(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()

    assert _report(run, out, "--format", "html") == 0

    html = (out / "benchmark-report.html").read_text()
    assert "<h1>CausalPulse Benchmark Study Report</h1>" in html
    assert list(out.glob("*.svg")) == []


def test_repeated_limitations_and_a_narratives_file_reach_the_document(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    narratives = tmp_path / "narratives.json"
    narratives.write_text(
        json.dumps({"E1": "A reader-facing paragraph for the first expectation."})
    )

    _report(
        run,
        out,
        "--limitation",
        "First stated limitation.",
        "--limitation",
        "Second stated limitation.",
        "--narratives",
        str(narratives),
    )

    text = (out / "benchmark-report.md").read_text()
    assert "First stated limitation." in text and "Second stated limitation." in text
    assert "A reader-facing paragraph for the first expectation." in text


def test_a_narratives_file_naming_an_unknown_expectation_is_refused(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    narratives = tmp_path / "narratives.json"
    narratives.write_text(json.dumps({"E99": "x"}))

    with pytest.raises(ValueError, match="E99"):
        _report(run, out, "--narratives", str(narratives))


def test_a_narratives_file_that_is_not_an_object_of_strings_is_refused(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    narratives = tmp_path / "narratives.json"
    narratives.write_text(json.dumps(["E1"]))

    with pytest.raises(ValueError, match="JSON object"):
        _report(run, out, "--narratives", str(narratives))


@pytest.mark.parametrize("fmt", ["html", "markdown"])
def test_an_output_directory_that_does_not_exist_is_refused_for_both_formats(run, tmp_path, fmt):
    with pytest.raises(FileNotFoundError):
        _report(run, tmp_path / "missing", "--format", fmt)


def test_an_incomplete_run_is_refused(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    victim = sorted(RunStore.open(run).expected_cells())[0]
    (run / "cells" / f"{victim}.parquet").unlink()

    with pytest.raises(IncompleteRunError):
        _report(run, out)

    assert list(out.iterdir()) == []
