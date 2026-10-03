import json

import pandas as pd
import pytest
from cp_cli.__main__ import main


def _args(out, *extra):
    return ["analyze", "--scenario", "D", "--n-units", "400", "--seed", "0",
            "--out", str(out), *extra]


def test_report_writes_both_formats_per_estimate_and_a_summary(tmp_path):
    out = tmp_path / "results"

    code = main(_args(out, "--report", "--estimators", "difference_in_means,lin_regression"))

    assert code == 0
    for method in ("difference_in_means", "lin_regression"):
        directory = out / "reports" / method
        assert (directory / "report.html").read_text().startswith("<!DOCTYPE html>")
        markdown = (directory / "report.md").read_text()
        for svg in directory.glob("*.svg"):
            assert svg.name in markdown
    summary = json.loads((out / "summary.json").read_text())
    assert summary["failures"] == []
    assert summary["comparisons"][0]["methods"] == ["difference_in_means", "lin_regression"]
    assert list(pd.read_csv(out / "estimates.csv").method) == [
        "difference_in_means", "lin_regression"
    ]
    assert (out / "manifest.json").exists()


def test_without_report_estimators_are_still_required(tmp_path):
    with pytest.raises(ValueError, match="--estimators is required unless --report"):
        main(_args(tmp_path / "results"))


def test_a_fact_table_that_cannot_be_built_is_one_line_and_loses_nothing_else(
    tmp_path, monkeypatch, capsys
):
    out = tmp_path / "results"

    def refuse(outcome):
        raise ValueError("check 'Bad Name' cannot be a fact id segment")

    monkeypatch.setattr("cp_cli.commands.analyze.fact_table", refuse)

    code = main(_args(out, "--report", "--estimators", "difference_in_means"))

    assert code == 1
    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1
    assert "facts.json was not written" in err
    assert "Bad Name" in err
    for name in ("estimates.csv", "manifest.json", "summary.json"):
        assert (out / name).exists(), name
    assert not (out / "facts.json").exists()


def test_every_analyze_run_removes_what_an_earlier_analysis_left_behind(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    for name in ("facts.json", "interpretation.md", "interpretation.json"):
        (out / name).write_text("left by an earlier run")

    code = main(_args(out, "--estimators", "difference_in_means"))

    assert code == 0
    for name in ("facts.json", "interpretation.md", "interpretation.json"):
        assert not (out / name).exists(), name
    assert (out / "estimates.csv").exists()
    assert (out / "manifest.json").exists()


def test_a_report_run_writes_the_fact_table_after_the_estimates(tmp_path):
    out = tmp_path / "results"

    assert main(_args(out, "--report", "--estimators", "difference_in_means")) == 0

    facts = out / "facts.json"
    assert facts.exists()
    assert facts.stat().st_mtime_ns >= (out / "manifest.json").stat().st_mtime_ns
