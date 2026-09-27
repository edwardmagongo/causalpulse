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
