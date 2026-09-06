import json

import pandas as pd
import pytest
from cp_cli.__main__ import main


def _simulate(tmp_path, scenario="D", n_units=400, seed=0):
    path = tmp_path / f"{scenario}.csv"
    main(["simulate", "--scenario", scenario, "--n-units", str(n_units),
          "--seed", str(seed), "--out", str(path)])
    return path


def test_analyzing_a_scenario_writes_estimates_and_a_manifest(tmp_path):
    out = tmp_path / "results"

    code = main(
        ["analyze", "--scenario", "D", "--n-units", "400", "--seed", "0",
         "--estimators", "difference_in_means,lin_regression", "--out", str(out)]
    )

    assert code == 0
    estimates = pd.read_csv(out / "estimates.csv")
    assert list(estimates.method) == ["difference_in_means", "lin_regression"]
    assert estimates.point.notna().all()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["dataset"]["kind"] == "scenario"
    assert [e["name"] for e in manifest["estimators"]] == [
        "difference_in_means", "lin_regression"
    ]


def test_analyzing_a_csv_records_a_fixture_reference_with_its_hash(tmp_path):
    data = _simulate(tmp_path)
    out = tmp_path / "results"

    code = main(
        ["analyze", "--data", str(data), "--unit", "unit_id", "--treatment", "treated",
         "--outcome", "outcome", "--covariates", "x1,x2", "--outcome-type", "binary",
         "--estimators", "difference_in_means", "--out", str(out)]
    )

    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["dataset"]["kind"] == "fixture"
    assert len(manifest["dataset"]["content_sha256"]) == 64


def test_a_scenario_source_needs_no_schema_flags(tmp_path):
    # The DGP carries the schema; requiring it again would be asking the user
    # to restate something the system already knows.
    code = main(["analyze", "--scenario", "A", "--n-units", "300", "--seed", "0",
                 "--estimators", "difference_in_means", "--out", str(tmp_path / "r")])

    assert code == 0


def test_an_unknown_estimator_is_refused_by_name(tmp_path):
    with pytest.raises(KeyError, match="frobnicate"):
        main(["analyze", "--scenario", "A", "--n-units", "300", "--seed", "0",
              "--estimators", "frobnicate", "--out", str(tmp_path / "r")])


def test_giving_neither_a_scenario_nor_data_is_refused(tmp_path):
    with pytest.raises(ValueError, match="--scenario"):
        main(["analyze", "--estimators", "difference_in_means", "--out", str(tmp_path / "r")])


def test_giving_both_a_scenario_and_data_is_refused(tmp_path):
    data = _simulate(tmp_path)
    with pytest.raises(ValueError, match="both"):
        main(["analyze", "--scenario", "D", "--n-units", "400", "--seed", "0",
              "--data", str(data), "--estimators", "difference_in_means",
              "--out", str(tmp_path / "r")])


def test_a_csv_missing_a_declared_column_is_refused_naming_it(tmp_path):
    data = _simulate(tmp_path)
    with pytest.raises((ValueError, KeyError), match="nonexistent"):
        main(["analyze", "--data", str(data), "--unit", "unit_id",
              "--treatment", "nonexistent", "--outcome", "outcome",
              "--covariates", "", "--outcome-type", "binary",
              "--estimators", "difference_in_means", "--out", str(tmp_path / "r")])
