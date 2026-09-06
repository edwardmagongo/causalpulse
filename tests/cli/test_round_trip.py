"""analyze -> replay, end to end through the command line.

Every other test in tests/cli/ exercises one command. This file exercises the
seam between two, which is where a serialisation or path-handling mistake
lives -- the kind of defect a library-level test cannot see because it never
writes a file.
"""

import json

import pandas as pd
import pytest
from cp_cli.__main__ import main


def test_a_scenario_analysis_replays_to_identical_estimates(tmp_path, capsys):
    out = tmp_path / "results"
    main(["analyze", "--scenario", "D", "--n-units", "500", "--seed", "2",
          "--estimators", "difference_in_means,lin_regression,ipw", "--out", str(out)])
    recorded = pd.read_csv(out / "estimates.csv")
    capsys.readouterr()

    code = main(["replay", str(out / "manifest.json")])

    # `replay` (Task 6) already performs the structured comparison itself: it
    # reads estimates.csv back with a round-trip-safe float parser and diffs
    # each replayed point against the recorded one at position, exactly. So
    # the exit code IS the numeric verdict -- asserting it, plus that replay
    # actually compared against the recorded file (rather than silently
    # falling back to the weaker two-replay comparison because estimates.csv
    # went missing), is stronger than scraping printed numbers back out of
    # text and re-implementing the comparison in the test.
    assert code == 0
    printed = capsys.readouterr().out
    assert f"comparing against recorded estimates at {out / 'estimates.csv'}" in printed
    assert f"all {len(recorded)} estimator(s) match exactly" in printed


def test_a_csv_analysis_replays_to_identical_estimates(tmp_path):
    data = tmp_path / "d.csv"
    main(["simulate", "--scenario", "D", "--n-units", "400", "--seed", "5", "--out", str(data)])
    out = tmp_path / "results"
    main(["analyze", "--data", str(data), "--unit", "unit_id", "--treatment", "treated",
          "--outcome", "outcome", "--covariates", "x1,x2", "--outcome-type", "binary",
          "--estimators", "difference_in_means,ipw", "--out", str(out)])

    assert main(["replay", str(out / "manifest.json")]) == 0


def test_editing_the_csv_after_analysis_makes_replay_refuse(tmp_path):
    # The manifest records the file's sha256 for exactly this. Nothing had
    # asserted it end to end before now.
    data = tmp_path / "d.csv"
    main(["simulate", "--scenario", "D", "--n-units", "400", "--seed", "5", "--out", str(data)])
    out = tmp_path / "results"
    main(["analyze", "--data", str(data), "--unit", "unit_id", "--treatment", "treated",
          "--outcome", "outcome", "--covariates", "x1,x2", "--outcome-type", "binary",
          "--estimators", "difference_in_means", "--out", str(out)])

    frame = pd.read_csv(data)
    frame.loc[0, "outcome"] = 1 - frame.loc[0, "outcome"]
    frame.to_csv(data, index=False)

    with pytest.raises(ValueError, match="does not match its recorded content"):
        main(["replay", str(out / "manifest.json")])


def test_moving_the_csv_makes_replay_refuse_rather_than_silently_regenerating(tmp_path):
    data = tmp_path / "d.csv"
    main(["simulate", "--scenario", "D", "--n-units", "300", "--seed", "6", "--out", str(data)])
    out = tmp_path / "results"
    main(["analyze", "--data", str(data), "--unit", "unit_id", "--treatment", "treated",
          "--outcome", "outcome", "--covariates", "x1,x2", "--outcome-type", "binary",
          "--estimators", "difference_in_means", "--out", str(out)])

    data.unlink()

    with pytest.raises(FileNotFoundError):
        main(["replay", str(out / "manifest.json")])


def test_the_manifest_records_the_estimator_configuration(tmp_path):
    out = tmp_path / "results"
    main(["analyze", "--scenario", "A", "--n-units", "300", "--seed", "0",
          "--estimators", "difference_in_means", "--random-seed", "7", "--out", str(out)])

    manifest = json.loads((out / "manifest.json").read_text())

    assert manifest["estimators"][0]["config"]["random_seed"] == 7
