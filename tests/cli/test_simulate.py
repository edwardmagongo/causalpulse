import json

import pandas as pd
import pytest
from cp_cli.__main__ import main
from cp_simulation.params import AXIS_NAMES


def test_the_written_reference_names_every_axis_not_just_the_two_that_predate_them(tmp_path):
    # Minor 1 (final-review.md): the writer's job is to record every axis
    # explicitly, mirroring `analyze`
    # (test_the_written_manifest_names_every_axis_not_just_the_two_that_predate_them
    # in tests/cli/test_analyze.py), so a replay is exact no matter what a
    # ScenarioParams default does later. `simulate` never varies these axes
    # away from their defaults, so a writer that only recorded `n_units` and
    # `confounding_strength` would still produce a byte-identical CSV and
    # every other test here would stay green -- only reading the reference
    # catches it.
    out = tmp_path / "d.csv"

    code = main(
        ["simulate", "--scenario", "D", "--n-units", "400", "--seed", "0", "--out", str(out)]
    )

    assert code == 0
    reference = json.loads((tmp_path / "d.reference.json").read_text())
    recorded = set(reference["params"])

    assert recorded >= {"n_units", *AXIS_NAMES}, (
        f"reference records {sorted(recorded)}; a replay of this run would fall back to "
        f"ScenarioParams defaults for {sorted({*AXIS_NAMES} - recorded)}"
    )


def test_a_cross_sectional_scenario_is_written_with_its_reference(tmp_path):
    out = tmp_path / "d.csv"

    code = main(
        ["simulate", "--scenario", "D", "--n-units", "400", "--seed", "0", "--out", str(out)]
    )

    assert code == 0
    frame = pd.read_csv(out)
    assert len(frame) == 400

    reference = json.loads((tmp_path / "d.reference.json").read_text())
    assert reference["kind"] == "scenario"
    assert reference["scenario"] == "D"
    assert reference["seed"] == 0
    assert reference["params"]["n_units"] == 400


def test_a_panel_scenario_takes_n_periods(tmp_path):
    out = tmp_path / "p1.csv"

    code = main(
        ["simulate", "--scenario", "P1", "--n-units", "100", "--n-periods", "6",
         "--seed", "0", "--out", str(out)]
    )

    assert code == 0
    assert len(pd.read_csv(out)) == 100 * 6


def test_the_same_seed_gives_the_same_file(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    for path in (first, second):
        main(["simulate", "--scenario", "D", "--n-units", "200", "--seed", "3", "--out", str(path)])

    assert first.read_text() == second.read_text()


def test_a_different_seed_gives_a_different_file(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    main(["simulate", "--scenario", "D", "--n-units", "200", "--seed", "3", "--out", str(first)])
    main(["simulate", "--scenario", "D", "--n-units", "200", "--seed", "4", "--out", str(second)])

    assert first.read_text() != second.read_text()


def test_an_unknown_scenario_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="ZZ"):
        main(["simulate", "--scenario", "ZZ", "--n-units", "100", "--seed", "0",
              "--out", str(tmp_path / "x.csv")])


def test_a_panel_scenario_without_n_periods_is_refused(tmp_path):
    with pytest.raises(ValueError, match="--n-periods"):
        main(["simulate", "--scenario", "P1", "--n-units", "100", "--seed", "0",
              "--out", str(tmp_path / "x.csv")])


def test_a_cross_sectional_scenario_with_n_periods_is_refused(tmp_path):
    # A cross-sectional generator has no period axis; accepting the flag and
    # ignoring it would let someone believe they had generated a panel.
    with pytest.raises(ValueError, match="--n-periods"):
        main(["simulate", "--scenario", "D", "--n-units", "100", "--n-periods", "6",
              "--seed", "0", "--out", str(tmp_path / "x.csv")])


def test_a_panel_scenario_with_confounding_strength_is_refused(tmp_path):
    # A panel generator has no confounding parameter; accepting the flag and
    # ignoring it would let someone believe their --confounding-strength had an effect.
    with pytest.raises(ValueError, match="--confounding-strength"):
        main(["simulate", "--scenario", "P1", "--n-units", "100", "--n-periods", "6",
              "--confounding-strength", "2.0", "--seed", "0", "--out", str(tmp_path / "x.csv")])


def test_the_reference_suffix_replaces_only_the_final_dot(tmp_path):
    # A guess like `out.with_suffix("").with_suffix(".reference.json")` strips
    # every dot-segment, not just the extension: `my.data.csv` would become
    # `my.reference.json`, silently dropping `data` and colliding with an
    # unrelated `my.csv` run in the same directory. A single `with_suffix`
    # call replaces only the last suffix, which is what is wanted here.
    out = tmp_path / "my.data.csv"

    code = main(
        ["simulate", "--scenario", "D", "--n-units", "50", "--seed", "0", "--out", str(out)]
    )

    assert code == 0
    assert (tmp_path / "my.data.reference.json").exists()
    assert not (tmp_path / "my.reference.json").exists()
