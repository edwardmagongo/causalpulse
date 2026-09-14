import pytest
from cp_cli.__main__ import main


def _analyze_scenario(tmp_path, estimators="difference_in_means,ipw"):
    out = tmp_path / "results"
    main(["analyze", "--scenario", "D", "--n-units", "400", "--seed", "0",
          "--estimators", estimators, "--out", str(out)])
    return out / "manifest.json"


def test_replaying_a_scenario_manifest_matches_exactly(tmp_path, capsys):
    manifest = _analyze_scenario(tmp_path)

    code = main(["replay", str(manifest)])

    assert code == 0
    assert "match" in capsys.readouterr().out.lower()


def test_replay_reports_each_estimator(tmp_path, capsys):
    manifest = _analyze_scenario(tmp_path)

    main(["replay", str(manifest)])

    out = capsys.readouterr().out
    assert "difference_in_means" in out
    assert "ipw" in out


def test_replaying_a_fixture_manifest_matches_exactly(tmp_path):
    data = tmp_path / "d.csv"
    main(["simulate", "--scenario", "D", "--n-units", "300", "--seed", "1", "--out", str(data)])
    out = tmp_path / "results"
    main(["analyze", "--data", str(data), "--unit", "unit_id", "--treatment", "treated",
          "--outcome", "outcome", "--covariates", "x1,x2", "--outcome-type", "binary",
          "--estimators", "difference_in_means", "--out", str(out)])

    assert main(["replay", str(out / "manifest.json")]) == 0


def test_replay_refuses_a_fixture_that_changed(tmp_path):
    data = tmp_path / "d.csv"
    main(["simulate", "--scenario", "D", "--n-units", "300", "--seed", "1", "--out", str(data)])
    out = tmp_path / "results"
    main(["analyze", "--data", str(data), "--unit", "unit_id", "--treatment", "treated",
          "--outcome", "outcome", "--covariates", "x1,x2", "--outcome-type", "binary",
          "--estimators", "difference_in_means", "--out", str(out)])

    data.write_text(data.read_text() + "999,1,0,0.0,0.0\n")

    with pytest.raises(ValueError, match="does not match its recorded content"):
        main(["replay", str(out / "manifest.json")])


def test_a_manifest_at_an_unknown_schema_version_is_refused(tmp_path):
    manifest_path = _analyze_scenario(tmp_path, estimators="difference_in_means")
    tampered = manifest_path.read_text().replace('"schema_version": 1', '"schema_version": 99')
    manifest_path.write_text(tampered)

    with pytest.raises(ValueError, match="schema_version"):
        main(["replay", str(manifest_path)])


def test_a_mismatch_exits_non_zero(tmp_path, monkeypatch, capsys):
    # The comparison must be capable of reporting failure, or a zero exit code
    # means nothing. Perturb one replayed estimate and assert the command says
    # so and exits non-zero.
    import cp_cli.commands.replay as replay_module

    manifest = _analyze_scenario(tmp_path, estimators="difference_in_means")
    original = replay_module.replay

    def perturbed(manifest_obj, resolve):
        import dataclasses

        estimates = original(manifest_obj, resolve)
        return tuple(
            dataclasses.replace(e, point=e.point + 1.0) for e in estimates
        )

    monkeypatch.setattr(replay_module, "replay", perturbed)

    code = main(["replay", str(manifest)])

    assert code != 0
    assert "differ" in capsys.readouterr().out.lower()


def test_a_panel_scenario_reference_is_refused_by_the_resolver(tmp_path):
    # resolve_scenario_reference supports n_units plus every ScenarioParams
    # axis, and refuses every other key by name rather than silently
    # regenerating different data. A panel reference carries n_periods, which
    # is none of those, so it cannot round-trip. Pinned here so the limit is
    # visible.
    import numpy as np
    from cp_core.manifest import ScenarioReference
    from cp_simulation.panel_scenarios import build_panel_scenario
    from cp_simulation.scenarios import resolve_scenario_reference

    schema = build_panel_scenario("P1").generate(
        n_units=100, n_periods=6, rng=np.random.default_rng(0)
    ).dataset.schema
    reference = ScenarioReference(
        scenario="P1", seed=0, params={"n_units": 100, "n_periods": 6}
    )

    with pytest.raises(ValueError, match="n_periods"):
        resolve_scenario_reference(reference, schema)
