import hashlib
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


@pytest.fixture
def cate(tmp_path, monkeypatch):
    import cp_cli.commands.bench as bench_module
    from cp_core.values import EstimatorConfig
    from cp_evaluation.grid import CateAxes, GridConfig

    monkeypatch.setitem(
        bench_module.TIERS,
        "cate",
        (
            lambda: GridConfig(
                cross_section=None, panel=None, cate=CateAxes(scenarios=("RICH",)),
                replications=2,
            ),
            ("causal_forest",),
            EstimatorConfig(random_seed=0, bootstrap_reps=20, options={"n_trees": 20}),
        ),
    )
    assert main(["bench", "run", "--tier", "cate", "--out", str(tmp_path / "cate")]) == 0
    return tmp_path / "cate"


@pytest.fixture
def p7(tmp_path, monkeypatch):
    import cp_cli.commands.bench as bench_module
    from cp_core.values import EstimatorConfig
    from cp_evaluation.grid import GridConfig, LongPanelAxes

    monkeypatch.setitem(
        bench_module.TIERS,
        "p7",
        (
            lambda: GridConfig(
                cross_section=None, panel=None, long_panel=LongPanelAxes(scenarios=("P7",)),
                replications=2,
            ),
            ("callaway_santanna", "twfe"),
            EstimatorConfig(random_seed=0, bootstrap_reps=20),
        ),
    )
    assert main(["bench", "run", "--tier", "p7", "--out", str(tmp_path / "p7")]) == 0
    return tmp_path / "p7"


@pytest.mark.parametrize("fmt, name", [("markdown", "benchmark-report.md"),
                                       ("html", "benchmark-report.html")])
def test_a_cate_run_adds_its_section_to_either_format(run, cate, tmp_path, fmt, name):
    out = tmp_path / f"out-{fmt}"
    out.mkdir()

    assert _report(run, out, "--format", fmt, "--cate-run", str(cate)) == 0

    assert "Heterogeneous Effects (CATE)" in (out / name).read_text()


def test_a_non_cate_run_is_refused_as_the_cate_run(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(ValueError, match="'cate'"):
        _report(run, out, "--cate-run", str(run))


@pytest.mark.parametrize("fmt, name", [("markdown", "benchmark-report.md"),
                                       ("html", "benchmark-report.html")])
def test_a_p7_run_adds_its_section_to_either_format(run, p7, tmp_path, fmt, name):
    out = tmp_path / f"out-{fmt}"
    out.mkdir()

    assert _report(run, out, "--format", fmt, "--p7-run", str(p7)) == 0

    assert "Scenario P7" in (out / name).read_text()


def test_a_non_p7_run_is_refused_as_the_p7_run(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(ValueError, match="'p7'"):
        _report(run, out, "--p7-run", str(run))


def test_bundle_writes_the_study_bundle_beside_the_report_it_names(run, tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    bundle = tmp_path / "bundle.json"

    assert _report(run, out, "--bundle", str(bundle)) == 0

    recorded = json.loads(bundle.read_text())
    report = (out / "benchmark-report.md").read_bytes()
    assert recorded["report_sha256"] == hashlib.sha256(report).hexdigest()
    # Built from a real store: the run's own provenance and each scenario's shape are there.
    assert [run["role"] for run in recorded["provenance"]] == ["study"]
    assert recorded["provenance"][0]["tier"] == "study"
    assert {s["name"]: s["shape"] for s in recorded["scenarios"]} == {
        "A": "cross_section",
        "P1": "panel",
    }
    assert "wrote the study bundle to" in capsys.readouterr().out


def test_a_cate_and_a_p7_run_are_named_in_the_bundles_provenance(run, cate, p7, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    bundle = tmp_path / "bundle.json"

    extra = ("--cate-run", str(cate), "--p7-run", str(p7), "--bundle", str(bundle))
    assert _report(run, out, *extra) == 0

    recorded = json.loads(bundle.read_text())
    assert [(r["role"], r["tier"]) for r in recorded["provenance"]] == [
        ("study", "study"),
        ("cate", "cate"),
        ("p7", "p7"),
    ]
    assert recorded["cate"] is not None and recorded["p7"] is not None


def test_bundle_is_refused_with_the_html_format_and_nothing_is_written(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    bundle = tmp_path / "bundle.json"

    with pytest.raises(ValueError, match="--format markdown"):
        _report(run, out, "--format", "html", "--bundle", str(bundle))

    assert not bundle.exists()
    assert list(out.iterdir()) == []


def test_two_bundles_from_one_store_are_byte_identical(run, tmp_path):
    texts = []
    for name in ("one", "two"):
        out = tmp_path / name
        out.mkdir()
        assert _report(run, out, "--bundle", str(out / "bundle.json")) == 0
        texts.append((out / "bundle.json").read_bytes())

    assert texts[0] == texts[1]
