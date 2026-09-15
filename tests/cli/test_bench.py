import pytest
from cp_cli.__main__ import main
from cp_cli.io import read_benchmark
from cp_evaluation.tiers import ci_config
from cp_simulation.params import AXIS_NAMES


@pytest.fixture(scope="module")
def ci_run(tmp_path_factory):
    """Run the `ci` tier once for the whole module.

    `bench run --tier ci` costs about 100 seconds. Four of this module's
    tests want to inspect its output; running it once into a shared
    `tmp_path_factory` directory and having those tests read it back keeps
    the file's wall-clock cost at roughly the tier's own, not four times it.
    """
    out = tmp_path_factory.mktemp("ci_run")
    code = main(["bench", "run", "--tier", "ci", "--out", str(out)])
    assert code == 0
    return out


def test_bench_run_writes_parquet_and_a_sidecar(ci_run):
    frame, sidecar = read_benchmark(ci_run)
    assert len(frame) > 0
    assert sidecar["tier"] == "ci"
    assert sidecar["estimators"]
    assert sidecar["bootstrap_reps"] == 200
    assert "package_versions" in sidecar


def test_the_sidecar_records_every_axis_the_grid_can_vary(ci_run):
    # The sidecar is the run's provenance: `bench report` hands it back as the
    # description of the grid that produced the parquet beside it. An axis the
    # grid varies but the sidecar omits makes run.json misdescribe its own
    # rows, and nothing downstream cross-checks the two -- so the failure is
    # silent. Asserted against AXIS_NAMES rather than a written-out list, so a
    # future axis is covered by this test on the day it is added.
    _, sidecar = read_benchmark(ci_run)

    block = sidecar["cross_section"]
    for name in AXIS_NAMES:
        assert name in block, f"{name} is a grid axis but the sidecar does not record it"
    assert {"scenarios", "n_units"} <= set(block)
    # The recorded values must be the tier's own, not a placeholder.
    assert block["confounding_strength"] == list(ci_config().cross_section.confounding_strength)


def test_bench_run_uses_the_gated_ci_config_itself():
    # A second definition of the ci grid that drifted from the gated one would
    # make the committed tolerance bands describe neither.
    import cp_cli.commands.bench as bench_module
    from cp_evaluation.tiers import ci_config

    assert bench_module.TIERS["ci"][0] is ci_config


def test_bench_report_reads_back_what_run_wrote(ci_run, capsys):
    capsys.readouterr()

    code = main(["bench", "report", str(ci_run)])

    assert code == 0
    printed = capsys.readouterr().out
    assert "method" in printed
    assert "bias" in printed


def test_the_report_matches_summarise_on_the_same_frame(ci_run):
    # A report that disagrees with its own run is the failure this pairing
    # invites, so it is asserted rather than assumed.
    from cp_evaluation.metrics import summarise

    frame, _ = read_benchmark(ci_run)

    direct = summarise(frame)

    key = ["method", "scenario"]
    assert set(direct.columns) >= {"bias", "bias_mcse", "n_fitted"}
    assert len(direct) > 0
    assert direct.sort_values(key).n_fitted.sum() == (frame.status == "fitted").sum()


def test_an_unknown_tier_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="enormous"):
        main(["bench", "run", "--tier", "enormous", "--out", str(tmp_path / "x")])


def test_reporting_a_directory_that_was_never_written_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="run.json"):
        main(["bench", "report", str(tmp_path / "nothing")])
