import cp_cli.commands.bench as bench_module
import pytest
from cp_cli.__main__ import main
from cp_cli.io import read_benchmark
from cp_evaluation.grid import CrossSectionAxes, GridConfig
from cp_evaluation.run import run_grid
from cp_evaluation.tiers import ci_config
from cp_simulation.params import AXIS_NAMES
from cp_simulation.scenarios import SCENARIOS


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


def test_the_sidecar_records_the_values_that_actually_landed(ci_run):
    # The companion to the block above. `cross_section` is what the grid was
    # *asked* for; a scenario is crossed only against the axes it declares, so
    # what it produced can differ -- and it is the produced values that
    # describe the parquet sitting beside this file. Checked against the frame
    # rather than against a restated list, since agreeing with that frame is
    # the entire claim the block makes.
    frame, sidecar = read_benchmark(ci_run)

    realised = sidecar["cross_section_realised"]
    assert set(realised) == set(sidecar["cross_section"]), (
        "the realised block must be comparable to the requested one entry by entry"
    )

    rows = frame[frame["shape"] == "cross_section"]
    for name, values in realised.items():
        column = "scenario" if name == "scenarios" else name
        assert values == sorted(rows[column].dropna().unique().tolist()), name

    # The loop above recomputes what `_realised_block` computes, so it pins
    # "the block agrees with the parquet" but would mirror an error made in
    # both. These two are hand-derived from `ci_config()` instead, and anchor
    # it to something the implementation cannot move: the cross-sectional
    # scenarios are A, B and D, and the panel scenarios must not leak into a
    # block describing the cross-sectional arm.
    assert realised["scenarios"] == ["A", "B", "D"]
    assert realised["n_units"] == [500]

    # `ci` is a grid that gets what it asks for: its confounding axis carries
    # ScenarioParams' own default, so the scenarios that declare none of that
    # axis collapse onto a strength it requested anyway. No discrepancy to
    # report, and `bench report` should therefore stay quiet.
    assert bench_module.axis_discrepancies(sidecar) == []


def test_the_realised_block_exposes_a_strength_the_config_never_asked_for():
    # The case the realised block exists for, driven through `run_grid` rather
    # than a hand-built frame: a frame written to agree with the assertion
    # would test nothing.
    #
    # A declares no `confounding_strength` and D declares it, so at an axis of
    # (0.5, 2.0) -- which omits ScenarioParams' default -- D's rows carry 0.5
    # and 2.0 while A's carry 1.0, a strength this config never requested. The
    # requested block alone would describe a grid of {0.5, 2.0}, which is not
    # the grid in the parquet.
    assert "confounding_strength" not in SCENARIOS["A"].axes
    assert "confounding_strength" in SCENARIOS["D"].axes

    axes = CrossSectionAxes(
        scenarios=("A", "D"), n_units=(200,), confounding_strength=(0.5, 2.0)
    )
    frame = run_grid(
        GridConfig(cross_section=axes, panel=None, replications=1),
        estimators=["difference_in_means"],
    )

    realised = bench_module._realised_block(frame, axes)

    assert realised["confounding_strength"] == [0.5, 1.0, 2.0]
    assert 1.0 not in axes.confounding_strength

    # And the discrepancy is what `bench report` would print, naming the axis.
    sidecar = {
        "cross_section": bench_module._axis_block(axes),
        "cross_section_realised": realised,
    }
    lines = bench_module.axis_discrepancies(sidecar)
    assert len(lines) == 1
    assert "cross_section.confounding_strength" in lines[0]
    assert "requested [0.5, 2.0]" in lines[0]
    assert "realised [0.5, 1.0, 2.0]" in lines[0]


def test_axis_order_alone_is_not_reported_as_a_discrepancy():
    # The requested block preserves the order the axis was written in and the
    # realised block is sorted, so a comparison by list rather than by set
    # would flag every grid whose axis was not written in ascending order --
    # making the note meaningless by firing on runs that got exactly what they
    # asked for.
    sidecar = {
        "cross_section": {"n_units": [2000, 500]},
        "cross_section_realised": {"n_units": [500, 2000]},
    }

    assert bench_module.axis_discrepancies(sidecar) == []


def test_a_sidecar_written_before_the_realised_block_existed_reports_nothing():
    # `read_benchmark` hands back whatever JSON is on disk, including a
    # run.json from before these blocks were added. Reporting one must not
    # raise, and must not invent a discrepancy from a block that is absent.
    assert bench_module.axis_discrepancies({"cross_section": {"n_units": [500]}}) == []
    assert bench_module.axis_discrepancies({}) == []


def test_bench_run_uses_the_gated_ci_config_itself():
    # A second definition of the ci grid that drifted from the gated one would
    # make the committed tolerance bands describe neither.
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
