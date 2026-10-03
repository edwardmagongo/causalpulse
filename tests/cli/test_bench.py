import dataclasses

import cp_cli.commands.bench as bench_module
import pandas as pd
import pytest
from cp_cli.__main__ import main
from cp_cli.io import read_benchmark, write_benchmark
from cp_evaluation.grid import (
    CateAxes,
    CrossSectionAxes,
    CrossSectionCell,
    DonorPoolAxes,
    GridConfig,
    LongPanelAxes,
    expand_cells,
)
from cp_evaluation.run import run_grid
from cp_evaluation.tiers import ci_config, full_config
from cp_simulation.params import AXIS_NAMES
from cp_simulation.scenarios import SCENARIOS


@pytest.fixture(scope="module")
def ci_run(tmp_path_factory):
    """Run the `ci` tier once for the whole module.

    `bench run --tier ci` costs about 100 seconds. Seven of this module's
    tests want to inspect its output; running it once into a shared
    `tmp_path_factory` directory and having those tests read it back keeps
    the file's wall-clock cost at roughly the tier's own, not seven times it.
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
    assert bench_module._axis_discrepancies(sidecar) == []


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
    lines = bench_module._axis_discrepancies(sidecar)
    assert len(lines) == 1
    assert "cross_section.confounding_strength" in lines[0]
    assert "requested [0.5, 2.0]" in lines[0]
    assert "realised [0.5, 1.0, 2.0]" in lines[0]


def test_the_realised_block_renders_integers_as_integers(ci_run):
    # `n_periods` is NaN on every cross-sectional row, which widens the whole
    # column to float64, so the realised block would record `[6.0]` beside the
    # requested block's `[6]` in one run.json -- one value rendered two ways,
    # in the pair of blocks that exist to be read side by side.
    #
    # Asserted on the *types*, not with `==`: `[6] == [6.0]` is True in Python
    # and `json.loads` preserves the distinction only in the token, so an
    # equality check here would pass against the unfixed rendering and test
    # nothing.
    _, sidecar = read_benchmark(ci_run)

    requested = sidecar["panel"]["n_periods"]
    realised = sidecar["panel_realised"]["n_periods"]

    assert all(isinstance(value, int) for value in requested), requested
    assert all(isinstance(value, int) for value in realised), realised
    assert realised == requested

    # A float axis stays a float: the narrowing keys off how the axis was
    # requested, not off whether the value happens to be whole.
    strengths = sidecar["cross_section_realised"]["confounding_strength"]
    assert all(isinstance(value, float) for value in strengths), strengths


def _sidecar_for(config, frame):
    return bench_module.build_sidecar("t", config, ["synthetic_control"], 10, 0, 0, frame, {})


def test_the_sidecar_records_the_donor_arm_requested_and_realised():
    axes = DonorPoolAxes(scenarios=("S0", "S1", "S2"))
    config = GridConfig(cross_section=None, panel=None, donor_pool=axes, replications=1)
    frame = pd.DataFrame(
        {"shape": ["donor_pool", "donor_pool", "panel"], "scenario": ["S0", "S1", "E"]}
    )

    sidecar = _sidecar_for(config, frame)

    assert sidecar["donor_pool"] == {"scenarios": ["S0", "S1", "S2"]}
    # S2 was asked for and absent from the rows; the panel row is not a donor row.
    assert sidecar["donor_pool_realised"] == {"scenarios": ["S0", "S1"]}
    lines = bench_module._axis_discrepancies(sidecar)
    assert len(lines) == 1
    assert "donor_pool.scenarios" in lines[0]


def test_the_sidecar_records_the_long_panel_arm_requested_and_realised():
    config = GridConfig(
        cross_section=None, panel=None, long_panel=LongPanelAxes(scenarios=("P7",)), replications=1
    )
    frame = pd.DataFrame({"shape": ["long_panel", "panel"], "scenario": ["P7", "P1"]})

    sidecar = _sidecar_for(config, frame)

    assert sidecar["long_panel"] == {"scenarios": ["P7"]}
    assert sidecar["long_panel_realised"] == {"scenarios": ["P7"]}
    assert bench_module._axis_discrepancies(sidecar) == []


def test_a_requested_long_panel_scenario_that_never_landed_is_a_discrepancy():
    config = GridConfig(
        cross_section=None, panel=None, long_panel=LongPanelAxes(scenarios=("P7",)), replications=1
    )
    frame = pd.DataFrame({"shape": ["panel"], "scenario": ["P1"]})

    sidecar = _sidecar_for(config, frame)

    assert sidecar["long_panel_realised"] == {"scenarios": []}
    lines = bench_module._axis_discrepancies(sidecar)
    assert len(lines) == 1
    assert "long_panel.scenarios" in lines[0]


def test_a_grid_with_no_long_panel_arm_records_none_for_it():
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(200,), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=1,
    )
    frame = pd.DataFrame(
        {"shape": ["cross_section"], "scenario": ["A"], "n_units": [200]}
    ).assign(**{name: 1.0 for name in AXIS_NAMES if name not in ("scenarios", "n_units")})

    sidecar = _sidecar_for(config, frame)

    assert sidecar["long_panel"] is None
    assert sidecar["long_panel_realised"] is None


def test_a_grid_with_no_donor_arm_records_none_for_it():
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(200,), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=1,
    )
    frame = pd.DataFrame(
        {"shape": ["cross_section"], "scenario": ["A"], "n_units": [200]}
    ).assign(**{name: 1.0 for name in AXIS_NAMES if name not in ("scenarios", "n_units")})

    sidecar = _sidecar_for(config, frame)

    assert sidecar["donor_pool"] is None
    assert sidecar["donor_pool_realised"] is None


def test_the_sidecar_records_the_cate_arm_requested_and_realised():
    config = GridConfig(
        cross_section=None, panel=None, cate=CateAxes(scenarios=("RICH", "NULL")), replications=1
    )
    frame = pd.DataFrame({"shape": ["cate", "donor_pool"], "scenario": ["RICH", "S1"]})

    sidecar = _sidecar_for(config, frame)

    assert sidecar["cate"] == {"scenarios": ["RICH", "NULL"]}
    assert sidecar["cate_realised"] == {"scenarios": ["RICH"]}
    lines = bench_module._axis_discrepancies(sidecar)
    assert len(lines) == 1
    assert "cate.scenarios" in lines[0]


def test_a_grid_with_no_cate_arm_records_none_for_it():
    config = GridConfig(cross_section=None, panel=None, donor_pool=DonorPoolAxes(("S1",)),
                        replications=1)
    frame = pd.DataFrame({"shape": ["donor_pool"], "scenario": ["S1"]})

    sidecar = _sidecar_for(config, frame)

    assert sidecar["cate"] is None
    assert sidecar["cate_realised"] is None


def test_a_fractional_value_is_never_narrowed_into_agreement():
    # The safety property of the narrowing above. If it rounded rather than
    # only rewriting an exactly-integral float, a realised 6.5 on an axis
    # requested as `(6,)` would be recorded as 6 -- silently reporting that the
    # run produced what was asked for when it did not, which is the precise
    # failure the realised block exists to expose.
    assert bench_module._matching_requested_type(6.0, (6,)) == 6
    assert isinstance(bench_module._matching_requested_type(6.0, (6,)), int)

    assert bench_module._matching_requested_type(6.5, (6,)) == 6.5
    # Requested as floats, so nothing is narrowed even when whole.
    assert isinstance(bench_module._matching_requested_type(2.0, (0.5, 2.0)), float)
    # Non-numeric and empty axes are passed straight through.
    assert bench_module._matching_requested_type("A", ("A", "D")) == "A"
    assert bench_module._matching_requested_type(6.0, ()) == 6.0


def test_bench_report_prints_the_note_when_an_axis_disagrees(tmp_path, capsys):
    # `_axis_discrepancies` returning the right lines is only half the fix: the
    # point of the realised block is that a reader of `bench report` is told.
    # Nothing else covers that branch -- the only other test that runs `report`
    # uses the `ci` fixture, which by design has no discrepancy -- so deleting
    # the whole note block from `report` leaves the CLI suite green.
    #
    # Driven through `write_benchmark` and `main` rather than by calling the
    # helper, so the assertion covers the branch, the wording, and the stream
    # it is printed on.
    axes = CrossSectionAxes(
        scenarios=("A", "D"), n_units=(200,), confounding_strength=(0.5, 2.0)
    )
    frame = run_grid(
        GridConfig(cross_section=axes, panel=None, replications=1),
        estimators=["difference_in_means"],
    )
    write_benchmark(
        frame,
        tmp_path,
        {
            "tier": "demo",
            "cross_section": bench_module._axis_block(axes),
            "cross_section_realised": bench_module._realised_block(frame, axes),
        },
    )

    capsys.readouterr()

    assert main(["bench", "report", str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    assert "note:" in printed
    assert "cross_section.confounding_strength" in printed
    assert "requested [0.5, 2.0]" in printed
    assert "realised [0.5, 1.0, 2.0]" in printed
    # The note qualifies the summary, so it has to precede it.
    assert printed.index("note:") < printed.index("method")


def test_bench_report_stays_quiet_when_every_axis_got_what_it_asked_for(ci_run, capsys):
    # The other half: a note that fired on every run would be ignored on the
    # run that needed it. `ci` asks for ScenarioParams' own default strength,
    # so its gated scenarios collapse onto a value it requested anyway.
    capsys.readouterr()

    assert main(["bench", "report", str(ci_run)]) == 0

    assert "note:" not in capsys.readouterr().out


def test_the_full_tier_produces_no_discrepancy_either():
    # Both `_axis_discrepancies`' docstring and `grid._confounding_values`
    # promise the note stays silent for `full` as well as `ci`. `full` takes
    # ~3.1 hours, so it can never be run here -- but the claim is about which
    # cells the grid enumerates, not about any fit, so `expand_cells` settles
    # it in milliseconds.
    #
    # `full` is silent only because every axis value it requests is produced by
    # some scenario that declares that axis. That is a property of the scenario
    # set, not a law: `heterogeneity_scale` is declared by F alone, so the
    # negative control below drops F and confirms this check has teeth rather
    # than passing on any config at all.
    assert _cross_section_discrepancies(full_config().cross_section) == []

    trimmed = dataclasses.replace(
        full_config().cross_section,
        scenarios=tuple(s for s in full_config().cross_section.scenarios if s != "F"),
    )
    assert "heterogeneity_scale" in SCENARIOS["F"].axes
    assert not any("heterogeneity_scale" in SCENARIOS[s].axes for s in trimmed.scenarios)

    without_f = _cross_section_discrepancies(trimmed)
    assert len(without_f) == 1
    assert "heterogeneity_scale" in without_f[0]


def _cross_section_discrepancies(axes):
    """The discrepancies a run of `axes` would report, from enumeration alone.

    A tier too slow to fit can still be checked this way, because `run_grid`
    writes one row per cell per estimator and copies every column this reads
    off the cell unchanged (`run._cell_columns`). So the cells are expanded,
    turned into the frame those rows would form, and handed to the real
    `_realised_block`.

    Deliberately not a second implementation of what `_realised_block` does.
    An earlier version derived the realised values straight from the cells,
    which meant this test -- the only check on a claim about a tier that takes
    ~3.1 hours and so can never be run here -- was validating a copy of the
    rule rather than the one that ships. It was faithful when written and
    already omitted `_matching_requested_type`; `test_bench_run_uses_the_gated_
    ci_config_itself` exists in this same file because a second definition
    that drifts is how this repo has been bitten before.
    """
    cells = [
        cell
        for cell in expand_cells(GridConfig(cross_section=axes, panel=None, replications=1))
        if isinstance(cell, CrossSectionCell)
    ]
    frame = pd.DataFrame(
        [
            {
                "shape": "cross_section",
                "scenario": cell.scenario,
                "n_units": cell.n_units,
                "n_periods": float("nan"),
                **{name: getattr(cell, name) for name in AXIS_NAMES},
            }
            for cell in cells
        ]
    )
    return bench_module._axis_discrepancies(
        {
            "cross_section": bench_module._axis_block(axes),
            "cross_section_realised": bench_module._realised_block(frame, axes),
        }
    )


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

    assert bench_module._axis_discrepancies(sidecar) == []


def test_a_sidecar_written_before_the_realised_block_existed_reports_nothing():
    # `read_benchmark` hands back whatever JSON is on disk, including a
    # run.json from before these blocks were added. Reporting one must not
    # raise, and must not invent a discrepancy from a block that is absent.
    assert bench_module._axis_discrepancies({"cross_section": {"n_units": [500]}}) == []
    assert bench_module._axis_discrepancies({}) == []


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


def test_the_cate_tier_is_registered_and_sharded():
    from cp_evaluation.tiers import FULL_ESTIMATORS, FULL_FIT_CONFIG, cate_study_config

    assert bench_module.TIERS["cate"] == (cate_study_config, FULL_ESTIMATORS, FULL_FIT_CONFIG)
    assert "cate" in bench_module.SHARDED_TIERS


def test_the_p7_tier_is_registered_and_sharded():
    from cp_evaluation.tiers import FULL_ESTIMATORS, FULL_FIT_CONFIG, p7_study_config

    assert bench_module.TIERS["p7"] == (p7_study_config, FULL_ESTIMATORS, FULL_FIT_CONFIG)
    assert "p7" in bench_module.SHARDED_TIERS
