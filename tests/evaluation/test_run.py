import dataclasses

import numpy as np
import pandas as pd
import pytest
from cp_causal.registry import all_estimators, get_estimator
from cp_causal.router import route
from cp_core.values import Diagnostic, EstimatorConfig
from cp_evaluation.cate import CATE_METRIC_COLUMNS
from cp_evaluation.grid import (
    CateAxes,
    CateCell,
    CrossSectionAxes,
    CrossSectionCell,
    DonorPoolAxes,
    DonorPoolCell,
    GridConfig,
    LongPanelAxes,
    PanelAxes,
    cell_rng,
    expand_cells,
)
from cp_evaluation.resolve import resolve_cell
from cp_evaluation.run import ROW_COLUMNS, fit_cell, resolve_candidates, run_grid
from cp_evaluation.tiers import full_config
from cp_simulation.cate_scenarios import CATE_N_UNITS, CATE_SCENARIOS
from cp_simulation.donor_scenarios import DONOR_SCENARIOS
from cp_simulation.params import ScenarioParams
from cp_simulation.scenarios import build_scenario

_NEW_AXIS_COLUMNS = ("treatment_intercept", "noise_scale", "effect_scale", "heterogeneity_scale")


def _tiny_cross_section(
    scenarios=("A",),
    n_units=(400,),
    confounding_strength=(1.0,),
    replications=2,
    **axes,
) -> GridConfig:
    return GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=scenarios,
            n_units=n_units,
            confounding_strength=confounding_strength,
            **axes,
        ),
        panel=None,
        replications=replications,
    )


def _tiny_donor_pool(scenarios=("S1",), replications=1) -> GridConfig:
    return GridConfig(
        cross_section=None,
        panel=None,
        donor_pool=DonorPoolAxes(scenarios=scenarios),
        replications=replications,
    )


def test_run_grid_returns_one_row_per_cell_and_estimator():
    frame = run_grid(_tiny_cross_section())

    assert list(frame.columns) == list(ROW_COLUMNS)
    # 1 scenario x 1 size x 1 strength x 2 replications = 2 cells, and every
    # registered estimator gets a row in each, eligible or not.
    assert len(frame) == 2 * frame.method.nunique()
    assert set(frame.status) <= {"fitted", "ineligible", "refused"}


def test_panel_estimators_are_ineligible_on_a_cross_section_and_recorded_as_such():
    frame = run_grid(_tiny_cross_section(replications=1))

    twfe = frame[frame.method == "twfe"]
    assert len(twfe) == 1
    assert twfe.status.iloc[0] == "ineligible"
    assert twfe.reason.iloc[0]  # a non-empty explanation, not silence
    assert pd.isna(twfe.point.iloc[0])


def test_an_estimator_below_its_minimum_sample_size_is_ineligible_not_absent():
    # causal_forest needs 50 treated and 50 control units. At n_units=40 it
    # cannot run -- and its absence must read as *absent*, not as fine.
    frame = run_grid(_tiny_cross_section(n_units=(40,), replications=1))

    forest = frame[frame.method == "causal_forest"]
    assert len(forest) == 1
    assert forest.status.iloc[0] == "ineligible"


def test_scenario_h_is_recorded_as_refused_not_dropped():
    # Scenario H carries missing outcomes and every estimator refuses it
    # (Phase 11's missing-data refusal). The rows must still exist.
    frame = run_grid(_tiny_cross_section(scenarios=("H",), replications=1))

    refused = frame[frame.status == "refused"]
    assert len(refused) > 0
    assert refused.reason.str.contains("missing").any()
    assert frame.point.isna().all()


def test_fitted_rows_carry_the_estimate_and_its_truth():
    frame = run_grid(_tiny_cross_section(scenarios=("A",), n_units=(500,), replications=1))
    fitted = frame[frame.status == "fitted"]

    assert len(fitted) > 0
    assert fitted.point.notna().all()
    assert fitted.truth.notna().all()
    assert fitted.estimand.isin(["ATE", "ATT"]).all()
    assert fitted.runtime_seconds.gt(0).all()


def test_a_panel_grid_fits_panel_estimators():
    config = GridConfig(
        cross_section=None,
        panel=PanelAxes(scenarios=("P1",), n_units=(150,), n_periods=(6,)),
        replications=1,
    )

    frame = run_grid(config)
    fitted = frame[frame.status == "fitted"]

    assert "twfe" in set(fitted.method)
    assert (fitted.estimand == "ATT").all()


def test_results_do_not_depend_on_the_order_cells_are_evaluated_in():
    # The property that makes a later move to a process pool safe.
    #
    # Sweeping a single confounding_strength value would make this test pass
    # by construction rather than by the property holding: every cell would
    # already be unique on (scenario, n_units, seed), so `run_grid`'s final
    # sort would never need to break a tie and a sort key that omitted
    # confounding_strength entirely would go uncaught. Two confounding
    # values produce genuinely tied cells -- e.g. (D, 400, strength=1.0,
    # seed=0) and (D, 400, strength=2.0, seed=0) -- that agree on every other
    # column, which is exactly the case a total sort key must resolve.
    # Sweeping two scenarios keeps that tie exercised alongside more than one
    # scenario, as the previous version of this test already did.
    #
    # The confounding tie is D's, not A's: A declares no `confounding_strength`
    # (`SCENARIOS["A"].axes`), so its baseline block sits at the single default
    # strength and never ties on that axis. D declares it and so spans both
    # values. A is still swept, for the new-axis ties described below.
    # A widened treatment_intercept is swept for the same reason, and it is
    # the sharper case: `expand_cells` pins every one-at-a-time cell to a
    # single anchor, so (A, 400, strength=1.0, intercept=-1.5, seed=0) and
    # (A, 400, strength=1.0, intercept=1.5, seed=0) agree on scenario,
    # n_units, confounding_strength AND seed -- every component the sort key
    # carried before Phase 18. Without the new axes in `run_grid`'s sort key
    # those rows tie completely, and a stable sort leaves them in evaluation
    # order, which is precisely what `_shuffle` perturbs. Sweeping only
    # confounding_strength leaves that untested: the axes could be dropped
    # from the key and nothing here would notice.
    #
    # Widening only treatment_intercept left the other three new axes'
    # sort-key entries almost unexercised -- dropping effect_scale
    # from `run_grid`'s sort key passed this test with only
    # treatment_intercept widened, because effect_scale never varied at all,
    # so there was nothing for that axis to tie on. A widened effect_scale
    # (both A and D declare it) creates the same kind of tie for a second
    # axis: (A, 400, strength=1.0, effect_scale=0.5, seed=0) and (A, 400,
    # strength=1.0, effect_scale=2.0, seed=0) agree on every column but
    # effect_scale, so dropping it from the key ties them too.
    config = _tiny_cross_section(
        scenarios=("A", "D"),
        n_units=(400,),
        confounding_strength=(1.0, 2.0),
        treatment_intercept=(-1.5, 0.0, 1.5),
        effect_scale=(0.5, 1.0, 2.0),
        replications=2,
    )

    # Restricted to three estimators rather than the whole registry. The
    # property under test is the totality of the sort key, which every row
    # exercises equally, so the whole registry buys nothing here and costs
    # ~327s against ~7s -- a 46x difference, on a suite whose full gate already
    # runs over two and a half hours (2h33m at the last recorded run). `twfe` is kept
    # deliberately: it is ineligible on every cross-sectional cell, so the
    # ineligible-row path stays inside the frame comparison rather than being
    # dropped along with the estimators.
    #
    # MEASURED 2026-09-19, this tree, warm after a discarded warm-up run, with
    # both `run_grid` calls below timed together: 327.2s for the whole registry
    # and 7.12s for the three estimators named here.
    #
    # Both ends are measured rather than derived, because the derived figure
    # was misread twice. The original read "16 cells twice ... ~224s", where 16
    # was the config's *total* cells at the time (8 distinct x 2 replications)
    # and "twice" the two `run_grid` calls -- correct as written. It was then
    # read as 16 *distinct* cells, called impossible, and replaced with a claim
    # that was itself false; the repair after that rescaled the 224s for the
    # config's growth (24 cells once `effect_scale` was widened here, 22 under
    # the `confounding_strength` gating) but left the "~4.5s here" beside it
    # unscaled, when it had grown by exactly the same factor. Timing both ends
    # retires the per-cell rate entirely, so there is no unit left to misread.
    estimators = ["difference_in_means", "lin_regression", "twfe"]
    straight = run_grid(config, estimators=estimators)
    shuffled = run_grid(config, estimators=estimators, _shuffle=np.random.default_rng(0))

    # Deliberately omits confounding_strength: this comparison key must still
    # leave the two confounding_strength values tied for a given (scenario,
    # n_units, seed, method), so that a non-total sort key inside `run_grid`
    # surfaces here as a mismatched row order instead of being silently
    # resolved by this test's own sort.
    key = ["scenario", "n_units", "seed", "method"]
    left = straight.sort_values(key).reset_index(drop=True).drop(columns=["runtime_seconds"])
    right = shuffled.sort_values(key).reset_index(drop=True).drop(columns=["runtime_seconds"])
    pd.testing.assert_frame_equal(left, right)


def test_a_configuration_error_is_not_recorded_as_a_refusal():
    # `causal_forest.fit` raises `ValueError` for a non-positive `n_trees`
    # tuning option -- a caller passing a bad `fit_config` on a perfectly
    # clean dataset, not the documented missing-data refusal. If `run_grid`
    # classified every `ValueError` as that refusal, it would record
    # status="refused" here and keep going: every downstream metric would
    # then be computed over the surviving estimators and reported as a
    # result, silently hiding a real configuration bug behind a benchmark row
    # that looks like an ordinary, uninteresting non-result.
    bad_config = EstimatorConfig(random_seed=0, options={"n_trees": 0})

    with pytest.raises(ValueError, match="n_trees"):
        run_grid(_tiny_cross_section(replications=1), fit_config=bad_config)


def test_an_unexpected_exception_propagates_rather_than_becoming_a_row():
    # A benchmark that swallows a crash and reports a metric over the
    # survivors is worse than one that fails.
    import cp_evaluation.run as run_module

    def explode(name):  # noqa: ANN001, ANN202
        raise RuntimeError("boom")

    original = run_module.get_estimator
    run_module.get_estimator = explode
    try:
        with pytest.raises(RuntimeError, match="boom"):
            run_grid(_tiny_cross_section(replications=1))
    finally:
        run_module.get_estimator = original


def test_restricting_to_two_estimators_yields_rows_for_exactly_those_two():
    names = ["difference_in_means", "lin_regression"]
    frame = run_grid(_tiny_cross_section(replications=1), estimators=names)

    assert set(frame.method) == {"difference_in_means", "lin_regression"}
    assert len(frame) == 2


def test_an_unknown_estimator_name_is_refused_by_name():
    names = ["difference_in_means", "bogus_estimator"]
    with pytest.raises(ValueError, match="bogus_estimator"):
        run_grid(_tiny_cross_section(replications=1), estimators=names)


def test_restricting_estimators_does_not_change_the_values_of_the_rows_that_remain():
    # Cheap config: one scenario, small n_units, few replications, and two
    # fast analytic estimators -- no bootstrapped forest fits here.
    config = _tiny_cross_section(n_units=(200,), replications=2)
    names = ["difference_in_means", "lin_regression"]

    full = run_grid(config)
    restricted = run_grid(config, estimators=names)

    subset = full[full.method.isin(names)].sort_values(["seed", "method"]).reset_index(drop=True)
    restricted = restricted.sort_values(["seed", "method"]).reset_index(drop=True)

    # check_dtype=False: `full` also carries ineligible/refused rows from
    # other estimators, so its otherwise-integer columns (e.g. n_treated)
    # are upcast to float64 by the NaNs those rows introduce. `restricted`
    # has no such rows and keeps its integer dtype. The values themselves
    # are what this test is about, not the incidental dtype.
    pd.testing.assert_frame_equal(
        subset.drop(columns=["runtime_seconds"]),
        restricted.drop(columns=["runtime_seconds"]),
        check_dtype=False,
    )


def test_cross_section_rows_carry_the_new_axis_columns_at_their_cell_values():
    # Restricted to one fast analytic estimator rather than the whole
    # registry -- the columns under test here are
    # cell-level (see `_cell_columns`), identical across every method's row
    # for the same cell, so which estimator produced the row is irrelevant
    # to what this test checks.
    frame = run_grid(_tiny_cross_section(replications=1), estimators=["difference_in_means"])

    for column in _NEW_AXIS_COLUMNS:
        assert column in frame.columns
    # This config never perturbs a new axis, so every row sits at
    # ScenarioParams' own default for each of the four -- not NaN, and not
    # some other placeholder.
    default = ScenarioParams()
    assert (frame["treatment_intercept"] == default.treatment_intercept).all()
    assert (frame["noise_scale"] == default.noise_scale).all()
    assert (frame["effect_scale"] == default.effect_scale).all()
    assert (frame["heterogeneity_scale"] == default.heterogeneity_scale).all()


def test_cross_section_rows_carry_the_realised_treated_share_not_the_intercept():
    # treatment_intercept sets a logit-scale intercept, but
    # a reader of the results thinks in a treated fraction, not a logit
    # shift. This must be a real, measured value in [0, 1] -- not the raw
    # intercept relabelled.
    #
    # Restricted to two fast analytic estimators rather than the whole
    # registry. Two, not one, deliberately -- the
    # nunique() == 1 check below is about every *method's* row reporting the
    # same measured share, so it needs at least two methods to still mean
    # anything after restricting.
    frame = run_grid(
        _tiny_cross_section(replications=1),
        estimators=["difference_in_means", "lin_regression"],
    )

    assert "realised_treated_share" in frame.columns
    assert frame["realised_treated_share"].between(0.0, 1.0).all()
    # One cell (replications=1, one scenario/n_units/confounding value) ->
    # every method's row must report the same measured share, since it is a
    # property of the generated dataset, not of which estimator read it.
    assert frame["realised_treated_share"].nunique() == 1
    # And it must not equal the (unconverted) intercept itself -- that would
    # be the intercept relabelled, not a measured share.
    assert not np.isclose(
        frame["realised_treated_share"].iloc[0], ScenarioParams().treatment_intercept
    )


def test_the_realised_treated_share_moves_with_treatment_intercept():
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",),
            n_units=(2000,),
            confounding_strength=(1.0,),
            treatment_intercept=(-1.5, 0.0, 1.5),
        ),
        panel=None,
        replications=1,
    )

    frame = run_grid(config, estimators=["difference_in_means"])
    by_intercept = frame.set_index("treatment_intercept")["realised_treated_share"]

    assert by_intercept.loc[-1.5] < by_intercept.loc[0.0] < by_intercept.loc[1.5]


def test_panel_rows_carry_nan_for_the_new_axis_columns_and_realised_share():
    # run.py already writes confounding_strength=np.nan for panel cells,
    # since the panel arm has no such axis. The four axes Phase 18 adds, and
    # the realised treated share derived from them, are cross-sectional-only
    # in exactly the same way and must follow the same convention rather
    # than inventing a second one (e.g. a sentinel float, or 0.0).
    #
    # Restricted to `twfe` rather than fitting the whole registry
    # (33.5s -> well under a second) -- these columns are
    # cell-level (see `_cell_columns`'s panel branch), NaN for every method's
    # row on a panel cell regardless of which estimator produced it, so one
    # estimator checks exactly what this test's name says.
    config = GridConfig(
        cross_section=None,
        panel=PanelAxes(scenarios=("P1",), n_units=(150,), n_periods=(6,)),
        replications=1,
    )

    frame = run_grid(config, estimators=["twfe"])

    for column in (*_NEW_AXIS_COLUMNS, "realised_treated_share"):
        assert column in frame.columns
        assert frame[column].isna().all()


def test_a_perturbed_row_traces_back_to_its_own_cell_and_its_own_rng_stream():
    # A finding of the Phase 18 whole-branch review: `resolve_cell`'s seam had no
    # guard. `resolve_cell` (resolve.py:51-60) omitting one of
    # `noise_scale=`/`effect_scale=`/`heterogeneity_scale=` from the
    # `ScenarioParams` it builds, `_cell_columns` (run.py:104-137) writing one
    # axis's value into another axis's column, and `_identity_payload`
    # (grid.py:314-351) never adding a non-default axis to a cell's RNG seed
    # all left the whole suite green (184 passed against every mutant). This
    # pins the seam directly: scenario F declares all four new axes, so
    # widening every one of them and running the grid exercises resolve.py,
    # run.py and grid.py's `_identity_payload`/`cell_rng` at once.
    #
    # The widened values are `full_config()`'s own -- not restated here as
    # fresh literals -- so this test tracks the axis spacing the production
    # tier actually ships rather than a second, possibly-diverging set.
    full_axes = full_config().cross_section
    assert full_axes is not None
    widened = {name: getattr(full_axes, name) for name in _NEW_AXIS_COLUMNS}

    default = ScenarioParams()
    n_units = max(full_axes.n_units)
    confounding_strength = default.confounding_strength

    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("F",),
            n_units=(n_units,),
            confounding_strength=(confounding_strength,),
            **widened,
        ),
        panel=None,
        replications=1,
    )
    frame = run_grid(config, estimators=["difference_in_means"])

    baseline_cell = CrossSectionCell(
        scenario="F", n_units=n_units, confounding_strength=confounding_strength, seed=0
    )
    baseline_frame = resolve_cell(baseline_cell, root_entropy=0).dataset.frame

    default_by_axis = {name: getattr(default, name) for name in _NEW_AXIS_COLUMNS}
    diffs = pd.DataFrame(
        {name: frame[name] != default_by_axis[name] for name in _NEW_AXIS_COLUMNS}
    )
    perturbed_mask = diffs.any(axis=1)

    # One row per non-default value on each of the four axes.
    expected_perturbed = sum(
        1
        for name in _NEW_AXIS_COLUMNS
        for value in widened[name]
        if value != default_by_axis[name]
    )
    assert perturbed_mask.sum() == expected_perturbed

    for row_index in frame.index[perturbed_mask]:
        row = frame.loc[row_index]
        moved = [name for name in _NEW_AXIS_COLUMNS if diffs.loc[row_index, name]]
        # Exactly the intended axis column differs from its default -- catches
        # `_cell_columns` writing one axis's value into another's column.
        assert len(moved) == 1

        cell = CrossSectionCell(
            scenario="F",
            n_units=n_units,
            confounding_strength=confounding_strength,
            seed=0,
            **{name: float(row[name]) for name in _NEW_AXIS_COLUMNS},
        )

        # The resolved dataset equals build_scenario(...).generate(rng=cell_rng(cell))
        # with the axes read from the cell -- catches `resolve_cell` omitting
        # `noise_scale=`/`effect_scale=`/`heterogeneity_scale=`.
        expected_params = ScenarioParams(
            confounding_strength=confounding_strength,
            **{name: float(row[name]) for name in _NEW_AXIS_COLUMNS},
        )
        expected_rng = cell_rng(cell, root_entropy=0)
        expected_frame = (
            build_scenario("F", expected_params)
            .generate(n_units=n_units, rng=expected_rng)
            .dataset.frame
        )
        actual_frame = resolve_cell(cell, root_entropy=0).dataset.frame
        pd.testing.assert_frame_equal(actual_frame, expected_frame)

        # The perturbed cell's data -- and its RNG stream -- differ from its
        # baseline twin's. The stream check is the one that catches
        # `_identity_payload` never adding the non-default axis: without it,
        # `noise_scale`/`effect_scale`/`heterogeneity_scale` alone can still
        # change the *data* even on an identical stream, so only comparing
        # datasets would miss that specific mutant.
        assert not actual_frame.equals(baseline_frame)
        assert cell_rng(cell, root_entropy=0).random() != cell_rng(
            baseline_cell, root_entropy=0
        ).random()


def test_realised_treated_share_is_not_the_undisturbed_sigmoid_of_the_intercept():
    # A finding of the Phase 18 whole-branch review: `resolve_cell`'s seam had no
    # guard. A `_realised_treated_share` mutated to
    # `sigmoid(cell.treatment_intercept)` was "killed only by accident" --
    # only the empty-control test's `== 1.0` check happened to fail, because
    # sigmoid(20) is not exactly 1. This pins it directly, at the intercept
    # the accident did not cover: treatment_intercept=0.0 (the default),
    # where sigmoid(0) == 0.5 exactly, so any measured share from a finite
    # n=400 sample landing on precisely 0.5 would be the giveaway.
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(400,), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=1,
    )

    frame = run_grid(config, estimators=["difference_in_means"])

    assert ScenarioParams().treatment_intercept == 0.0
    assert frame["realised_treated_share"].iloc[0] != 0.5


def test_synthetic_control_is_eligible_on_a_donor_pool_cell():
    frame = run_grid(_tiny_donor_pool(), estimators=["synthetic_control"])

    row = frame[frame.method == "synthetic_control"]
    assert len(row) == 1
    assert row.status.iloc[0] == "fitted"
    assert row.estimand.iloc[0] == "ATT"
    assert row.ci_basis.iloc[0] == "permutation"


@pytest.mark.slow
def test_bsts_is_eligible_on_a_donor_pool_cell():
    frame = run_grid(_tiny_donor_pool(), estimators=["bsts"])

    row = frame[frame.method == "bsts"]
    assert len(row) == 1
    assert row.status.iloc[0] == "fitted"
    assert row.ci_basis.iloc[0] == "posterior"


def test_synthetic_control_and_bsts_stay_ineligible_off_the_donor_pool_shape():
    # The boundary this phase opens, checked on both sides: still ineligible
    # on cross-section and panel, exactly as before.
    cross = run_grid(_tiny_cross_section(), estimators=["synthetic_control", "bsts"])
    panel_config = GridConfig(
        cross_section=None,
        panel=PanelAxes(scenarios=("P1",), n_units=(150,), n_periods=(6,)),
        replications=1,
    )
    panel = run_grid(panel_config, estimators=["synthetic_control", "bsts"])

    assert len(cross) == 2 * len(expand_cells(_tiny_cross_section()))
    assert len(panel) == 2 * len(expand_cells(panel_config))
    assert (cross.status == "ineligible").all()
    assert (panel.status == "ineligible").all()


def test_s5_leaves_synthetic_control_and_bsts_ineligible_on_insufficient_pre_periods():
    # S5 narrows to 2 pre-periods; both estimators declare min_pre_periods=10,
    # so the router should refuse eligibility before any fit is attempted.
    frame = run_grid(_tiny_donor_pool(scenarios=("S5",)), estimators=["synthetic_control", "bsts"])

    assert len(frame) == 2
    assert (frame.status == "ineligible").all()
    assert frame.reason.str.contains("pre-treatment").all()


def test_donor_pool_rows_carry_the_registrys_own_n_units_and_n_periods():
    frame = run_grid(_tiny_donor_pool(scenarios=("S1",)), estimators=["synthetic_control"])

    spec = DONOR_SCENARIOS["S1"]
    assert len(frame) == 1
    assert (frame.n_units == spec.n_units).all()
    assert (frame.n_periods == spec.n_periods).all()
    assert frame.confounding_strength.isna().all()
    assert frame.realised_treated_share.isna().all()
    assert (frame["shape"] == "donor_pool").all()


def _eligible_on_donor_cell(scenario: str) -> set[str]:
    dataset = resolve_cell(DonorPoolCell(scenario, 0), 0).dataset
    return {d.estimator_name for d in route(dataset) if d.eligible}


def test_exactly_three_estimators_are_eligible_on_a_donor_cell():
    # Read from each estimator's own requirements, not from route's output: a donor cell has
    # a single treated unit and no covariates, so only a panel estimator whose treated-unit
    # floor is 1 can be eligible on it.
    single_treated_panel = {
        name
        for name, cls in all_estimators().items()
        if cls.requirements.needs_panel
        and not cls.requirements.needs_covariates
        and cls.requirements.min_treated_units <= 1
    }
    assert single_treated_panel == {"synthetic_control", "bsts", "hierarchical_did"}

    assert _eligible_on_donor_cell("S1") == single_treated_panel


def test_only_hierarchical_did_is_eligible_on_the_short_pre_period_donor_cell():
    # S5 has 2 pre-periods; synthetic_control and bsts each require at least 10, and
    # hierarchical_did requires at most 2.
    estimators = all_estimators()
    assert estimators["synthetic_control"].requirements.min_pre_periods == 10
    assert estimators["bsts"].requirements.min_pre_periods == 10
    assert estimators["hierarchical_did"].requirements.min_pre_periods <= 2

    assert _eligible_on_donor_cell("S5") == {"hierarchical_did"}


def _tiny_cate(scenarios=("RICH",), replications=1) -> GridConfig:
    return GridConfig(
        cross_section=None,
        panel=None,
        cate=CateAxes(scenarios=scenarios),
        replications=replications,
    )


_CHEAP = EstimatorConfig(random_seed=0, bootstrap_reps=50)


def _reporting_per_unit_cate(per_unit: np.ndarray):
    """`difference_in_means`, plus a `per_unit_cate` diagnostic fixed by the test."""
    base = get_estimator("difference_in_means")

    class _Reports:
        name = base.name
        requirements = base.requirements

        def fit(self, dataset, config):
            estimate = base().fit(dataset, config)
            return dataclasses.replace(
                estimate,
                diagnostics={
                    **estimate.diagnostics,
                    "per_unit_cate": Diagnostic(
                        name="per_unit_cate", value=per_unit, unit=None, interpretation="test"
                    ),
                },
            )

    return _Reports


def _fit_with(monkeypatch, cell, per_unit):
    monkeypatch.setattr(
        "cp_evaluation.run.get_estimator", lambda name: _reporting_per_unit_cate(per_unit)
    )
    return fit_cell(
        cell,
        root_entropy=0,
        fit_config=_CHEAP,
        candidates=resolve_candidates(["difference_in_means"]),
    )


def test_row_columns_end_with_the_three_cate_metrics():
    assert ROW_COLUMNS[-3:] == CATE_METRIC_COLUMNS


def test_a_cate_cell_row_carries_its_identity_and_a_measured_treated_share():
    frame = run_grid(_tiny_cate(), estimators=["difference_in_means"], fit_config=_CHEAP)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["shape"] == "cate"
    assert row["scenario"] == "RICH"
    assert row["n_units"] == CATE_N_UNITS
    assert row["status"] == "fitted"
    assert 0.3 < row["realised_treated_share"] < 0.7
    for column in (*_NEW_AXIS_COLUMNS, "confounding_strength", "n_periods"):
        assert np.isnan(row[column]), column


def test_an_estimator_without_a_per_unit_cate_leaves_the_cate_columns_blank():
    frame = run_grid(_tiny_cate(), estimators=["difference_in_means"], fit_config=_CHEAP)

    row = frame.iloc[0]
    assert row["status"] == "fitted"
    assert row["reason"] == ""
    for column in CATE_METRIC_COLUMNS:
        assert np.isnan(row[column]), column


def test_a_per_unit_cate_equal_to_the_truth_scores_perfectly(monkeypatch):
    cell = CateCell("RICH", 0)
    truth = resolve_cell(cell, root_entropy=0).cate_truth.values

    (row,) = _fit_with(monkeypatch, cell, truth.copy())

    assert row["cate_correlation"] == pytest.approx(1.0)
    assert row["cate_rmse"] == pytest.approx(0.0)
    assert row["cate_calibration_slope"] == pytest.approx(1.0)


def test_a_per_unit_cate_on_a_cross_sectional_cell_is_not_scored(monkeypatch):
    cell = CrossSectionCell(scenario="A", n_units=400, confounding_strength=1.0, seed=0)

    (row,) = _fit_with(monkeypatch, cell, np.zeros(400))

    for column in CATE_METRIC_COLUMNS:
        assert np.isnan(row[column]), column


def test_a_misaligned_per_unit_cate_raises_rather_than_scoring(monkeypatch):
    with pytest.raises(ValueError, match="shape"):
        _fit_with(monkeypatch, CateCell("RICH", 0), np.zeros(CATE_N_UNITS - 1))


def _computing_per_unit_cate(reverse_rows: bool):
    """`difference_in_means`, plus a `per_unit_cate` diagnostic computed inside
    `fit` from the dataset it is handed -- not precomputed from `resolve_cell`.

    Takes the covariates in `dataset.schema.covariates` order out of
    `dataset.frame` and applies the RICH scenario's own effect component's
    `tau_values`. This is deliberately not the cell's clipped truth
    (`clip(p0 + tau, 0, 1) - p0`): it is the unclipped tau, so it can only
    ever be highly correlated with the truth, never identical to it -- which
    is what makes this test able to fail on a real row misalignment rather
    than passing by construction. `reverse_rows=True` feeds the covariates
    back in reversed row order, breaking the dataset/truth alignment as a
    negative control.
    """
    base = get_estimator("difference_in_means")
    effect = CATE_SCENARIOS["RICH"].build().effect

    class _Computes:
        name = base.name
        requirements = base.requirements

        def fit(self, dataset, config):
            estimate = base().fit(dataset, config)
            covariates = dataset.frame[list(dataset.schema.covariates)].to_numpy()
            if reverse_rows:
                covariates = covariates[::-1]
            per_unit = effect.tau_values(covariates)
            return dataclasses.replace(
                estimate,
                diagnostics={
                    **estimate.diagnostics,
                    "per_unit_cate": Diagnostic(
                        name="per_unit_cate", value=per_unit, unit=None, interpretation="test"
                    ),
                },
            )

    return _Computes


def test_a_per_unit_cate_computed_from_the_datasets_own_covariates_scores_highly(monkeypatch):
    # Guards against a future row reordering in `generate`/`resolve`: unlike
    # `test_a_per_unit_cate_equal_to_the_truth_scores_perfectly`, whose
    # estimate IS the truth (so it cannot detect a reordering that shifts
    # both identically), this stub derives its estimate independently, from
    # the dataset `fit` is actually handed. It reports the unclipped
    # `tau_values`, not the cell's truth (`clip(p0 + tau, 0, 1) - p0`), so a
    # correlation just under 1.0 is expected in general -- clipping bites
    # only where `p0 + tau` leaves `[0, 1]`, which this seed's draw happens
    # not to trigger.
    cell = CateCell("RICH", 0)
    monkeypatch.setattr(
        "cp_evaluation.run.get_estimator", lambda name: _computing_per_unit_cate(False)
    )
    (row,) = fit_cell(
        cell,
        root_entropy=0,
        fit_config=_CHEAP,
        candidates=resolve_candidates(["difference_in_means"]),
    )

    assert row["cate_correlation"] > 0.99


def test_reversing_the_dataset_row_order_visibly_degrades_the_correlation(monkeypatch):
    # The negative control proving the positive test above can fail: feeding
    # the same effect component the same covariates in reversed row order
    # misaligns every row against the cell's truth, and the correlation must
    # drop sharply.
    cell = CateCell("RICH", 0)
    monkeypatch.setattr(
        "cp_evaluation.run.get_estimator", lambda name: _computing_per_unit_cate(True)
    )
    (row,) = fit_cell(
        cell,
        root_entropy=0,
        fit_config=_CHEAP,
        candidates=resolve_candidates(["difference_in_means"]),
    )

    assert row["cate_correlation"] < 0.5


@pytest.mark.slow
def test_the_real_forest_is_scored_on_a_rich_cell():
    # Plumbing, not a quality bar: how well the forest recovers CATE is the
    # study's measurement to report over many replications, not a threshold
    # to fit to one seed.
    frame = run_grid(
        _tiny_cate(), estimators=["causal_forest"], fit_config=EstimatorConfig(random_seed=0)
    )

    (row,) = frame.to_dict("records")
    assert row["status"] == "fitted"
    for column in CATE_METRIC_COLUMNS:
        assert np.isfinite(row[column]), column
    assert row["cate_correlation"] > 0.0


def test_a_long_panel_row_carries_its_protocol_and_nan_axes():
    config = GridConfig(
        cross_section=None, panel=None, long_panel=LongPanelAxes(scenarios=("P7",)), replications=1
    )

    frame = run_grid(
        config,
        estimators=["callaway_santanna"],
        fit_config=EstimatorConfig(random_seed=0, bootstrap_reps=20),
    )

    (row,) = frame.to_dict("records")
    assert row["shape"] == "long_panel"
    assert (row["n_units"], row["n_periods"]) == (600, 20)
    assert row["status"] == "fitted"
    assert row["estimand"] == "ATT"
    for column in (*_NEW_AXIS_COLUMNS, "confounding_strength", "realised_treated_share"):
        assert np.isnan(row[column]), column


@pytest.mark.slow
def test_hierarchical_did_fits_a_long_panel_p7_cell_end_to_end():
    # Plumbing, not a quality bar: this only shows one P7 cell reaches a fitted row with a finite
    # estimate and truth. The study reports how well hierarchical_did does over 50 replications.
    config = GridConfig(
        cross_section=None, panel=None, long_panel=LongPanelAxes(scenarios=("P7",)), replications=1
    )

    frame = run_grid(
        config, estimators=["hierarchical_did"], fit_config=EstimatorConfig(random_seed=0)
    )

    (row,) = frame.to_dict("records")
    assert row["status"] == "fitted"
    assert row["estimand"] == "ATT"
    assert np.isfinite(row["point"])
    assert np.isfinite(row["truth"])
