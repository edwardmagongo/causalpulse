import numpy as np
import pandas as pd
import pytest
from cp_core.values import EstimatorConfig
from cp_evaluation.grid import CrossSectionAxes, CrossSectionCell, GridConfig, PanelAxes, cell_rng
from cp_evaluation.resolve import resolve_cell
from cp_evaluation.run import ROW_COLUMNS, run_grid
from cp_evaluation.tiers import full_config
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
    # values produce genuinely tied cells -- e.g. (A, 400, strength=1.0,
    # seed=0) and (A, 400, strength=2.0, seed=0) -- that agree on every other
    # column, which is exactly the case a total sort key must resolve.
    # Sweeping two scenarios keeps that tie exercised alongside more than one
    # scenario, as the previous version of this test already did.
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
    # final-review.md Minor 6: treatment_intercept alone left the other three
    # new axes' sort-key entries almost unexercised -- dropping effect_scale
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
    # exercises equally; fitting all of them fits 16 cells twice and costs
    # ~224s against ~4.5s here, on a suite whose full gate already runs
    # 2h47m. `twfe` is kept deliberately: it is ineligible on every
    # cross-sectional cell, so the ineligible-row path stays inside the frame
    # comparison rather than being dropped along with the estimators.
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
    # final-review.md Minor 7: restricted to one fast analytic estimator
    # rather than the whole registry -- the columns under test here are
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
    # Spec section 4.1: treatment_intercept sets a logit-scale intercept, but
    # a reader of the results thinks in a treated fraction, not a logit
    # shift. This must be a real, measured value in [0, 1] -- not the raw
    # intercept relabelled.
    #
    # final-review.md Minor 7: restricted to two fast analytic estimators
    # rather than the whole registry. Two, not one, deliberately -- the
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
    # final-review.md Minor 7: restricted to `twfe` rather than fitting the
    # whole registry (33.5s -> well under a second) -- these columns are
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
    # final-review.md I1: `resolve_cell` (resolve.py:51-60) omitting one of
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
    # final-review.md I1: a `_realised_treated_share` mutated to
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
