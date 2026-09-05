import numpy as np
import pandas as pd
import pytest
from cp_core.values import EstimatorConfig
from cp_evaluation.grid import CrossSectionAxes, GridConfig, PanelAxes
from cp_evaluation.run import ROW_COLUMNS, run_grid


def _tiny_cross_section(
    scenarios=("A",), n_units=(400,), confounding_strength=(1.0,), replications=2
) -> GridConfig:
    return GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=scenarios, n_units=n_units, confounding_strength=confounding_strength
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
    config = _tiny_cross_section(
        scenarios=("A", "D"),
        n_units=(400,),
        confounding_strength=(1.0, 2.0),
        replications=2,
    )

    straight = run_grid(config)
    shuffled = run_grid(config, _shuffle=np.random.default_rng(0))

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
