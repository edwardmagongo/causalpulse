import numpy as np
import pandas as pd
import pytest
from cp_evaluation.grid import CrossSectionAxes, GridConfig, PanelAxes
from cp_evaluation.run import ROW_COLUMNS, run_grid


def _tiny_cross_section(scenarios=("A",), n_units=(400,), replications=2) -> GridConfig:
    return GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=scenarios, n_units=n_units, confounding_strength=(1.0,)
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
    config = _tiny_cross_section(scenarios=("A", "D"), n_units=(400,), replications=2)

    straight = run_grid(config)
    shuffled = run_grid(config, _shuffle=np.random.default_rng(0))

    key = ["scenario", "n_units", "seed", "method"]
    left = straight.sort_values(key).reset_index(drop=True).drop(columns=["runtime_seconds"])
    right = shuffled.sort_values(key).reset_index(drop=True).drop(columns=["runtime_seconds"])
    pd.testing.assert_frame_equal(left, right)


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
