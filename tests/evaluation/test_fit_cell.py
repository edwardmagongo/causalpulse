import numpy as np
import pandas as pd
import pytest
from cp_core.values import EstimatorConfig
from cp_evaluation.grid import CrossSectionAxes, GridConfig, expand_cells
from cp_evaluation.run import (
    ROW_COLUMNS,
    fit_cell,
    resolve_candidates,
    run_grid,
    sort_result_frame,
)

_ESTIMATORS = ("difference_in_means", "lin_regression")


def _config() -> GridConfig:
    return GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A", "D"), n_units=(300,), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=2,
    )


def test_fit_cell_over_every_cell_is_run_grid():
    config = _config()
    fit_config = EstimatorConfig(random_seed=0, bootstrap_reps=50)
    candidates = resolve_candidates(_ESTIMATORS)

    rows = [
        row
        for cell in expand_cells(config)
        for row in fit_cell(cell, root_entropy=0, fit_config=fit_config, candidates=candidates)
    ]
    rebuilt = sort_result_frame(pd.DataFrame(rows, columns=list(ROW_COLUMNS)))
    direct = run_grid(config, estimators=_ESTIMATORS, fit_config=fit_config)

    # `runtime_seconds` is wall clock: two fits of the same cell never agree on
    # it, and nothing else about a row is allowed to differ.
    pd.testing.assert_frame_equal(
        rebuilt.drop(columns="runtime_seconds"), direct.drop(columns="runtime_seconds")
    )


def test_fit_cell_returns_one_row_per_candidate_estimator():
    (cell, *_) = expand_cells(_config())
    rows = fit_cell(
        cell,
        root_entropy=0,
        fit_config=EstimatorConfig(random_seed=0, bootstrap_reps=50),
        candidates=resolve_candidates(_ESTIMATORS),
    )

    assert sorted(row["method"] for row in rows) == sorted(_ESTIMATORS)
    assert all(set(row) == set(ROW_COLUMNS) for row in rows)


def test_an_estimator_that_cannot_run_gets_an_ineligible_row_with_its_reason():
    # `twfe` needs a panel. It is absent from the fit but present in the rows, so
    # its absence reads as absent rather than as fine.
    (cell, *_) = expand_cells(_config())
    rows = fit_cell(
        cell,
        root_entropy=0,
        fit_config=EstimatorConfig(random_seed=0, bootstrap_reps=50),
        candidates=resolve_candidates(("difference_in_means", "twfe")),
    )

    twfe = [row for row in rows if row["method"] == "twfe"]
    assert len(twfe) == 1
    assert twfe[0]["status"] == "ineligible" and twfe[0]["reason"]


def test_an_unknown_estimator_is_refused_by_name():
    with pytest.raises(ValueError, match="no_such_estimator"):
        resolve_candidates(("difference_in_means", "no_such_estimator"))


def test_no_estimator_list_means_every_registered_one():
    assert resolve_candidates(None) is None


def test_sort_result_frame_is_independent_of_input_order():
    config = _config()
    frame = run_grid(
        config,
        estimators=_ESTIMATORS,
        fit_config=EstimatorConfig(random_seed=0, bootstrap_reps=50),
    )
    shuffled = frame.sample(frac=1.0, random_state=np.random.RandomState(3))

    pd.testing.assert_frame_equal(sort_result_frame(shuffled), frame)
