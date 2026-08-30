"""Estimators refuse missing data, from two shared choke points.

Before this, five different failures surfaced from five different places --
`cannot convert float NaN to integer`, two `Interval bounds must be finite`,
a scikit-learn `Input y contains NaN`, and a bare `SVD did not converge` --
none of which named the column at fault. One refusal replaces all five.
"""

import numpy as np
import pytest
from cp_causal._panel import panel_arrays
from cp_causal._support import arrays, require_no_missing
from cp_causal.registry import all_estimators
from cp_core.dataset import CausalDataset
from cp_core.values import EstimatorConfig
from cp_simulation.scenarios import build_scenario

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)
CROSS_SECTIONAL = ("difference_in_means", "lin_regression", "ipw", "aipw", "causal_forest")


def with_missing_outcome(column: str = "outcome"):
    simulated = build_scenario("D").generate(
        n_units=400, rng=np.random.default_rng(0)
    )
    frame = simulated.dataset.frame.copy()
    frame.loc[frame.index[:40], column] = np.nan
    return CausalDataset(frame=frame, schema=simulated.dataset.schema)


def test_the_two_choke_points_are_independent():
    # panel_arrays must not delegate to arrays; if it ever does, one of the
    # two call sites becomes dead and a later edit could remove the live one.
    import inspect

    source = inspect.getsource(panel_arrays)
    assert "arrays(" not in source.replace("panel_arrays(", "")


@pytest.mark.parametrize("estimator_name", CROSS_SECTIONAL)
def test_every_cross_sectional_estimator_refuses_a_missing_outcome(estimator_name):
    dataset = with_missing_outcome()
    with pytest.raises(ValueError, match="missing values"):
        all_estimators()[estimator_name]().fit(dataset, CONFIG)


def test_the_refusal_names_the_offending_column():
    with pytest.raises(ValueError, match="outcome"):
        arrays(with_missing_outcome())


def test_the_refusal_names_a_missing_covariate():
    with pytest.raises(ValueError, match="x1"):
        arrays(with_missing_outcome(column="x1"))


def test_the_refusal_points_at_the_remedy():
    with pytest.raises(ValueError, match="complete_cases"):
        arrays(with_missing_outcome())


def test_a_complete_dataset_still_passes_through():
    dataset = build_scenario("D").generate(
        n_units=200, rng=np.random.default_rng(0)
    ).dataset
    covariates, treatment, outcome = arrays(dataset)
    assert covariates.shape[0] == treatment.shape[0] == outcome.shape[0] == 200


def test_the_panel_choke_point_refuses_too():
    """Built by hand, because no panel scenario produces a missing value.

    The missingness component attaches to the cross-sectional DGP only, so a
    test relying on a scenario here would pass vacuously -- and the panel path
    is the one real data hits first, since attrition is a panel phenomenon.
    """
    from cp_simulation.panel_scenarios import build_panel_scenario

    simulated = build_panel_scenario("P1").generate(
        n_units=40, n_periods=5, rng=np.random.default_rng(0)
    )
    frame = simulated.dataset.frame.copy()
    frame.loc[frame.index[:10], simulated.dataset.schema.outcome] = np.nan
    dataset = CausalDataset(frame=frame, schema=simulated.dataset.schema)
    with pytest.raises(ValueError, match="missing values"):
        panel_arrays(dataset)


def test_a_panel_estimator_refuses_through_the_panel_choke_point():
    from cp_simulation.panel_scenarios import build_panel_scenario

    simulated = build_panel_scenario("P1").generate(
        n_units=40, n_periods=5, rng=np.random.default_rng(0)
    )
    frame = simulated.dataset.frame.copy()
    frame.loc[frame.index[:10], simulated.dataset.schema.outcome] = np.nan
    dataset = CausalDataset(frame=frame, schema=simulated.dataset.schema)
    with pytest.raises(ValueError, match="missing values"):
        all_estimators()["twfe"]().fit(dataset, CONFIG)


def test_require_no_missing_accepts_a_complete_dataset():
    dataset = build_scenario("A").generate(
        n_units=100, rng=np.random.default_rng(0)
    ).dataset
    assert require_no_missing(dataset) is None


PANEL_ESTIMATORS = ("twfe", "did_2x2", "event_study")
STAGGERED_ESTIMATORS = ("callaway_santanna", "hierarchical_did")
DONOR_ESTIMATORS = ("synthetic_control", "bsts")


def _holed(dataset, n_rows=10):
    """The same dataset with its outcome punched out on the first rows."""
    frame = dataset.frame.copy()
    frame.loc[frame.index[:n_rows], dataset.schema.outcome] = np.nan
    return CausalDataset(frame=frame, schema=dataset.schema)


def _shaped_for(estimator_name):
    """A dataset of the shape the named estimator actually accepts.

    One fixture per family rather than one for all twelve: a donor-pool
    estimator refuses a multi-treated panel on its own requirements, which
    would make the refusal below pass for the wrong reason.
    """
    from cp_simulation.donor_scenarios import DONOR_SCENARIOS, build_donor_scenario
    from cp_simulation.panel_scenarios import build_panel_scenario

    if estimator_name in DONOR_ESTIMATORS:
        spec = DONOR_SCENARIOS["S1"]
        return build_donor_scenario("S1").generate(
            spec.n_units, spec.n_periods, np.random.default_rng(0)
        ).dataset
    if estimator_name in STAGGERED_ESTIMATORS:
        return build_panel_scenario("P3").generate(
            n_units=120, n_periods=10, rng=np.random.default_rng(0)
        ).dataset
    if estimator_name in PANEL_ESTIMATORS:
        return build_panel_scenario("P1").generate(
            n_units=60, n_periods=6, rng=np.random.default_rng(0)
        ).dataset
    return build_scenario("D").generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset


@pytest.mark.parametrize(
    "estimator_name",
    CROSS_SECTIONAL + PANEL_ESTIMATORS + STAGGERED_ESTIMATORS + DONOR_ESTIMATORS,
)
def test_every_registered_estimator_refuses_missing_data(estimator_name):
    """All twelve, not the six the tests above happen to reach.

    The two choke points are claimed to cover five estimators through
    `arrays` and seven through `panel_arrays`. That claim is what this pins:
    an estimator added later that reads its frame directly, or an existing
    one refactored onto a third path, passes every other test in this file
    and fails only here. Phase 5 shipped a registry test that passed while
    the real import path was broken, and Phase 6 broke an exact-set
    assertion by registering an estimator -- both are this shape.

    Costs nothing despite covering two MCMC estimators: the guard raises
    before any sampler is constructed.
    """
    dataset = _holed(_shaped_for(estimator_name))
    with pytest.raises(ValueError, match="missing values"):
        all_estimators()[estimator_name]().fit(dataset, CONFIG)


def test_the_registered_set_is_exactly_the_estimators_covered_above():
    # If a thirteenth estimator is registered, the parametrization above is
    # silently incomplete. This is the assertion that notices.
    covered = frozenset(
        CROSS_SECTIONAL + PANEL_ESTIMATORS + STAGGERED_ESTIMATORS + DONOR_ESTIMATORS
    )
    assert covered == frozenset(all_estimators())
