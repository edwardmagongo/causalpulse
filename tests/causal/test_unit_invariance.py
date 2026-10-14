"""An estimate does not depend on the units its data are given in.

A covariate in other units (dollars or cents) is the same covariate: every
cross-sectional estimator must give the same point and interval. The outcome in
other units is the same outcome: the point and interval scale with it. Before
2026-10-14 `lin_regression`, `aipw` and `causal_forest` failed the first,
through least squares that cut small directions once one column dwarfed the
others.
"""

import dataclasses

import cp_causal  # noqa: F401  (registers the estimators)
import numpy as np
import pytest
from cp_causal.registry import get_estimator
from cp_core.dataset import CausalDataset
from cp_core.results import EffectEstimate
from cp_core.values import EstimatorConfig
from cp_simulation.params import ScenarioParams
from cp_simulation.scenarios import build_scenario

ESTIMATORS = ["difference_in_means", "lin_regression", "ipw", "aipw", "causal_forest"]


@pytest.fixture(scope="module")
def dataset() -> CausalDataset:
    return build_scenario("D", ScenarioParams()).generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset


@pytest.fixture(scope="module")
def continuous(dataset: CausalDataset) -> CausalDataset:
    """The same units with a continuous outcome: every scenario's is binary."""
    rng = np.random.default_rng(1)
    frame = dataset.frame.copy()
    frame[dataset.schema.outcome] = (
        2.0 * frame["x1"]
        - frame["x2"]
        + 0.5 * frame[dataset.schema.treatment]
        + rng.normal(size=len(frame))
    )
    schema = dataclasses.replace(dataset.schema, outcome_type="continuous")
    return CausalDataset(frame=frame, schema=schema)


def fit(name: str, dataset: CausalDataset) -> EffectEstimate:
    return get_estimator(name)().fit(dataset, EstimatorConfig())


def times(dataset: CausalDataset, column: str, factor: float) -> CausalDataset:
    frame = dataset.frame.copy()
    frame[column] = frame[column] * factor
    return CausalDataset(frame=frame, schema=dataset.schema)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("factor", [1e2, 1e6, 1e9])
def test_a_covariate_in_other_units_gives_the_same_estimate(name, factor, dataset):
    a = fit(name, dataset)
    b = fit(name, times(dataset, dataset.schema.covariates[0], factor))

    assert b.point == pytest.approx(a.point, rel=1e-6)
    assert b.ci.lower == pytest.approx(a.ci.lower, rel=1e-6)
    assert b.ci.upper == pytest.approx(a.ci.upper, rel=1e-6)


@pytest.mark.parametrize("name", ESTIMATORS)
def test_the_outcome_in_other_units_scales_the_estimate(name, continuous):
    a = fit(name, continuous)
    b = fit(name, times(continuous, continuous.schema.outcome, 1e3))

    assert b.point == pytest.approx(1e3 * a.point, rel=1e-6)
    assert b.ci.lower == pytest.approx(1e3 * a.ci.lower, rel=1e-6)
    assert b.ci.upper == pytest.approx(1e3 * a.ci.upper, rel=1e-6)
