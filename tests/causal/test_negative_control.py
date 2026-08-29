"""The negative-control check: refit on a column treatment does not affect.

The swap is the part most likely to rot silently, so it is pinned against a
direct fit of the same estimator on the same column. Phase 9 shipped a defect
of exactly that shape, where a falsification slot reported one estimator's
numbers under another's name.
"""

from dataclasses import replace

import numpy as np
import pytest
from cp_causal.registry import all_estimators
from cp_causal.sensitivity import falsify, negative_control_test
from cp_core.dataset import CausalDataset
from cp_core.values import EstimatorConfig
from cp_simulation.donor_scenarios import DONOR_SCENARIOS, build_donor_scenario
from cp_simulation.scenarios import DEFAULT_NEGATIVE_CONTROL, build_scenario

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)

CROSS_SECTIONAL = ("difference_in_means", "lin_regression", "ipw", "aipw")


def with_control(scenario_id="D", n_units=800, seed=0):
    return build_scenario(
        scenario_id, negative_control=DEFAULT_NEGATIVE_CONTROL
    ).generate(n_units=n_units, rng=np.random.default_rng(seed)).dataset


@pytest.mark.parametrize("estimator_name", CROSS_SECTIONAL)
def test_the_swap_reproduces_a_direct_fit_on_the_same_column(estimator_name):
    # The guard against a drifted swap. If this fails, the reported number
    # describes a fit nobody asked for.
    dataset = with_control()
    result = negative_control_test(dataset, estimator_name, CONFIG)
    assert result is not None

    direct_schema = replace(
        dataset.schema, outcome="outcome_nc", negative_control_outcome=None
    )
    direct = all_estimators()[estimator_name]().fit(
        CausalDataset(frame=dataset.frame, schema=direct_schema), CONFIG
    )
    assert result.estimate == pytest.approx(direct.point, rel=1e-9, abs=1e-12)
    assert result.outcome_column == "outcome_nc"
    assert result.estimator == estimator_name


def test_excludes_null_agrees_with_the_reported_interval():
    result = negative_control_test(with_control(), "lin_regression", CONFIG)
    assert result is not None
    assert result.excludes_null is not result.ci.contains(0.0)


def test_a_dataset_without_a_declared_control_returns_none():
    dataset = build_scenario("D").generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset
    assert negative_control_test(dataset, "lin_regression", CONFIG) is None


def test_a_panel_estimator_returns_none():
    spec = DONOR_SCENARIOS["S1"]
    dataset = build_donor_scenario("S1").generate(
        spec.n_units, spec.n_periods, np.random.default_rng(0)
    ).dataset
    assert negative_control_test(dataset, "synthetic_control", CONFIG) is None


def test_an_unknown_estimator_is_refused_rather_than_silently_none():
    with pytest.raises(KeyError):
        negative_control_test(with_control(), "lin_regresion", CONFIG)


def test_falsify_reports_the_negative_control_with_no_reason():
    result = falsify(with_control(), "lin_regression", CONFIG, n_permutations=20)
    assert result.negative_control is not None
    assert result.negative_control_reason is None


def test_falsify_states_a_reason_when_no_control_is_declared():
    dataset = build_scenario("D").generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset
    result = falsify(dataset, "lin_regression", CONFIG, n_permutations=20)
    assert result.negative_control is None
    assert result.negative_control_reason
    assert len(result.negative_control_reason.split()) >= 6


def test_falsify_states_a_family_reason_on_a_panel_estimator():
    spec = DONOR_SCENARIOS["S1"]
    dataset = build_donor_scenario("S1").generate(
        spec.n_units, spec.n_periods, np.random.default_rng(0)
    ).dataset
    result = falsify(dataset, "synthetic_control", CONFIG)
    assert result.negative_control is None
    assert "cross-section" in result.negative_control_reason
