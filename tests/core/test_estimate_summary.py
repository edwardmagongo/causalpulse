import dataclasses
import json

import numpy as np
from cp_causal.registry import get_estimator
from cp_core.results import estimate_summary
from cp_core.values import EstimatorConfig
from cp_simulation.params import ScenarioParams
from cp_simulation.scenarios import build_scenario


def _fit():
    dataset = build_scenario("D", ScenarioParams()).generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset
    return get_estimator("lin_regression")().fit(dataset, EstimatorConfig(random_seed=0))


def test_the_summary_matches_the_estimate_and_survives_json():
    estimate = _fit()

    summary = estimate_summary(estimate)

    assert json.loads(json.dumps(summary)) == summary
    assert summary["method"] == "lin_regression"
    assert summary["point"] == estimate.point
    assert summary["ci_lower"] == estimate.ci.lower
    assert summary["ci_upper"] == estimate.ci.upper
    assert summary["ci_level"] == estimate.ci.level
    # The fit must carry a bootstrap interval, or `None == None` would pass.
    assert estimate.bootstrap_ci is not None
    assert summary["bootstrap_ci_level"] == estimate.bootstrap_ci.level
    assert summary["scale"] == estimate.scale
    assert type(summary["n_treated"]) is int


def test_absent_optional_values_are_none_not_nan():
    estimate = dataclasses.replace(_fit(), se=None, bootstrap_ci=None, p_value=None)

    summary = estimate_summary(estimate)

    assert summary["se"] is None
    assert summary["bootstrap_ci_lower"] is None
    assert summary["bootstrap_ci_upper"] is None
    assert summary["bootstrap_ci_level"] is None
    assert summary["p_value"] is None
