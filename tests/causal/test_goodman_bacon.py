import numpy as np
import pytest
from cp_causal.estimators.did import TwoWayFixedEffects
from cp_causal.goodman_bacon import goodman_bacon_decomposition
from cp_core.values import EstimatorConfig
from cp_simulation.panel_scenarios import build_panel_scenario

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)


def generate(scenario_id, seed=0, n_units=400):
    return build_panel_scenario(scenario_id).generate(
        n_units=n_units, n_periods=8, rng=np.random.default_rng(seed)
    )


def test_weights_sum_to_one():
    decomposition = goodman_bacon_decomposition(generate("P3").dataset)

    assert sum(c.weight for c in decomposition.components) == pytest.approx(1.0)


def test_the_weighted_average_reproduces_the_twfe_estimate():
    # The decomposition is only meaningful if it adds back up to the thing it
    # decomposes. This is the test that makes the shares interpretable.
    panel = generate("P3")

    decomposition = goodman_bacon_decomposition(panel.dataset)
    twfe = TwoWayFixedEffects().fit(panel.dataset, CONFIG).point

    assert decomposition.weighted_average == pytest.approx(twfe, abs=0.05)


def test_a_uniform_design_has_no_already_treated_comparisons():
    # With one adoption period there is no earlier cohort to serve as a control,
    # which is exactly why the 2x2 is unbiased there.
    decomposition = goodman_bacon_decomposition(generate("P1").dataset)

    assert decomposition.already_treated_share == 0.0


def test_a_staggered_design_carries_already_treated_comparisons():
    decomposition = goodman_bacon_decomposition(generate("P3").dataset)

    assert decomposition.already_treated_share > 0.1
    kinds = {c.kind for c in decomposition.components}
    assert "already_treated_as_control" in kinds
    assert "never_treated_as_control" in kinds


def test_component_kinds_name_the_comparison_they_describe():
    decomposition = goodman_bacon_decomposition(generate("P3").dataset)

    for component in decomposition.components:
        assert component.kind in {
            "never_treated_as_control",
            "later_treated_as_control",
            "already_treated_as_control",
        }
        assert component.weight > 0
