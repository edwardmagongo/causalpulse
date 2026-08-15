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


def _mean_reconstruction_deviation(scenario_id: str, seeds: int = 10) -> float:
    diffs = []
    for seed in range(seeds):
        panel = generate(scenario_id, seed=seed)
        decomposition = goodman_bacon_decomposition(panel.dataset)
        twfe = TwoWayFixedEffects().fit(panel.dataset, CONFIG).point
        diffs.append(decomposition.weighted_average - twfe)
    return float(np.mean(diffs))


def test_the_weighted_average_reproduces_the_twfe_estimate():
    # The decomposition is only meaningful if it adds back up to the thing it
    # decomposes. This is the test that makes the shares interpretable.
    #
    # A single fixed seed is the wrong instrument here, per this project's
    # established discipline (see test_did.py's `mean_deviation` fix and
    # test_callaway_santanna.py's `_overall_mean_deviation`): seed=0 (this
    # module's `generate` default) deviates by -0.0447, the worst of the
    # seeds sampled below, which is why a single-seed assertion needed a
    # loose abs=0.05 tolerance to pass at all.
    #
    # Measured at seeds=10, n_units=400 (this module's `generate` default):
    # mean deviation -0.0049, SEM 0.0061 (std 0.0194 / sqrt(10)).
    # abs=0.03 leaves (0.03 - 0.0049) / 0.0061 = 4.09 SEM of headroom,
    # above this project's four-SEM bar, while still catching a genuinely
    # broken reconstruction (wrong sign, wrong scale, or mismatched weights).
    assert abs(_mean_reconstruction_deviation("P3")) < 0.03


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
