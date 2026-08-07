import numpy as np
import pandas as pd
import pytest
from cp_core.dataset import CausalDataset, CausalSchema
from cp_simulation.oracle import naive_difference_in_means, oracle_ate, oracle_att
from cp_simulation.scenarios import SCENARIOS, build_scenario
from cp_simulation.truth import GroundTruth, SimulatedDataset


@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_oracle_recovers_the_declared_effect(scenario_id: str):
    spec = SCENARIOS[scenario_id]
    simulated = build_scenario(scenario_id).generate(
        n_units=40000, rng=np.random.default_rng(0)
    )

    assert oracle_ate(simulated) == pytest.approx(spec.declared_effect, abs=0.002)


@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_oracle_att_matches_ground_truth(scenario_id: str):
    simulated = build_scenario(scenario_id).generate(
        n_units=40000, rng=np.random.default_rng(0)
    )

    assert oracle_att(simulated) == pytest.approx(simulated.truth.true_att)


@pytest.mark.parametrize("scenario_id", ["A", "B", "C"])
def test_naive_estimator_is_unbiased_under_randomisation(scenario_id: str):
    spec = SCENARIOS[scenario_id]
    estimates = [
        naive_difference_in_means(
            build_scenario(scenario_id)
            .generate(n_units=8000, rng=np.random.default_rng(seed))
            .dataset
        )
        for seed in range(40)
    ]
    mean_estimate = float(np.mean(estimates))
    monte_carlo_se = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))

    assert abs(mean_estimate - spec.declared_effect) < 4 * monte_carlo_se


def test_naive_estimator_is_biased_upward_under_confounding():
    # Scenario D is the reason propensity and doubly-robust methods exist. If
    # the naive estimator were unbiased here, the DGP would not be confounded.
    estimates = [
        naive_difference_in_means(
            build_scenario("D", confounding_strength=1.0)
            .generate(n_units=8000, rng=np.random.default_rng(seed))
            .dataset
        )
        for seed in range(40)
    ]
    bias = float(np.mean(estimates)) - SCENARIOS["D"].declared_effect

    assert bias > 0.05


def test_naive_bias_grows_with_confounding_strength():
    def mean_bias(strength: float) -> float:
        estimates = [
            naive_difference_in_means(
                build_scenario("D", confounding_strength=strength)
                .generate(n_units=8000, rng=np.random.default_rng(seed))
                .dataset
            )
            for seed in range(20)
        ]
        return float(np.mean(estimates)) - SCENARIOS["D"].declared_effect

    assert mean_bias(2.0) > mean_bias(1.0) > mean_bias(0.25)


def test_scenario_b_naive_estimator_does_not_systematically_find_an_effect():
    estimates = [
        naive_difference_in_means(
            build_scenario("B")
            .generate(n_units=8000, rng=np.random.default_rng(seed))
            .dataset
        )
        for seed in range(60)
    ]
    monte_carlo_se = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))

    assert abs(float(np.mean(estimates))) < 4 * monte_carlo_se


# Guard tests for oracle_ate
def test_oracle_ate_rejects_empty_cate_vector():
    """Guard 1: oracle_ate must refuse an empty true_cate vector."""
    truth = GroundTruth(
        true_ate=0.0,
        true_att=0.0,
        true_cate=np.array([]),
        true_propensity=np.array([]),
        true_confounders=(),
        scenario_id="test",
        config_hash="test_hash",
    )
    frame = pd.DataFrame({"unit": [], "outcome": [], "treatment": []})
    schema = CausalSchema(
        unit="unit",
        treatment="treatment",
        outcome="outcome",
        covariates=(),
        outcome_type="continuous",
    )
    dataset = CausalDataset(frame=frame, schema=schema)
    simulated = SimulatedDataset(dataset=dataset, truth=truth)
    with pytest.raises(ValueError, match="true_cate is empty or undefined"):
        oracle_ate(simulated)


def test_oracle_ate_rejects_nan_cate():
    """Guard 1: oracle_ate must refuse a nan true_cate mean."""
    truth = GroundTruth(
        true_ate=0.0,
        true_att=float("nan"),
        true_cate=np.array([float("nan")]),
        true_propensity=np.array([]),
        true_confounders=(),
        scenario_id="test",
        config_hash="test_hash",
    )
    frame = pd.DataFrame({"unit": [0], "outcome": [1.0], "treatment": [0]})
    schema = CausalSchema(
        unit="unit",
        treatment="treatment",
        outcome="outcome",
        covariates=(),
        outcome_type="continuous",
    )
    dataset = CausalDataset(frame=frame, schema=schema)
    simulated = SimulatedDataset(dataset=dataset, truth=truth)
    with pytest.raises(ValueError, match="true_cate is empty or undefined"):
        oracle_ate(simulated)


# Guard tests for oracle_att
def test_oracle_att_rejects_nan_att():
    """Guard: oracle_att must refuse a nan true_att (no treated units drawn)."""
    truth = GroundTruth(
        true_ate=0.0,
        true_att=float("nan"),
        true_cate=np.array([0.0]),
        true_propensity=np.array([]),
        true_confounders=(),
        scenario_id="test",
        config_hash="test_hash",
    )
    frame = pd.DataFrame({"unit": [0], "outcome": [1.0], "treatment": [0]})
    schema = CausalSchema(
        unit="unit",
        treatment="treatment",
        outcome="outcome",
        covariates=(),
        outcome_type="continuous",
    )
    dataset = CausalDataset(frame=frame, schema=schema)
    simulated = SimulatedDataset(dataset=dataset, truth=truth)
    with pytest.raises(ValueError, match="true_att"):
        oracle_att(simulated)


# Guard tests for naive_difference_in_means
def test_naive_difference_in_means_rejects_empty_treated_arm():
    """Guard 2: naive_difference_in_means must reject an empty treated arm."""
    frame = pd.DataFrame(
        {
            "unit": [0, 1, 2],
            "outcome": [1.0, 2.0, 3.0],
            "treatment": [0, 0, 0],  # All control, no treated
        }
    )
    schema = CausalSchema(
        unit="unit",
        treatment="treatment",
        outcome="outcome",
        covariates=(),
        outcome_type="continuous",
    )
    dataset = CausalDataset(frame=frame, schema=schema)
    with pytest.raises(ValueError, match="no treated units"):
        naive_difference_in_means(dataset)


def test_naive_difference_in_means_rejects_empty_control_arm():
    """Guard 2: naive_difference_in_means must reject an empty control arm."""
    frame = pd.DataFrame(
        {
            "unit": [0, 1, 2],
            "outcome": [1.0, 2.0, 3.0],
            "treatment": [1, 1, 1],  # All treated, no control
        }
    )
    schema = CausalSchema(
        unit="unit",
        treatment="treatment",
        outcome="outcome",
        covariates=(),
        outcome_type="continuous",
    )
    dataset = CausalDataset(frame=frame, schema=schema)
    with pytest.raises(ValueError, match="no control units"):
        naive_difference_in_means(dataset)
