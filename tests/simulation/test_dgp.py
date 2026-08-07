import numpy as np
import pytest
from cp_simulation.components import (
    ConstantEffect,
    LogisticBaseline,
    PropensityAssignment,
    RandomAssignment,
    UnitGenerator,
)
from cp_simulation.dgp import CrossSectionDGP


def build_dgp(assignment, tau: float = 0.05) -> CrossSectionDGP:
    return CrossSectionDGP(
        unit_generator=UnitGenerator(covariate_names=("x1", "x2")),
        baseline=LogisticBaseline(intercept=-1.0, coefficients=(0.8, 0.5)),
        effect=ConstantEffect(tau=tau),
        assignment=assignment,
        scenario_id="TEST",
    )


def test_generate_returns_a_valid_causal_dataset():
    dgp = build_dgp(RandomAssignment(share=0.5))
    simulated = dgp.generate(n_units=500, rng=np.random.default_rng(0))

    frame = simulated.dataset.frame
    assert simulated.dataset.n_units == 500
    assert set(frame.columns) == {"unit_id", "x1", "x2", "treated", "outcome"}
    assert simulated.dataset.schema.outcome_type == "binary"


def test_true_ate_is_measured_not_declared():
    # The declared tau is an input; true_ate must be computed from realised
    # potential outcomes, so that clipping can never silently corrupt it.
    dgp = build_dgp(RandomAssignment(share=0.5), tau=0.05)
    simulated = dgp.generate(n_units=20000, rng=np.random.default_rng(1))

    assert simulated.truth.true_ate == pytest.approx(0.05, abs=1e-9)
    assert simulated.truth.true_cate.shape == (20000,)


def test_ground_truth_is_not_reachable_from_the_dataset_frame():
    dgp = build_dgp(RandomAssignment())
    simulated = dgp.generate(n_units=100, rng=np.random.default_rng(2))

    leaked = {"p0", "p1", "tau", "propensity", "y0", "y1"}
    assert leaked.isdisjoint(set(simulated.dataset.frame.columns))


def test_generation_is_reproducible_for_a_fixed_seed():
    dgp = build_dgp(RandomAssignment())
    first = dgp.generate(n_units=300, rng=np.random.default_rng(7))
    second = dgp.generate(n_units=300, rng=np.random.default_rng(7))

    np.testing.assert_array_equal(
        first.dataset.frame["outcome"].to_numpy(),
        second.dataset.frame["outcome"].to_numpy(),
    )
    assert first.truth.config_hash == second.truth.config_hash


def test_config_hash_changes_when_configuration_changes():
    first = build_dgp(RandomAssignment(), tau=0.05).generate(
        n_units=100, rng=np.random.default_rng(0)
    )
    second = build_dgp(RandomAssignment(), tau=0.03).generate(
        n_units=100, rng=np.random.default_rng(0)
    )

    assert first.truth.config_hash != second.truth.config_hash


def test_true_att_equals_true_ate_under_random_assignment():
    dgp = build_dgp(RandomAssignment(share=0.5))
    simulated = dgp.generate(n_units=20000, rng=np.random.default_rng(3))

    assert simulated.truth.true_att == pytest.approx(simulated.truth.true_ate, abs=1e-9)


def test_confounded_assignment_records_its_confounders():
    dgp = build_dgp(PropensityAssignment(coefficients=(0.9, 0.7), strength=1.0))
    simulated = dgp.generate(n_units=1000, rng=np.random.default_rng(4))

    assert simulated.truth.true_confounders == ("x1", "x2")
    assert simulated.truth.true_propensity.shape == (1000,)


def test_random_assignment_records_no_confounders():
    dgp = build_dgp(RandomAssignment(share=0.5))
    simulated = dgp.generate(n_units=500, rng=np.random.default_rng(20))

    assert simulated.truth.true_confounders == ()


def test_zero_coefficient_covariate_is_not_named_a_confounder():
    # x1 does not enter the propensity index at all, so calling it a confounder
    # would misreport the identification problem the scenario actually poses.
    dgp = build_dgp(PropensityAssignment(coefficients=(0.0, 0.7), strength=1.0))
    simulated = dgp.generate(n_units=500, rng=np.random.default_rng(21))

    assert simulated.truth.true_confounders == ("x2",)


def test_zero_strength_propensity_records_no_confounders():
    # strength=0 collapses the propensity to a constant: assignment stops
    # depending on covariates, so the scenario is unconfounded despite being
    # built from a PropensityAssignment.
    dgp = build_dgp(PropensityAssignment(coefficients=(0.9, 0.7), strength=0.0))
    simulated = dgp.generate(n_units=500, rng=np.random.default_rng(22))

    assert simulated.truth.true_confounders == ()
    np.testing.assert_allclose(simulated.truth.true_propensity, 0.5)


def test_generate_raises_on_zero_units():
    dgp = build_dgp(RandomAssignment())
    with pytest.raises(ValueError, match="n_units"):
        dgp.generate(n_units=0, rng=np.random.default_rng(0))


def test_generate_raises_on_negative_units():
    dgp = build_dgp(RandomAssignment())
    with pytest.raises(ValueError, match="n_units"):
        dgp.generate(n_units=-3, rng=np.random.default_rng(0))
