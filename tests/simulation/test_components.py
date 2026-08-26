import numpy as np
import pytest
from cp_simulation.components import (
    BaselineSelectionAssignment,
    ConstantEffect,
    HeterogeneousEffect,
    LogisticBaseline,
    PropensityAssignment,
    RandomAssignment,
    UnitGenerator,
)
from cp_simulation.truth import GroundTruth


def test_unit_generator_produces_named_covariates():
    generator = UnitGenerator(covariate_names=("x1", "x2"))
    frame = generator.generate(n_units=100, rng=np.random.default_rng(0))

    assert list(frame.columns) == ["unit_id", "x1", "x2"]
    assert frame["unit_id"].nunique() == 100


def test_unit_generator_is_reproducible_for_a_fixed_seed():
    generator = UnitGenerator(covariate_names=("x1",))
    first = generator.generate(n_units=50, rng=np.random.default_rng(42))
    second = generator.generate(n_units=50, rng=np.random.default_rng(42))

    np.testing.assert_array_equal(first["x1"].to_numpy(), second["x1"].to_numpy())


def test_baseline_probabilities_respect_the_boundary_constraint():
    # Clipping p0 + tau at 1 distorts the true effect, so baselines are bounded.
    generator = UnitGenerator(covariate_names=("x1", "x2"))
    frame = generator.generate(n_units=5000, rng=np.random.default_rng(1))
    baseline = LogisticBaseline(intercept=-1.0, coefficients=(0.8, 0.5))

    p0 = baseline.probabilities(frame[["x1", "x2"]].to_numpy())

    assert p0.min() >= 0.05
    assert p0.max() <= 0.85


def test_constant_effect_returns_the_requested_tau_everywhere():
    effect = ConstantEffect(tau=0.05)
    covariates = np.zeros((10, 2))

    np.testing.assert_allclose(effect.tau_values(covariates), np.full(10, 0.05))


def test_heterogeneous_effect_varies_by_covariate():
    # New users (x=1) get +8pp, returning users (x=0) get +1pp.
    effect = HeterogeneousEffect(base_tau=0.01, covariate_index=0, increment=0.07)
    covariates = np.array([[1.0], [0.0], [1.0]])

    np.testing.assert_allclose(effect.tau_values(covariates), [0.08, 0.01, 0.08])


def test_heterogeneous_effect_validates_covariate_index():
    effect = HeterogeneousEffect(base_tau=0.01, covariate_index=5, increment=0.07)
    covariates = np.array([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match=r"covariate_index must lie in \[0, 2\), got 5"):
        effect.tau_values(covariates)


def test_heterogeneous_effect_rejects_negative_covariate_index():
    effect = HeterogeneousEffect(base_tau=0.01, covariate_index=-1, increment=0.07)
    covariates = np.array([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match=r"covariate_index must lie in \[0, 2\), got -1"):
        effect.tau_values(covariates)


def test_random_assignment_is_independent_of_covariates():
    rng = np.random.default_rng(2)
    covariates = rng.normal(size=(20000, 1))
    baseline = np.full(covariates.shape[0], 0.5)
    treatment, propensity = RandomAssignment(share=0.5).assign(covariates, baseline, rng)

    assert np.allclose(propensity, 0.5)
    correlation = np.corrcoef(treatment, covariates[:, 0])[0, 1]
    assert abs(correlation) < 0.03


def test_propensity_assignment_correlates_treatment_with_covariates():
    rng = np.random.default_rng(3)
    covariates = rng.normal(size=(20000, 1))
    baseline = np.full(covariates.shape[0], 0.5)
    treatment, propensity = PropensityAssignment(
        coefficients=(1.0,), strength=1.0
    ).assign(covariates, baseline, rng)

    assert np.corrcoef(treatment, covariates[:, 0])[0, 1] > 0.2
    assert propensity.min() > 0.0 and propensity.max() < 1.0


def test_propensity_assignment_with_zero_strength_is_random():
    rng = np.random.default_rng(4)
    covariates = rng.normal(size=(10000, 1))
    baseline = np.full(covariates.shape[0], 0.5)
    _, propensity = PropensityAssignment(coefficients=(1.0,), strength=0.0).assign(
        covariates, baseline, rng
    )

    np.testing.assert_allclose(propensity, 0.5)


def test_ground_truth_is_frozen():
    truth = GroundTruth(
        true_ate=0.05,
        true_att=0.05,
        true_cate=np.full(3, 0.05),
        true_propensity=np.full(3, 0.5),
        true_confounders=(),
        scenario_id="A",
        config_hash="abc123",
    )
    with pytest.raises(AttributeError):
        truth.true_ate = 0.1  # type: ignore[misc]


def test_random_assignment_is_randomised():
    assert RandomAssignment().provenance() == "randomised"


def test_propensity_assignment_is_observational():
    assert PropensityAssignment(coefficients=(0.5, 0.5)).provenance() == "observational"


def test_propensity_assignment_at_zero_strength_is_randomised():
    # Matches `confounders()`, which already returns () at zero strength:
    # at strength 0 the propensity is a constant and assignment is a coin flip.
    # The two declarations must not disagree about the same mechanism.
    zero = PropensityAssignment(coefficients=(0.5, 0.5), strength=0.0)
    assert zero.confounders(("x1", "x2")) == ()
    assert zero.provenance() == "randomised"


def test_baseline_selection_is_observational():
    assert BaselineSelectionAssignment().provenance() == "observational"


def test_baseline_selection_at_zero_strength_is_randomised():
    zero = BaselineSelectionAssignment(strength=0.0)
    assert zero.confounders(("x1", "x2")) == ()
    assert zero.provenance() == "randomised"
