"""Treatment is measured with error; the truth is kept, out of reach.

The rates are checked against their realised frequencies rather than against
the parameter that produced them, because a component that silently ignored
its own parameters would still pass a test that only read them back.
"""

import numpy as np
import pytest
from cp_simulation.components import NonDifferentialMisclassification
from cp_simulation.dgp import CrossSectionDGP
from cp_simulation.scenarios import build_scenario


def a_dgp(component):
    base = build_scenario("A")
    return CrossSectionDGP(
        unit_generator=base.unit_generator, baseline=base.baseline,
        effect=base.effect, assignment=base.assignment,
        scenario_id=base.scenario_id, misclassification=component,
    )


@pytest.mark.parametrize(("se", "sp"), [(-0.1, 0.9), (1.1, 0.9), (0.9, -0.1), (0.9, 1.1)])
def test_rates_outside_the_unit_interval_are_refused(se, sp):
    with pytest.raises(ValueError, match="must lie in"):
        NonDifferentialMisclassification(sensitivity=se, specificity=sp)


def test_rates_summing_to_one_or_less_are_refused():
    # At se + sp = 1 the observed label is independent of the true one, so the
    # component would be generating noise rather than mismeasurement.
    with pytest.raises(ValueError, match="sensitivity \\+ specificity"):
        NonDifferentialMisclassification(sensitivity=0.5, specificity=0.5)


def test_perfect_rates_reproduce_the_input_exactly():
    component = NonDifferentialMisclassification(sensitivity=1.0, specificity=1.0)
    treatment = np.array([0, 1, 1, 0, 1])
    observed = component.observe(treatment, np.random.default_rng(0))
    assert np.array_equal(observed, treatment)


def test_the_realised_rates_match_the_declared_ones():
    # Measured against realised frequencies: at 200_000 units the binomial
    # standard error on each rate is under 0.002, so 0.01 is ample headroom
    # and the test still fails if a rate is ignored or swapped.
    component = NonDifferentialMisclassification(sensitivity=0.8, specificity=0.95)
    rng = np.random.default_rng(0)
    treatment = rng.integers(0, 2, size=200_000)
    observed = component.observe(treatment, rng)
    realised_se = observed[treatment == 1].mean()
    realised_sp = 1.0 - observed[treatment == 0].mean()
    assert realised_se == pytest.approx(0.8, abs=0.01)
    assert realised_sp == pytest.approx(0.95, abs=0.01)


def test_the_frame_carries_the_observed_treatment_not_the_true_one():
    dgp = a_dgp(NonDifferentialMisclassification(sensitivity=0.7, specificity=0.9))
    simulated = dgp.generate(n_units=4000, rng=np.random.default_rng(0))
    frame_treatment = simulated.dataset.frame["treated"].to_numpy()
    assert not np.array_equal(frame_treatment, simulated.truth.true_treatment)


def test_the_true_treatment_is_recorded_in_ground_truth():
    dgp = a_dgp(NonDifferentialMisclassification(sensitivity=0.7, specificity=0.9))
    simulated = dgp.generate(n_units=500, rng=np.random.default_rng(0))
    assert simulated.truth.true_treatment is not None
    assert simulated.truth.true_treatment.shape == (500,)


def test_true_treatment_is_none_when_no_misclassification_is_configured():
    simulated = build_scenario("A").generate(
        n_units=200, rng=np.random.default_rng(0)
    )
    assert simulated.truth.true_treatment is None


def test_the_true_treatment_never_reaches_the_frame():
    dgp = a_dgp(NonDifferentialMisclassification(sensitivity=0.7, specificity=0.9))
    simulated = dgp.generate(n_units=500, rng=np.random.default_rng(0))
    assert set(simulated.dataset.frame.columns) == {
        "unit_id", "x1", "x2", "treated", "outcome"
    }


def test_the_outcome_is_realised_from_the_true_treatment():
    """The ordering that makes the recovery harness meaningful.

    Under heavy misclassification the naive difference in means on the
    OBSERVED treatment must be attenuated well below the true effect. If the
    outcome were realised from the observed treatment instead, no attenuation
    would exist and Task 9 would be asserting nothing.
    """
    dgp = a_dgp(NonDifferentialMisclassification(sensitivity=0.7, specificity=0.7))
    simulated = dgp.generate(n_units=40_000, rng=np.random.default_rng(0))
    frame = simulated.dataset.frame
    observed_gap = (
        frame.loc[frame["treated"] == 1, "outcome"].mean()
        - frame.loc[frame["treated"] == 0, "outcome"].mean()
    )
    assert abs(observed_gap) < abs(simulated.truth.true_ate)


def test_the_component_changes_the_config_hash():
    base = build_scenario("A")
    misclassified = a_dgp(
        NonDifferentialMisclassification(sensitivity=0.8, specificity=0.9)
    )
    assert misclassified.config_hash != base.config_hash
