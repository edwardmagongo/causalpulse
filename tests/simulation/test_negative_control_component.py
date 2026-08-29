"""The negative-control outcome shares the confounders and ignores treatment.

The guarantee is structural, not statistical: treatment is not an argument to
the component, so no configuration of it can make the column depend on
treatment. The tests below check both that structural fact and its observable
consequence.
"""

import inspect

import numpy as np
import pytest
from cp_simulation.components import NegativeControlOutcome
from cp_simulation.dgp import CrossSectionDGP
from cp_simulation.scenarios import build_scenario


def test_treatment_is_not_an_argument_to_the_component():
    # The construction guarantee, checked structurally. If treatment ever
    # becomes a parameter here, the column stops being a negative control.
    parameters = inspect.signature(NegativeControlOutcome.probabilities).parameters
    assert set(parameters) == {"self", "covariates"}


def test_probabilities_depend_on_the_covariates():
    component = NegativeControlOutcome(intercept=-0.5, coefficients=(1.2, -0.8))
    low = component.probabilities(np.array([[0.0, 0.0]]))
    high = component.probabilities(np.array([[2.0, 0.0]]))
    assert high[0] > low[0]


def test_a_covariate_count_mismatch_is_refused():
    component = NegativeControlOutcome(intercept=0.0, coefficients=(1.0, 1.0))
    with pytest.raises(ValueError, match="expected 2 covariates"):
        component.probabilities(np.array([[0.0]]))


def test_probabilities_stay_inside_the_unit_interval():
    component = NegativeControlOutcome(intercept=0.0, coefficients=(50.0, -50.0))
    values = component.probabilities(np.array([[10.0, -10.0], [-10.0, 10.0]]))
    assert np.all((values > 0.0) & (values < 1.0))


def test_the_dgp_adds_the_column_only_when_the_component_is_present():
    without = build_scenario("A").generate(
        n_units=200, rng=np.random.default_rng(0)
    ).dataset
    assert "outcome_nc" not in without.frame.columns
    assert without.schema.negative_control_outcome is None


def test_the_dgp_declares_the_column_on_the_schema_when_present():
    base = build_scenario("A")
    dgp = CrossSectionDGP(
        unit_generator=base.unit_generator,
        baseline=base.baseline,
        effect=base.effect,
        assignment=base.assignment,
        scenario_id=base.scenario_id,
        negative_control=NegativeControlOutcome(
            intercept=-0.4, coefficients=(0.9, 0.4)
        ),
    )
    simulated = dgp.generate(n_units=200, rng=np.random.default_rng(0))
    assert simulated.dataset.schema.negative_control_outcome == "outcome_nc"
    assert "outcome_nc" in simulated.dataset.frame.columns


def test_the_negative_control_column_is_binary():
    base = build_scenario("A")
    dgp = CrossSectionDGP(
        unit_generator=base.unit_generator, baseline=base.baseline,
        effect=base.effect, assignment=base.assignment,
        scenario_id=base.scenario_id,
        negative_control=NegativeControlOutcome(
            intercept=-0.4, coefficients=(0.9, 0.4)
        ),
    )
    values = set(
        dgp.generate(n_units=200, rng=np.random.default_rng(0)).dataset.frame["outcome_nc"]
    )
    assert values <= {0, 1}


def test_the_component_changes_the_config_hash():
    # Two DGPs differing only by the negative control must not record the same
    # configuration hash, or a reproduction would silently rebuild the wrong one.
    base = build_scenario("A")
    with_nc = CrossSectionDGP(
        unit_generator=base.unit_generator, baseline=base.baseline,
        effect=base.effect, assignment=base.assignment,
        scenario_id=base.scenario_id,
        negative_control=NegativeControlOutcome(intercept=0.0, coefficients=(1.0, 1.0)),
    )
    assert with_nc.config_hash != base.config_hash
