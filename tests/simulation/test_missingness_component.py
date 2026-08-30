"""Missingness that hits its declared rate, and keeps the truth out of reach.

Rates are checked against realised frequencies rather than read back from the
parameters, because a component that ignored its own parameters would pass a
test that only re-read them.
"""

import numpy as np
import pytest
from cp_simulation.components import MissingnessMechanism
from cp_simulation.dgp import CrossSectionDGP
from cp_simulation.scenarios import build_scenario


def a_dgp(component):
    base = build_scenario("D")
    return CrossSectionDGP(
        unit_generator=base.unit_generator,
        baseline=base.baseline,
        effect=base.effect,
        assignment=base.assignment,
        scenario_id=base.scenario_id,
        missingness=component,
    )


@pytest.mark.parametrize("bad_rate", [-0.01, 1.0, 1.5])
def test_a_rate_outside_the_open_unit_interval_is_refused(bad_rate):
    with pytest.raises(ValueError, match="rate"):
        MissingnessMechanism(mechanism="MAR", rate=bad_rate)


def test_an_unknown_mechanism_is_refused():
    with pytest.raises(ValueError, match="mechanism"):
        MissingnessMechanism(mechanism="MCAR", rate=0.2)  # type: ignore[arg-type]


@pytest.mark.parametrize("mechanism", ["MAR", "MNAR"])
@pytest.mark.parametrize("rate", [0.1, 0.3])
@pytest.mark.parametrize("strength", [0.0, 1.0, 2.5])
def test_the_realised_rate_matches_the_declared_rate(mechanism, rate, strength):
    # At 200_000 units the binomial standard error on the rate is under 0.002,
    # so 0.01 is ample headroom and the test still fails if the intercept
    # solve is wrong or the rate is ignored.
    rng = np.random.default_rng(0)
    component = MissingnessMechanism(
        mechanism=mechanism, rate=rate, strength=strength
    )
    covariates = rng.normal(size=(200_000, 2))
    outcome = rng.binomial(1, 0.4, 200_000).astype(float)
    mask = component.missing_mask(covariates, outcome, rng)
    assert mask.mean() == pytest.approx(rate, abs=0.01)


@pytest.mark.parametrize("mechanism", ["MAR", "MNAR"])
def test_zero_strength_collapses_to_mcar(mechanism):
    # With no dependence, missingness must be independent of the outcome --
    # the baseline the measured harness compares the biased arms against.
    rng = np.random.default_rng(0)
    component = MissingnessMechanism(mechanism=mechanism, rate=0.3, strength=0.0)
    covariates = rng.normal(size=(200_000, 2))
    outcome = rng.binomial(1, 0.4, 200_000).astype(float)
    mask = component.missing_mask(covariates, outcome, rng)
    assert mask[outcome == 1].mean() == pytest.approx(
        mask[outcome == 0].mean(), abs=0.01
    )


def test_mnar_missingness_depends_on_the_outcome():
    rng = np.random.default_rng(0)
    component = MissingnessMechanism(mechanism="MNAR", rate=0.3, strength=2.0)
    covariates = rng.normal(size=(200_000, 2))
    outcome = rng.binomial(1, 0.4, 200_000).astype(float)
    mask = component.missing_mask(covariates, outcome, rng)
    # The whole point of MNAR: the missing rate differs by outcome value.
    assert abs(mask[outcome == 1].mean() - mask[outcome == 0].mean()) > 0.10


def test_mar_missingness_does_not_depend_on_the_outcome_given_the_covariates():
    # MAR's driver is the covariates. With covariates independent of the
    # outcome here, the realised rates by outcome value should match.
    rng = np.random.default_rng(0)
    component = MissingnessMechanism(mechanism="MAR", rate=0.3, strength=2.0)
    covariates = rng.normal(size=(200_000, 2))
    outcome = rng.binomial(1, 0.4, 200_000).astype(float)
    mask = component.missing_mask(covariates, outcome, rng)
    assert mask[outcome == 1].mean() == pytest.approx(
        mask[outcome == 0].mean(), abs=0.01
    )


def test_the_frame_carries_nan_and_ground_truth_keeps_the_complete_outcome():
    dgp = a_dgp(MissingnessMechanism(mechanism="MNAR", rate=0.25, strength=2.0))
    simulated = dgp.generate(n_units=4000, rng=np.random.default_rng(0))
    outcome = simulated.dataset.frame[simulated.dataset.schema.outcome]
    assert outcome.isna().any()
    assert simulated.truth.complete_outcome is not None
    assert not np.isnan(simulated.truth.complete_outcome).any()
    assert simulated.truth.complete_outcome.shape == (4000,)


def test_the_complete_outcome_never_reaches_the_frame():
    dgp = a_dgp(MissingnessMechanism(mechanism="MNAR", rate=0.25))
    simulated = dgp.generate(n_units=500, rng=np.random.default_rng(0))
    assert set(simulated.dataset.frame.columns) == {
        "unit_id",
        "x1",
        "x2",
        "treated",
        "outcome",
    }


def test_complete_outcome_is_none_when_no_missingness_is_configured():
    simulated = build_scenario("D").generate(
        n_units=200, rng=np.random.default_rng(0)
    )
    assert simulated.truth.complete_outcome is None
    assert not simulated.dataset.frame["outcome"].isna().any()


def test_the_component_changes_the_config_hash():
    base = build_scenario("D")
    with_missing = a_dgp(MissingnessMechanism(mechanism="MAR", rate=0.2))
    assert with_missing.config_hash != base.config_hash
