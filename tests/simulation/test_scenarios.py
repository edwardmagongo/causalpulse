import numpy as np
import pytest
from cp_simulation.scenarios import SCENARIOS, build_scenario


def test_all_scenarios_are_registered():
    assert sorted(SCENARIOS) == ["A", "B", "C", "D", "F"]


@pytest.mark.parametrize(
    ("scenario_id", "declared"),
    [("A", 0.05), ("B", 0.0), ("C", -0.03), ("D", 0.05)],
)
def test_measured_ate_matches_the_declared_label(scenario_id: str, declared: float):
    # A scenario advertising "+5pp" must actually deliver +5pp, or every
    # downstream bias number is wrong.
    dgp = build_scenario(scenario_id)
    simulated = dgp.generate(n_units=40000, rng=np.random.default_rng(0))

    assert simulated.truth.true_ate == pytest.approx(declared, abs=0.002)


def test_scenario_f_is_registered_with_a_heterogeneous_effect():
    simulated = build_scenario("F").generate(n_units=20000, rng=np.random.default_rng(0))
    frame = simulated.dataset.frame
    cate = simulated.truth.true_cate

    new_users = frame["x2"].to_numpy() == 1
    np.testing.assert_allclose(cate[new_users], 0.08, atol=1e-12)
    np.testing.assert_allclose(cate[~new_users], 0.01, atol=1e-12)


def test_scenario_f_declared_effect_is_the_population_average():
    # The declared label for a heterogeneous scenario is the mean CATE, which
    # depends on the covariate mix — so it must be measured, not asserted equal
    # to either subgroup effect.
    spec = SCENARIOS["F"]
    simulated = build_scenario("F").generate(n_units=40000, rng=np.random.default_rng(1))

    assert simulated.truth.true_ate == pytest.approx(spec.declared_effect, abs=0.002)
    assert 0.01 < simulated.truth.true_ate < 0.08


def test_scenario_f_has_no_confounders():
    simulated = build_scenario("F").generate(n_units=1000, rng=np.random.default_rng(2))

    assert simulated.truth.true_confounders == ()


def test_scenario_b_produces_exactly_zero_effect():
    dgp = build_scenario("B")
    simulated = dgp.generate(n_units=5000, rng=np.random.default_rng(1))

    np.testing.assert_allclose(simulated.truth.true_cate, 0.0, atol=1e-12)


def test_scenario_c_is_harmful():
    dgp = build_scenario("C")
    simulated = dgp.generate(n_units=5000, rng=np.random.default_rng(2))

    assert simulated.truth.true_ate < 0


def test_scenario_a_has_no_confounders_but_scenario_d_does():
    a = build_scenario("A").generate(n_units=1000, rng=np.random.default_rng(3))
    d = build_scenario("D").generate(n_units=1000, rng=np.random.default_rng(3))

    assert a.truth.true_confounders == ()
    assert d.truth.true_confounders == ("x1", "x2")


def test_confounding_strength_controls_treatment_covariate_dependence():
    weak = build_scenario("D", confounding_strength=0.2).generate(
        n_units=20000, rng=np.random.default_rng(4)
    )
    strong = build_scenario("D", confounding_strength=2.0).generate(
        n_units=20000, rng=np.random.default_rng(4)
    )

    def dependence(simulated) -> float:
        frame = simulated.dataset.frame
        return abs(np.corrcoef(frame["treated"], frame["x1"])[0, 1])

    assert dependence(strong) > dependence(weak)


def test_unknown_scenario_id_is_rejected():
    with pytest.raises(KeyError, match="Z"):
        build_scenario("Z")
