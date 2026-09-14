from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from cp_core.manifest import ScenarioReference
from cp_simulation.components import ConstantEffect, HeterogeneousEffect
from cp_simulation.params import ScenarioParams
from cp_simulation.scenarios import (
    SCENARIOS,
    build_scenario,
    resolve_scenario_reference,
)

# A small set that exercises `declared_effect` as a function rather than at
# one point: below 1 (scaling down), the pre-Phase-18 default, and above 1
# (scaling up, where clipping at the probability bounds has the most room to
# bite). Task 8 measures the grid's real values; this set only has to move
# the label.
_EFFECT_SCALES = (0.5, 1.0, 2.0)

# Measured across every scenario x _EFFECT_SCALES combination (30 seeds,
# n_units=40000): the worst |measured - declared| was 0.00088, at scenario F
# (the only scenario with real sampling variance in the mean — a homogeneous
# scenario's cate is constant across units, so its mean has none) and
# effect_scale=2.0 (where clipping at the probability bounds bites hardest).
# 0.003 keeps about 3.4x headroom over that worst case, and about 6x headroom
# over the seed=0 deviation this test itself sees (0.0005).
_DECLARED_EFFECT_TOLERANCE = 0.003


def test_all_scenarios_are_registered():
    assert sorted(SCENARIOS) == ["A", "B", "C", "D", "F", "G", "H"]


def _scaled_ground_truth(scenario_id: str, effect_scale: float) -> float:
    """Measured true_ate for `scenario_id` with its effect scaled by hand.

    `build_scenario` does not yet read `effect_scale` -- Task 6 wires that in
    -- so this reconstructs, independently of `declared_effect`, the DGP that
    wiring will produce: it reads the unscaled tau (or base_tau/increment)
    values off the scenario's own `build` and multiplies them by
    `effect_scale` itself, rather than calling anything `declared_effect`
    computes. That keeps this measurement and the label under test on
    separate code paths, so the assertion below binds today rather than only
    once Task 6 lands.
    """
    dgp = SCENARIOS[scenario_id].build(ScenarioParams())
    effect = dgp.effect
    if isinstance(effect, ConstantEffect):
        scaled_effect = ConstantEffect(tau=effect.tau * effect_scale)
    elif isinstance(effect, HeterogeneousEffect):
        scaled_effect = replace(
            effect,
            base_tau=effect.base_tau * effect_scale,
            increment=effect.increment * effect_scale,
        )
    else:
        raise TypeError(f"unhandled effect type {type(effect)!r} for scenario {scenario_id!r}")

    scaled_dgp = replace(dgp, effect=scaled_effect)
    simulated = scaled_dgp.generate(n_units=40000, rng=np.random.default_rng(0))
    return float(simulated.truth.true_ate)


@pytest.mark.parametrize("effect_scale", _EFFECT_SCALES)
@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_measured_ate_matches_the_declared_label(scenario_id: str, effect_scale: float):
    # A scenario advertising "+5pp" must actually deliver +5pp, or every
    # downstream bias number is wrong -- and that has to hold at every point
    # on the effect_scale axis now that effect size is a grid axis, not only
    # at the one point a scenario used to have.
    params = ScenarioParams(effect_scale=effect_scale)
    declared = SCENARIOS[scenario_id].declared_effect(params)

    measured = _scaled_ground_truth(scenario_id, effect_scale)

    assert measured == pytest.approx(declared, abs=_DECLARED_EFFECT_TOLERANCE)


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

    assert simulated.truth.true_ate == pytest.approx(
        spec.declared_effect(ScenarioParams()), abs=0.002
    )
    assert 0.01 < simulated.truth.true_ate < 0.08


# Spans below (0.0, collapsing the effect to fully homogeneous),
# at (1.0, the pre-Phase-18 default, already covered elsewhere) and above
# (3.0, the perturbed value `tests/simulation/test_axis_declarations.py`
# uses) the default, so this exercises the compensation formula's shape
# rather than one point on it.
_HETEROGENEITY_SCALES = (0.0, 1.0, 3.0)


@pytest.mark.parametrize("heterogeneity_scale", _HETEROGENEITY_SCALES)
def test_scenario_f_declared_effect_matches_measured_across_heterogeneity_scale(
    heterogeneity_scale: float,
):
    # Hazard 3 (spec section 4.4): scaling `increment` by `heterogeneity_scale`
    # without compensating `base_tau` moves the population mean CATE, silently
    # turning `heterogeneity_scale` into a second `effect_scale`.
    # `declared_effect` does not take `heterogeneity_scale` as an input at all
    # -- the compensation is what makes that omission correct -- so if a
    # scenario F implementation ever dropped it, `measured` would move away
    # from `declared` here while `declared` stayed fixed.
    #
    # This is the check `test_measured_ate_matches_the_declared_label` above
    # cannot be, despite touching the same label: it holds `heterogeneity_scale`
    # at its default (1.0) in every one of its cases, and at exactly 1.0 the
    # compensation term `(k - 1) * increment * P(x=1)` is zero regardless of
    # whether the implementation computes it at all. Confirmed by deliberately
    # dropping the compensation (`compensated_base_tau = base_tau`) and
    # re-running: this test fails at heterogeneity_scale=3.0 while
    # `test_measured_ate_matches_the_declared_label` stays green throughout.
    params = ScenarioParams(heterogeneity_scale=heterogeneity_scale)
    declared = SCENARIOS["F"].declared_effect(params)

    simulated = SCENARIOS["F"].build(params).generate(
        n_units=40000, rng=np.random.default_rng(3)
    )

    assert float(simulated.truth.true_ate) == pytest.approx(
        declared, abs=_DECLARED_EFFECT_TOLERANCE
    )


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


def test_scenario_g_is_registered_and_confounded():
    simulated = build_scenario("G").generate(n_units=20000, rng=np.random.default_rng(0))

    assert simulated.truth.true_ate == pytest.approx(0.05, abs=0.002)
    assert simulated.truth.true_confounders == ("x1", "x2")


def test_build_scenario_defaults_to_neither_component():
    from cp_simulation.scenarios import build_scenario

    dgp = build_scenario("D")
    assert dgp.negative_control is None
    assert dgp.misclassification is None


def test_build_scenario_attaches_a_negative_control_on_request():
    from cp_simulation.scenarios import DEFAULT_NEGATIVE_CONTROL, build_scenario

    dgp = build_scenario("D", negative_control=DEFAULT_NEGATIVE_CONTROL)
    simulated = dgp.generate(n_units=300, rng=np.random.default_rng(0))
    assert simulated.dataset.schema.negative_control_outcome == "outcome_nc"


def test_build_scenario_attaches_misclassification_on_request():
    from cp_simulation.components import NonDifferentialMisclassification
    from cp_simulation.scenarios import build_scenario

    dgp = build_scenario(
        "D",
        misclassification=NonDifferentialMisclassification(
            sensitivity=0.8, specificity=0.9
        ),
    )
    simulated = dgp.generate(n_units=300, rng=np.random.default_rng(0))
    assert simulated.truth.true_treatment is not None


def test_both_components_can_be_attached_together():
    from cp_simulation.components import NonDifferentialMisclassification
    from cp_simulation.scenarios import DEFAULT_NEGATIVE_CONTROL, build_scenario

    dgp = build_scenario(
        "D",
        negative_control=DEFAULT_NEGATIVE_CONTROL,
        misclassification=NonDifferentialMisclassification(
            sensitivity=0.8, specificity=0.9
        ),
    )
    simulated = dgp.generate(n_units=300, rng=np.random.default_rng(0))
    assert simulated.dataset.schema.negative_control_outcome == "outcome_nc"
    assert simulated.truth.true_treatment is not None


def test_confounding_strength_still_reaches_the_builder():
    # The opt-in parameters must not displace the existing positional one.
    from cp_simulation.scenarios import build_scenario

    weak = build_scenario("D", 0.1)
    strong = build_scenario("D", 4.0)
    assert weak.config_hash != strong.config_hash


def test_the_default_negative_control_matches_the_scenario_covariates():
    from cp_simulation.scenarios import DEFAULT_NEGATIVE_CONTROL, build_scenario

    dgp = build_scenario("A")
    assert len(DEFAULT_NEGATIVE_CONTROL.coefficients) == len(
        dgp.unit_generator.covariate_names
    )


def test_scenario_h_is_registered():
    from cp_simulation.scenarios import SCENARIOS

    assert "H" in SCENARIOS
    assert SCENARIOS["H"].declared_effect(ScenarioParams()) == 0.05


def test_scenario_h_produces_missing_outcomes():
    import numpy as np
    from cp_simulation.scenarios import build_scenario

    simulated = build_scenario("H").generate(
        n_units=2000, rng=np.random.default_rng(0)
    )
    outcome = simulated.dataset.frame[simulated.dataset.schema.outcome]
    assert outcome.isna().any()
    assert simulated.truth.complete_outcome is not None


def test_scenario_h_inherits_scenario_ds_confounding():
    # H is D plus missingness, per the parent spec, so its assignment
    # mechanism must be D's -- not randomised.
    from cp_simulation.scenarios import build_scenario

    assert (
        build_scenario("H").assignment.provenance()
        == build_scenario("D").assignment.provenance()
    )


def test_build_scenario_can_attach_missingness_to_any_scenario():
    import numpy as np
    from cp_simulation.components import MissingnessMechanism
    from cp_simulation.scenarios import build_scenario

    dgp = build_scenario(
        "A", missingness=MissingnessMechanism(mechanism="MNAR", rate=0.2)
    )
    simulated = dgp.generate(n_units=500, rng=np.random.default_rng(0))
    assert simulated.dataset.frame["outcome"].isna().any()


def test_resolve_scenario_reference_matches_a_direct_call():
    direct = build_scenario("D").generate(n_units=500, rng=np.random.default_rng(11)).dataset
    reference = ScenarioReference(scenario="D", seed=11, params={"n_units": 500})

    resolved = resolve_scenario_reference(reference, direct.schema)

    pd.testing.assert_frame_equal(resolved.frame, direct.frame)
    assert resolved.schema == direct.schema


def test_resolve_scenario_reference_honours_confounding_strength():
    # Not a redundant second happy path: `confounding_strength` is the one
    # optional param, and a resolver that dropped it would still match a
    # direct call at the default 1.0 and so pass the test above.
    direct = build_scenario("D", 2.5).generate(n_units=400, rng=np.random.default_rng(4)).dataset
    reference = ScenarioReference(
        scenario="D", seed=4, params={"n_units": 400, "confounding_strength": 2.5}
    )

    resolved = resolve_scenario_reference(reference, direct.schema)

    pd.testing.assert_frame_equal(resolved.frame, direct.frame)
    at_default = build_scenario("D").generate(n_units=400, rng=np.random.default_rng(4)).dataset
    assert not direct.frame.equals(at_default.frame)


def test_resolve_scenario_reference_refuses_an_unsupported_params_key():
    reference = ScenarioReference(
        scenario="D", seed=0, params={"n_units": 200, "outcome_proxy": "nope"}
    )
    schema = build_scenario("D").generate(n_units=200, rng=np.random.default_rng(0)).dataset.schema

    with pytest.raises(ValueError, match="unsupported keys"):
        resolve_scenario_reference(reference, schema)


def test_resolve_scenario_reference_requires_n_units():
    reference = ScenarioReference(scenario="D", seed=0, params={})
    schema = build_scenario("D").generate(n_units=200, rng=np.random.default_rng(0)).dataset.schema

    with pytest.raises(KeyError, match="n_units"):
        resolve_scenario_reference(reference, schema)


def test_resolve_scenario_reference_refuses_on_schema_mismatch():
    reference = ScenarioReference(scenario="D", seed=0, params={"n_units": 200})
    # A and D differ only in `assignment_provenance` -- 'randomised' against
    # 'observational'. That is the whole of what this guard can see: A, B and C
    # share one schema and D, G and H share another, so a reference naming the
    # wrong scenario *within* a group passes it. The check catches a registry
    # that changed under a stored manifest, not a mistyped scenario id.
    wrong_schema = build_scenario("A").generate(
        n_units=200, rng=np.random.default_rng(0)
    ).dataset.schema

    with pytest.raises(ValueError, match="does not match"):
        resolve_scenario_reference(reference, wrong_schema)
