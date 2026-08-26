import numpy as np
import pytest
from cp_simulation.components import UnitGenerator
from cp_simulation.panel_components import (
    ConstantDynamicEffect,
    LinearTimePath,
    NormalUnitEffects,
    RampDynamicEffect,
    StaggeredAdoption,
    TrendSelectedAdoption,
    UniformAdoption,
)
from cp_simulation.panel_dgp import PanelDGP
from cp_simulation.panel_scenarios import build_panel_scenario


def build(adoption, effect, slope_sd=0.0, error_rho=0.0, scenario_id="T"):
    return PanelDGP(
        unit_generator=UnitGenerator(covariate_names=("x1", "x2")),
        unit_effects=NormalUnitEffects(level_sd=1.0, slope_sd=slope_sd),
        time_path=LinearTimePath(slope=0.1),
        adoption=adoption,
        effect=effect,
        scenario_id=scenario_id,
        error_rho=error_rho,
    )


def test_generated_frame_is_long_and_balanced():
    dgp = build(UniformAdoption(start=5), ConstantDynamicEffect(tau=1.0))

    panel = dgp.generate(n_units=50, n_periods=8, rng=np.random.default_rng(0))

    assert len(panel.dataset.frame) == 50 * 8
    assert panel.dataset.n_units == 50
    assert panel.dataset.n_periods == 8
    assert panel.dataset.is_balanced
    assert panel.dataset.schema.outcome_type == "continuous"


def test_treatment_is_absorbing_and_matches_the_cohorts():
    dgp = build(StaggeredAdoption(), ConstantDynamicEffect(tau=1.0))

    panel = dgp.generate(n_units=100, n_periods=8, rng=np.random.default_rng(0))

    assert panel.dataset.treatment_is_absorbing
    derived = panel.dataset.cohort_by_unit().to_numpy()
    np.testing.assert_array_equal(derived.astype(int), panel.truth.true_cohorts)


def test_periods_are_strictly_positive():
    # The panel estimators use 0 as the never-treated sentinel, so a period of
    # 0 would be indistinguishable from "never adopted".
    dgp = build(UniformAdoption(start=5), ConstantDynamicEffect(tau=1.0))

    panel = dgp.generate(n_units=20, n_periods=8, rng=np.random.default_rng(0))

    assert panel.dataset.periods.min() >= 1


def test_att_by_cell_is_measured_from_the_potential_outcomes():
    dgp = build(StaggeredAdoption(), RampDynamicEffect(tau=1.0))

    panel = dgp.generate(n_units=300, n_periods=8, rng=np.random.default_rng(0))

    # A ramp gives exposure-1 cells an effect of exactly tau and exposure-2
    # cells exactly 2*tau, so the (g, t) grid is checkable by hand.
    assert panel.truth.true_att_gt[(3, 3)] == pytest.approx(1.0)
    assert panel.truth.true_att_gt[(3, 4)] == pytest.approx(2.0)
    assert panel.truth.true_att_gt[(5, 5)] == pytest.approx(1.0)


def test_overall_att_cannot_equal_the_input_parameter_under_a_ramp():
    # The aggregate is a cohort-size-weighted average over treated cells, so it
    # is structurally incapable of equalling any single input parameter. This is
    # what closes the measured-truth gap that the continuous scale leaves open.
    dgp = build(StaggeredAdoption(), RampDynamicEffect(tau=1.0))

    panel = dgp.generate(n_units=400, n_periods=8, rng=np.random.default_rng(0))

    assert panel.truth.true_att_overall > 2.0


def test_event_time_truth_is_keyed_by_exposure_and_is_non_negative_only():
    dgp = build(StaggeredAdoption(), RampDynamicEffect(tau=1.0))

    panel = dgp.generate(n_units=300, n_periods=8, rng=np.random.default_rng(0))

    assert min(panel.truth.true_att_by_event_time) == 0
    assert panel.truth.true_att_by_event_time[0] == pytest.approx(1.0)
    assert panel.truth.true_att_by_event_time[1] == pytest.approx(2.0)


def test_pre_trend_gap_is_near_zero_without_slope_dispersion():
    dgp = build(StaggeredAdoption(), ConstantDynamicEffect(tau=1.0), slope_sd=0.0)

    panel = dgp.generate(n_units=400, n_periods=8, rng=np.random.default_rng(0))

    assert abs(panel.truth.true_pre_trend_gap) < 0.05


def test_pre_trend_gap_is_large_under_trend_selected_adoption():
    # Parallel trends becomes an observable property of the scenario rather
    # than a claim about it.
    dgp = build(TrendSelectedAdoption(), ConstantDynamicEffect(tau=1.0), slope_sd=0.3)

    panel = dgp.generate(n_units=400, n_periods=8, rng=np.random.default_rng(0))

    assert panel.truth.true_pre_trend_gap > 0.3


def test_ar1_errors_raise_within_unit_serial_correlation():
    plain = build(UniformAdoption(), ConstantDynamicEffect(tau=0.0), error_rho=0.0)
    serial = build(UniformAdoption(), ConstantDynamicEffect(tau=0.0), error_rho=0.8)

    def lag_one_correlation(dgp):
        panel = dgp.generate(n_units=400, n_periods=8, rng=np.random.default_rng(0))
        frame = panel.dataset.frame.sort_values(["unit_id", "period"])
        wide = frame["outcome"].to_numpy().reshape(400, 8)
        centred = wide - wide.mean(axis=1, keepdims=True)
        return float(
            np.mean(
                [np.corrcoef(centred[:, :-1].ravel(), centred[:, 1:].ravel())[0, 1]]
            )
        )

    assert lag_one_correlation(serial) > lag_one_correlation(plain) + 0.2


def test_config_hash_is_stable_and_distinguishes_scenarios():
    a = build(UniformAdoption(), ConstantDynamicEffect(tau=1.0), scenario_id="A")
    b = build(UniformAdoption(), ConstantDynamicEffect(tau=2.0), scenario_id="A")

    assert a.config_hash == build(
        UniformAdoption(), ConstantDynamicEffect(tau=1.0), scenario_id="A"
    ).config_hash
    assert a.config_hash != b.config_hash


def test_generate_rejects_a_panel_too_short_to_difference():
    dgp = build(UniformAdoption(start=5), ConstantDynamicEffect(tau=1.0))

    with pytest.raises(ValueError, match="n_periods"):
        dgp.generate(n_units=20, n_periods=1, rng=np.random.default_rng(0))


def test_generated_panel_schema_carries_the_adoption_provenance():
    staggered = build_panel_scenario("P3").generate(
        n_units=120, n_periods=8, rng=np.random.default_rng(0)
    )
    trend_selected = build_panel_scenario("P5").generate(
        n_units=120, n_periods=8, rng=np.random.default_rng(0)
    )
    assert staggered.dataset.schema.assignment_provenance == "natural_experiment"
    assert trend_selected.dataset.schema.assignment_provenance == "observational"
