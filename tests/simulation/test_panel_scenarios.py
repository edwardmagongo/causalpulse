import numpy as np
import pytest
from cp_simulation.panel_scenarios import (
    HIER_N_PERIODS,
    HIER_N_UNITS,
    PANEL_N_PERIODS,
    PANEL_N_UNITS,
    PANEL_PROTOCOLS,
    PANEL_SCENARIOS,
    build_panel_scenario,
)


def generate(scenario_id, seed=0, n_units=PANEL_N_UNITS):
    return build_panel_scenario(scenario_id).generate(
        n_units=n_units,
        n_periods=PANEL_N_PERIODS,
        rng=np.random.default_rng(seed),
    )


def test_the_registry_holds_exactly_the_designed_scenarios():
    assert sorted(PANEL_SCENARIOS) == ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]


def test_panel_scenarios_do_not_collide_with_cross_sectional_ones():
    from cp_simulation.scenarios import SCENARIOS

    assert not set(PANEL_SCENARIOS) & set(SCENARIOS)


def test_unknown_scenario_is_refused_by_name():
    with pytest.raises(KeyError, match="P9"):
        build_panel_scenario("P9")


@pytest.mark.parametrize("scenario_id", ["P1", "P2", "P3", "P4", "P5", "P6"])
def test_measured_truth_matches_the_declared_label(scenario_id):
    # The declared_effect is documentation. This is the test that stops it
    # drifting away from what the scenario actually generates. P6 included
    # here even though it is the binary scenario: its declared_effect is
    # measured truth too, just on the probability scale, and it is exactly
    # where the declared/measured
    # divergence is most load-bearing -- a drift here would be the easiest
    # place for the label to silently stop matching the DGP.
    #
    # Same abs=0.01 tolerance as P1-P5, checked at the same single seed=0
    # convention this test already uses for them. Measured at seed=0: P6's
    # true_att_overall is 0.10635 against a declared 0.1061 (dev 0.00034);
    # across 10 seeds the largest deviation from the declared label was
    # 0.00098, comfortably inside 0.01 with no sign this is a lucky draw.
    panel = generate(scenario_id)
    spec = PANEL_SCENARIOS[scenario_id]

    assert panel.truth.true_att_overall == pytest.approx(
        spec.declared_effect, abs=0.01
    ), scenario_id


def test_p1_is_a_two_by_two_design():
    panel = generate("P1")

    adopting = {int(c) for c in panel.truth.true_cohorts if c > 0}
    assert adopting == {5}


def test_p2_has_exactly_zero_effect():
    panel = generate("P2")

    assert panel.truth.true_att_overall == 0.0
    np.testing.assert_array_equal(
        panel.truth.true_effect_matrix, np.zeros_like(panel.truth.true_effect_matrix)
    )


def test_p3_is_staggered_with_a_homogeneous_effect():
    panel = generate("P3")

    adopting = {int(c) for c in panel.truth.true_cohorts if c > 0}
    assert adopting == {3, 5, 7}
    assert len(set(np.round(list(panel.truth.true_att_by_cohort.values()), 6))) == 1


def test_p4_effects_differ_across_cohorts():
    # This is what breaks two-way fixed effects. If cohort effects ever become
    # equal, P4 has silently turned into P3 and every conclusion drawn from it
    # is void.
    panel = generate("P4")

    by_cohort = panel.truth.true_att_by_cohort
    assert by_cohort[3] > by_cohort[5] > by_cohort[7]


def test_p1_and_p3_have_parallel_trends_by_construction():
    for scenario_id in ("P1", "P3"):
        assert abs(generate(scenario_id).truth.true_pre_trend_gap) < 0.05, scenario_id


def test_p5_violates_parallel_trends_measurably():
    panel = generate("P5")

    # Measured, not asserted: the gap is computed from the untreated potential
    # outcomes the estimators never see.
    assert panel.truth.true_pre_trend_gap > 0.3


def test_p5_is_the_only_continuous_scenario_that_selects_on_trend():
    for scenario_id in ("P1", "P2", "P3", "P4"):
        spec = PANEL_SCENARIOS[scenario_id]
        assert not spec.build().adoption.violates_parallel_trends(), scenario_id
    assert PANEL_SCENARIOS["P5"].build().adoption.violates_parallel_trends()


def test_p7_has_many_cohorts_each_with_pre_and_post_periods():
    from cp_simulation.panel_scenarios import (
        HIER_N_PERIODS,
        HIER_N_UNITS,
        build_panel_scenario,
    )

    dgp = build_panel_scenario("P7")
    panel = dgp.generate(HIER_N_UNITS, HIER_N_PERIODS, np.random.default_rng(0))

    cohorts = sorted(set(panel.truth.true_cohorts.tolist()) - {0})
    assert len(cohorts) >= 6, f"P7 needs many cohorts to identify sigma_tau, got {cohorts}"
    for g in cohorts:
        assert 2 <= g <= HIER_N_PERIODS


def test_p7_is_the_only_fixed_protocol_and_reads_the_hier_constants():
    assert dict(PANEL_PROTOCOLS) == {"P7": (HIER_N_UNITS, HIER_N_PERIODS)}
    assert (HIER_N_UNITS, HIER_N_PERIODS) == (600, 20)
    assert set(PANEL_PROTOCOLS) <= set(PANEL_SCENARIOS)


def test_the_protocol_registry_is_read_only():
    with pytest.raises(TypeError):
        PANEL_PROTOCOLS["P1"] = (150, 7)  # type: ignore[index]
