"""Every panel estimator against every scenario it can be honestly scored on.

Not literally every estimator against every scenario: `DifferenceInDifferences`
is n/a on staggered adoption (P3-P5), and `EventStudy` is scored on P1/P3 only
-- see the note above `test_event_study_recovers_truth_where_its_point_is_the_
same_estimand` for why P4/P5 would compare two different quantities rather
than test recovery.

Tolerances are set from measurements taken at exactly this protocol - 400
units, 8 periods, 60 seeds, matching PANEL_N_UNITS/PANEL_N_PERIODS/PANEL_SEEDS
- run against the real merged estimators (not the earlier prototype
figures, which predate this code and drifted in several places already
corrected earlier in this phase). Each tolerance states its measured mean,
its Monte Carlo standard error (MCSE), and the headroom arithmetic in a
comment, so a later reader can check the arithmetic rather than trust it.
"""

import dataclasses

import numpy as np
import pytest
from cp_causal.estimators.callaway_santanna import CallawaySantAnna
from cp_causal.estimators.did import DifferenceInDifferences, TwoWayFixedEffects
from cp_causal.estimators.event_study import EventStudy
from cp_causal.goodman_bacon import goodman_bacon_decomposition
from cp_causal.registry import all_estimators
from cp_causal.router import eligible_estimators
from cp_core.values import EstimatorConfig
from cp_simulation.panel_scenarios import (
    PANEL_N_PERIODS,
    PANEL_N_UNITS,
    PANEL_SCENARIOS,
    PANEL_SEEDS,
    build_panel_scenario,
)

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)
STAGGERED_CAPABLE = (TwoWayFixedEffects(), CallawaySantAnna())


def generate(scenario_id, seed=0, n_units=PANEL_N_UNITS):
    return build_panel_scenario(scenario_id).generate(
        n_units=n_units, n_periods=PANEL_N_PERIODS, rng=np.random.default_rng(seed)
    )


def mean_bias(estimator, scenario_id, seeds=PANEL_SEEDS):
    errors = []
    for seed in range(seeds):
        panel = generate(scenario_id, seed=seed)
        estimate = estimator.fit(panel.dataset, CONFIG)
        errors.append(estimate.point - panel.truth.true_att_overall)
    return float(np.mean(errors))


@pytest.mark.parametrize("scenario_id", ["P1", "P2"])
def test_every_estimator_recovers_a_clean_two_by_two(scenario_id):
    # Measured at 400 units / 8 periods / 60 seeds against the real merged
    # estimators (not the earlier prototype figures): 2x2 and TWFE both
    # -0.0143 (MCSE 0.0088) on P1 and P2 (identical, as expected -- the
    # estimation error depends only on the Y(0) noise, and P2's tau=0 cannot
    # move it).
    # Headroom under 0.06: (0.06 - 0.0143) / 0.0088 = 5.19 MCSE.
    for estimator in (DifferenceInDifferences(), TwoWayFixedEffects()):
        assert abs(mean_bias(estimator, scenario_id)) < 0.06, estimator.name


@pytest.mark.parametrize("scenario_id", ["P1", "P3"])
def test_callaway_santanna_recovers_truth_where_it_should(scenario_id):
    # Measured: P1 -0.0064 (MCSE 0.0155), P3 +0.0008 (MCSE 0.0131).
    # Headroom under 0.09: (0.09 - 0.0064) / 0.0155 = 5.4 MCSE on P1,
    # (0.09 - 0.0008) / 0.0131 = 6.8 MCSE on P3.
    assert abs(mean_bias(CallawaySantAnna(), scenario_id)) < 0.09


@pytest.mark.parametrize("scenario_id", ["P1", "P3"])
def test_event_study_recovers_truth_where_its_point_is_the_same_estimand(scenario_id):
    # EventStudy.point is the unweighted mean over post-treatment event-time
    # coefficients (see event_study.py's own docstring). On P1 and P3, every
    # treated cohort's effect is constant across exposure, so that unweighted
    # average and true_att_overall's cell-count-weighted average are the same
    # quantity. On P4 they are NOT: a ramping effect makes later exposure
    # cells larger, so the unweighted mean (which treats every exposure
    # horizon equally) and the cell-weighted ATT (which weights by how many
    # treated cells actually exist at each horizon) diverge structurally --
    # measured at roughly +0.64 off true_att_overall on P4, which is not a
    # broken estimator, it is two different quantities being compared. P4/P5
    # are deliberately excluded from this test for that reason, not omitted
    # by oversight.
    #
    # Measured at 400 units / 8 periods / 60 seeds: P1 -0.0064 (MCSE 0.0155),
    # P3 -0.0047 (MCSE 0.0139).
    # Headroom under 0.09: (0.09 - 0.0064) / 0.0155 = 5.4 MCSE on P1,
    # (0.09 - 0.0047) / 0.0139 = 6.1 MCSE on P3.
    assert abs(mean_bias(EventStudy(), scenario_id)) < 0.09


def test_twfe_and_callaway_santanna_agree_under_homogeneous_effects():
    # P3's whole purpose. Two methods with different machinery landing on the
    # same answer is an independent cross-check on the cohort-period alignment
    # that no single-estimator test can provide - and it is the control
    # condition that makes P4's disagreement interpretable.
    # Measured mean paired difference (twfe - cs) +0.0039 (MCSE 0.0095) over
    # the same 60 seeds; headroom under 0.05 is (0.05 - 0.0039) / 0.0095 =
    # 4.85 MCSE.
    differences = []
    for seed in range(PANEL_SEEDS):
        dataset = generate("P3", seed=seed).dataset
        twfe = TwoWayFixedEffects().fit(dataset, CONFIG).point
        cs = CallawaySantAnna().fit(dataset, CONFIG).point
        differences.append(twfe - cs)

    assert abs(float(np.mean(differences))) < 0.05


def test_p4_breaks_twfe_and_not_callaway_santanna():
    # THE headline result of this phase, and deliberately two-sided. Asserting
    # only CS's accuracy would still pass if TWFE were secretly fine - which
    # would mean P4 had stopped being a heterogeneous-effects scenario and every
    # conclusion drawn from it was void.
    # Measured: TWFE -0.9946 (MCSE 0.0112, 88.8 MCSE from zero),
    #           CS    +0.0008 (MCSE 0.0131).
    # Headroom: TWFE floor 0.25 sits (0.9946 - 0.25) / 0.0112 = 66.5 MCSE below
    #           the measurement; CS ceiling 0.09 sits (0.09 - 0.0008) / 0.0131
    #           = 6.8 MCSE above it.
    twfe_bias = mean_bias(TwoWayFixedEffects(), "P4")
    cs_bias = mean_bias(CallawaySantAnna(), "P4")

    assert abs(twfe_bias) > 0.25, f"TWFE bias {twfe_bias:+.4f} - P4 is no longer heterogeneous"
    assert abs(cs_bias) < 0.09, f"CS bias {cs_bias:+.4f}"


def test_p4_rests_heavily_on_already_treated_comparisons():
    # The mechanism behind the bias above, measured rather than asserted.
    # Measured already_treated_share on P4 over 60 seeds: mean 0.2624,
    # min 0.2363 (seed 0's own draw: 0.2567). Threshold set to roughly half
    # the measured minimum, leaving every one
    # of the 60 sampled seeds with more than double the required margin.
    decomposition = goodman_bacon_decomposition(generate("P4").dataset)

    assert decomposition.already_treated_share > 0.1


def test_p5_defeats_every_estimator_including_callaway_santanna():
    # What stops CS being presented as magic. It assumes parallel trends too,
    # and P5 violates them.
    # Measured: TWFE +1.2704 (MCSE 0.0111), CS +1.6776 (MCSE 0.0138).
    # Headroom above the 0.5 floor: (1.2704 - 0.5) / 0.0111 = 69.4 MCSE for
    # TWFE, (1.6776 - 0.5) / 0.0138 = 85.3 MCSE for CS.
    for estimator in STAGGERED_CAPABLE:
        assert mean_bias(estimator, "P5") > 0.5, estimator.name


def test_p5_is_reported_as_a_parallel_trends_failure():
    # A biased estimate that says nothing about why is a worse outcome than a
    # biased estimate that flags its own assumption failure.
    for estimator in (EventStudy(), CallawaySantAnna()):
        result = estimator.fit(generate("P5").dataset, CONFIG)
        statuses = {a.name: a.status for a in result.assumptions}
        assert statuses["parallel_trends"] == "FAIL", estimator.name


def test_p1_is_reported_as_passing_parallel_trends():
    for estimator in (EventStudy(), CallawaySantAnna()):
        result = estimator.fit(generate("P1").dataset, CONFIG)
        statuses = {a.name: a.status for a in result.assumptions}
        assert statuses["parallel_trends"] == "PASS", estimator.name


def test_null_scenario_produces_no_systematic_significance():
    # P2 has exactly zero effect. At the 5% level roughly 1 in 20 fits should
    # read as significant by chance.
    #
    # This bound is ONE-SIDED and therefore cannot detect over-wide intervals:
    # an estimator reporting intervals twice as wide as they should be would
    # pass this comfortably while being badly miscalibrated. Interval coverage
    # is a benchmark-phase measurement, and this test is not evidence of good
    # calibration in either direction. The same limitation was recorded against
    # the cross-sectional harness in Phase 2.
    significant = 0
    total = 0
    for seed in range(20):
        dataset = generate("P2", seed=seed).dataset
        for estimator in (DifferenceInDifferences(), *STAGGERED_CAPABLE):
            significant += int(estimator.fit(dataset, CONFIG).is_significant)
            total += 1

    assert significant / total < 0.25


def test_clustering_matters_and_the_estimators_do_it():
    # The clustering trap, pinned. This MUST run at error_rho = 0.8 - at rho = 0
    # the clustered/unclustered ratio is close to 1 and the test would pass
    # whether or not clustering were wired up at all.
    # Measured clustered_se_ratio on P1 at error_rho=0.8, seed 0: 1.9094.
    # Threshold 1.5 leaves comfortable margin.
    serial = dataclasses.replace(build_panel_scenario("P1"), error_rho=0.8)
    panel = serial.generate(PANEL_N_UNITS, PANEL_N_PERIODS, np.random.default_rng(0))

    result = TwoWayFixedEffects().fit(panel.dataset, CONFIG)

    assert result.diagnostics["clustered_se_ratio"].value > 1.5


def test_the_router_offers_the_right_panel_estimators():
    uniform = set(eligible_estimators(generate("P1").dataset))
    staggered = set(eligible_estimators(generate("P3").dataset))

    assert {"did_2x2", "twfe", "event_study", "callaway_santanna", "hierarchical_did"} <= uniform
    # The 2x2 is excluded from staggered data with an explicit reason.
    assert "did_2x2" not in staggered
    assert {"twfe", "event_study", "callaway_santanna", "hierarchical_did"} <= staggered
    # Cross-sectional estimators need covariates and a cross-section; none of
    # them should be offered on a panel.
    assert "aipw" not in uniform


def test_every_panel_scenario_is_covered_by_this_harness():
    # If a scenario is added without a recovery expectation, say so loudly
    # rather than letting it drift untested. P6's expectation lives in
    # tests/simulation/test_panel_binary.py, which measures its link bias.
    # P7's lives in tests/causal/test_hierarchical_did_recovery.py, which is
    # the only estimator with enough cohorts to need it.
    assert set(PANEL_SCENARIOS) == {"P1", "P2", "P3", "P4", "P5", "P6", "P7"}


def test_importing_cp_causal_registers_every_panel_estimator():
    import cp_causal  # noqa: F401

    assert {
        "did_2x2",
        "twfe",
        "event_study",
        "callaway_santanna",
        "hierarchical_did",
    } <= set(all_estimators())
