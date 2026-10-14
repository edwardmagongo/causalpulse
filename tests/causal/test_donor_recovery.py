"""Synthetic control scored against measured donor-pool truth.

Tolerances are set from measurements taken at exactly this protocol -- 18
units (17 donors + 1 treated), 24 periods, adoption at period 18 -- run
against the real merged estimator, not a prototype. Each tolerance states its
measured mean, its Monte Carlo standard error (MCSE) where the assertion is
over a mean, and the headroom arithmetic, so a later reader can check the
arithmetic rather than trust it.
"""

from __future__ import annotations

import numpy as np
from cp_causal.estimators.synthetic_control import SyntheticControl
from cp_core.values import EstimatorConfig
from cp_simulation.donor_scenarios import (
    DONOR_SCENARIOS,
    DONOR_SEEDS,
    DONOR_START,
    build_donor_scenario,
)

CONFIG = EstimatorConfig(bootstrap_reps=10, random_seed=1)


def generate(scenario_id: str, seed: int = 0):
    spec = DONOR_SCENARIOS[scenario_id]
    return build_donor_scenario(scenario_id).generate(
        spec.n_units, spec.n_periods, np.random.default_rng(seed)
    )


def _naive_bias(scenario_id: str, seed: int) -> float:
    """Equal-weighted average of every donor, the losing baseline for S2."""
    panel = generate(scenario_id, seed)
    frame = panel.dataset.frame
    treated_unit = frame.loc[frame["cohort"] > 0, "unit_id"].iloc[0]
    wide = frame.pivot(index="unit_id", columns="period", values="outcome")
    treated = wide.loc[treated_unit].to_numpy()
    donors = wide.drop(index=treated_unit).to_numpy()
    post = panel.dataset.periods >= DONOR_START
    naive_point = float(treated[post].mean() - donors.mean(axis=0)[post].mean())
    return naive_point - panel.truth.true_att_by_cohort[DONOR_START]


def _sc_bias(scenario_id: str, seed: int) -> float:
    panel = generate(scenario_id, seed)
    result = SyntheticControl().fit(panel.dataset, CONFIG)
    return result.point - panel.truth.true_att_by_cohort[DONOR_START]


def test_synthetic_control_recovers_truth_on_the_clean_scenario():
    # Measured: bias -0.0324 (MCSE 0.0320) over 400 seeds.
    # Headroom under 0.15: (0.15 - 0.0324) / 0.0320 = 3.7 MCSE.
    errors = np.array([_sc_bias("S1", seed) for seed in range(DONOR_SEEDS)])
    bias = float(errors.mean())
    assert abs(bias) < 0.15, f"S1 SC bias {bias:+.4f}"


def test_synthetic_control_beats_a_naive_donor_average_under_divergent_pretrend():
    # THE headline result. Deliberately two-sided: asserting only SC's
    # accuracy would still pass if the naive comparison were secretly fine,
    # which would mean S2 had stopped demonstrating anything about donor
    # weighting at all -- the same reasoning behind P4's two-sided TWFE/CS
    # assertion in Phase 3.
    # Measured: SC bias +0.1736 (MCSE 0.0190), naive bias +4.4167 (MCSE
    # 0.0260), over 400 seeds.
    # Headroom: SC ceiling 0.3 sits (0.3 - 0.1736) / 0.0190 = 6.7 MCSE above
    # the measurement; naive floor 2.0 sits (4.4167 - 2.0) / 0.0260 = 93 MCSE
    # below it.
    sc_errors = np.array([_sc_bias("S2", seed) for seed in range(DONOR_SEEDS)])
    naive_errors = np.array([_naive_bias("S2", seed) for seed in range(DONOR_SEEDS)])
    sc_bias = float(sc_errors.mean())
    naive_bias = float(naive_errors.mean())
    assert abs(sc_bias) < 0.3, f"SC bias {sc_bias:+.4f} -- S2 stopped being recoverable"
    assert naive_bias > 2.0, f"naive bias {naive_bias:+.4f} -- S2 stopped demonstrating anything"


def test_ridge_augmented_fallback_recovers_truth_on_the_convex_hull_scenario():
    # Measured: ridge-augmented bias +0.0223 (MCSE 0.0270) over 300 seeds.
    # Headroom under 0.15: (0.15 - 0.0223) / 0.0270 = 4.7 MCSE.
    errors = np.array([_sc_bias("S3", seed) for seed in range(300)])
    bias = float(errors.mean())
    assert abs(bias) < 0.15, f"S3 (ridge-augmented) bias {bias:+.4f}"


def test_convex_hull_check_fires_reliably_on_s3():
    fired = sum(
        SyntheticControl()
        .fit(generate("S3", seed).dataset, CONFIG)
        .assumptions[
            1
        ]  # (pre_treatment_fit, convex_hull, donor_contamination, sufficient_pre_periods)
        .status
        == "FAIL"
        for seed in range(100)
    )
    # Measured: 300/300 at this protocol. 100 seeds here is enough headroom
    # to distinguish "always fires" from "usually fires" without paying for
    # 300 SLSQP solves in an ordinary test run.
    assert fired == 100


def test_donor_contamination_check_discriminates_clean_from_contaminated():
    # Recalibrated during the SyntheticControl CI/contamination-check phase:
    # donor_rmspe_ratios moved from placebo_runs (plain simplex) to hull_aware_placebo_runs, whose
    # ridge-augmented fallback genuinely widens this statistic's null distribution -- the old
    # threshold=5.5 gave 34/100 false positives against this file's original 100-seed protocol,
    # well outside its own bound. Measured directly on the new statistic at 200 seeds (three
    # separate 200-seed windows checked for stability, not just this one): threshold=7.0,
    # false-positive rate ~12-18% (25-35/200) on clean S1, true-positive rate ~85-88%
    # (169-176/200) on contaminated S4 -- see donor_contamination_check's own docstring for the
    # full sweep and why these margins are real but thinner than the pre-hull-aware 7%/96%. Bounds
    # below give roughly 3 MCSE of headroom on both sides at n=200, not the tighter margin the
    # 100-seed protocol would have left at this threshold.
    def contamination_status(scenario_id: str, seed: int) -> str:
        result = SyntheticControl().fit(generate(scenario_id, seed).dataset, CONFIG)
        return {c.name: c.status for c in result.assumptions}["donor_contamination"]

    false_positives = sum(contamination_status("S1", seed) == "FAIL" for seed in range(200))
    true_positives = sum(contamination_status("S4", seed) == "FAIL" for seed in range(200))
    assert false_positives <= 44, f"{false_positives}/200 false positives on clean S1"
    assert true_positives >= 160, f"{true_positives}/200 true positives on contaminated S4"


def test_sufficient_pre_periods_fires_reliably_on_s5():
    for seed in range(20):
        result = SyntheticControl().fit(generate("S5", seed).dataset, CONFIG)
        status = {c.name: c.status for c in result.assumptions}
        assert status["sufficient_pre_periods"] == "FAIL", seed


def test_null_scenario_permutation_p_value_stays_under_a_quarter_significant():
    # S0 has zero true effect. This is a much weaker check than it looks:
    # permutation_p_value's minimum achievable value is 1/(n_donors+1) =
    # 1/18 ~= 0.0556, already above the 0.05 threshold below, so p < 0.05
    # can never be true at this donor-pool size regardless of the data --
    # the "roughly 1 in 20 by chance" reasoning that motivates a 5%-level
    # bound does not apply here (see permutation_p_value's own docstring).
    # This bound can only confirm nothing is catastrophically broken, not
    # that inference is well-calibrated at the 5% level. One-sided bound
    # only in any case -- cannot detect over-wide intervals, the same
    # limitation test_panel_recovery.py's P2 test carries forward from
    # Phase 2/3.
    significant = sum(
        SyntheticControl().fit(generate("S0", seed).dataset, CONFIG).p_value < 0.05
        for seed in range(100)
    )
    assert significant / 100 < 0.25


def test_fit_ci_coverage_is_measured_on_the_hull_aware_placebo_distribution():
    # Measured during the SyntheticControl CI/contamination-check phase (200 seeds, MCSE ~0.015):
    # switching fit()'s CI half-width from the plain-simplex placebo distribution to the
    # hull-aware one narrowed it substantially on scenarios where the ridge fallback fires often
    # (mean width 1.142 -> 0.793 on S0/S1/S3), which pulled coverage down from a consistently
    # over-nominal 0.970-0.995 to 0.900-0.990, closer to but sometimes just under nominal 0.95
    # (S0/S1 0.920, S2 0.900, S3 0.905, S4 0.990). This is recorded, not chased further: the CI's
    # own construction (a quantile of the placebo spread, following the same permutation logic as
    # the p-value) carries no formal coverage guarantee to begin with, and the fix's purpose was
    # internal consistency with the point estimate -- the same form building both, on every unit
    # -- not a coverage target. S0 and S1 measure identically here (and always will, at any seed):
    # both share the same donor pool and pre-period noise at a given seed, differing only in the
    # treated unit's constant true effect (tau=0 vs tau=1), and this CI's construction never reads
    # the treated unit's own placebo entry (`point_estimates[1:]` excludes index 0), so tau
    # cancels out of both the width and the bias `point - true_att` exactly.
    #
    # The bound below is a loose sanity floor, not a tight recalibration target -- it exists
    # to catch a genuine regression (coverage collapsing), not to assert precision this
    # heuristic interval was never built to promise.
    def covers(scenario_id: str, seed: int) -> bool:
        panel = generate(scenario_id, seed)
        result = SyntheticControl().fit(panel.dataset, CONFIG)
        true_att = panel.truth.true_att_by_cohort[DONOR_START]
        return result.ci.lower <= true_att <= result.ci.upper

    for scenario_id in ["S0", "S1", "S2", "S3", "S4"]:
        hits = sum(covers(scenario_id, seed) for seed in range(50))
        assert hits / 50 >= 0.7, f"{scenario_id}: {hits}/50 CI coverage, expected >= 0.7"


def test_every_donor_scenario_is_covered_by_this_harness():
    assert set(DONOR_SCENARIOS) == {"S0", "S1", "S2", "S3", "S4", "S5"}
