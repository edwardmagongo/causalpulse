"""Donor-pool scenario registry: declared-vs-measured truth, the same
anti-drift discipline Phase 3's P1-P6 registry applies."""

from __future__ import annotations

import numpy as np
import pytest
from cp_simulation.donor_scenarios import (
    DONOR_N_PERIODS,
    DONOR_N_UNITS,
    DONOR_SCENARIOS,
    DONOR_START,
    build_donor_scenario,
)


def test_every_scenario_id_is_present():
    assert set(DONOR_SCENARIOS) == {"S0", "S1", "S2", "S3", "S4", "S5"}


@pytest.mark.parametrize("scenario_id", ["S0", "S1", "S2", "S3", "S4"])
def test_measured_truth_matches_the_declared_label(scenario_id):
    spec = DONOR_SCENARIOS[scenario_id]
    deviations = []
    for seed in range(10):
        panel = build_donor_scenario(scenario_id).generate(
            spec.n_units, spec.n_periods, np.random.default_rng(seed)
        )
        deviations.append(abs(panel.truth.true_att_by_cohort[DONOR_START] - spec.declared_effect))
    # The continuous, unclipped scale makes per-seed truth numerically equal
    # to the input tau by construction (a stated Phase 3
    # limitation, inherited here) -- this asserts consistency, not that truth
    # is independently measured. S0-S4 all adopt at DONOR_START with a
    # constant effect, so `true_att_by_cohort[DONOR_START]` is well-defined
    # for every one of them.
    assert max(deviations) < 1e-9


def test_s5_has_only_two_pre_periods():
    spec = DONOR_SCENARIOS["S5"]
    panel = build_donor_scenario("S5").generate(
        spec.n_units, spec.n_periods, np.random.default_rng(0)
    )
    adoption = panel.dataset.cohort_by_unit()
    earliest = int(adoption[adoption > 0].min())
    pre_periods = int(np.count_nonzero(panel.dataset.periods < earliest))
    assert pre_periods == 2


def test_s0_has_zero_true_effect():
    panel = build_donor_scenario("S0").generate(
        DONOR_N_UNITS, DONOR_N_PERIODS, np.random.default_rng(0)
    )
    assert panel.truth.true_att_by_cohort[DONOR_START] == 0.0


def test_every_scenario_has_exactly_one_treated_unit():
    for scenario_id, spec in DONOR_SCENARIOS.items():
        panel = build_donor_scenario(scenario_id).generate(
            spec.n_units, spec.n_periods, np.random.default_rng(0)
        )
        assert panel.dataset.n_treated_units == 1, scenario_id


def test_s4_shocks_exactly_one_donor_and_leaves_ground_truth_untouched():
    clean = build_donor_scenario("S1").generate(
        DONOR_N_UNITS, DONOR_N_PERIODS, np.random.default_rng(0)
    )
    contaminated = build_donor_scenario("S4").generate(
        DONOR_N_UNITS, DONOR_N_PERIODS, np.random.default_rng(0)
    )
    # Same seed, same underlying draw -- the only difference should be the
    # shock added to one donor's post-shock outcome. Ground truth describes
    # the treated unit alone and must be identical.
    assert contaminated.truth.true_att_by_cohort == clean.truth.true_att_by_cohort

    clean_frame = clean.dataset.frame.set_index(["unit_id", "period"])["outcome"]
    contaminated_frame = contaminated.dataset.frame.set_index(["unit_id", "period"])["outcome"]
    diff = (contaminated_frame - clean_frame).abs()
    shocked_rows = diff[diff > 1e-9]
    # Exactly one donor, exactly the post-shock periods (18..24 = 7 periods).
    assert shocked_rows.index.get_level_values("unit_id").nunique() == 1
    assert len(shocked_rows) == 7
