"""Falsification: does the finding survive tests it should fail if it were spurious?

This file tests routing and shape only -- which technique fires for which
estimator family, and that every absence carries a reason. Where the p-value
actually lands is an empirical claim measured in test_falsification_measured.py
under a protocol that can support it. Asserting a p-value here, on 20
permutations of one seed, would be measuring the rate from a single instance.
"""

import numpy as np
import pytest
from cp_causal.registry import all_estimators
from cp_causal.sensitivity import falsify, randomization_inference
from cp_core.values import EstimatorConfig
from cp_simulation.donor_scenarios import DONOR_SCENARIOS, build_donor_scenario
from cp_simulation.panel_scenarios import build_panel_scenario
from cp_simulation.scenarios import build_scenario

FAST = EstimatorConfig(bootstrap_reps=25, random_seed=1)


def cross_section(n_units=400):
    return build_scenario("A").generate(
        n_units=n_units, rng=np.random.default_rng(0)
    ).dataset


def uniform_panel(n_units=120):
    # 120 rather than a smaller fixture on purpose. A panel placebo group is
    # half the never-treated pool, so below about 100 units it clears
    # `Requirements` but is still too small for the estimator's own bootstrap.
    # The repo's own panel protocol runs at 400 units; 120 is the cheapest
    # size at which this technique genuinely runs. `test_a_panel_too_small_...`
    # below pins the behaviour on the other side of that boundary.
    return (
        build_panel_scenario("P1")
        .generate(n_units=n_units, n_periods=6, rng=np.random.default_rng(0))
        .dataset
    )


def donor_panel(scenario_id="S1"):
    spec = DONOR_SCENARIOS[scenario_id]
    return (
        build_donor_scenario(scenario_id)
        .generate(spec.n_units, spec.n_periods, np.random.default_rng(0))
        .dataset
    )


def test_a_cross_sectional_estimator_gets_randomization_inference_only():
    result = falsify(cross_section(), "lin_regression", FAST, n_permutations=20)
    assert result.randomization_inference is not None
    assert result.randomization_inference_reason is None
    for absent, reason in (
        (result.placebo_dates, result.placebo_dates_reason),
        (result.placebo_units, result.placebo_units_reason),
        (result.leave_one_donor_out, result.leave_one_donor_out_reason),
    ):
        assert absent is None
        assert reason


def test_a_panel_estimator_gets_placebo_dates_and_placebo_units():
    result = falsify(uniform_panel(), "twfe", FAST, n_placebo_units=3)
    assert result.placebo_dates is not None
    assert result.placebo_units is not None
    assert result.leave_one_donor_out is None
    assert result.leave_one_donor_out_reason
    assert result.randomization_inference is None
    assert result.randomization_inference_reason


def test_a_donor_pool_estimator_gets_placebo_units_and_leave_one_donor_out():
    result = falsify(donor_panel(), "synthetic_control", FAST)
    assert result.placebo_units is not None
    # A tuple, possibly empty: only donors carrying weight above the 0.05
    # threshold are refit, and a diffuse weight vector can leave none. Empty is
    # a real answer, so the contract check is `is not None`, not a length.
    assert result.leave_one_donor_out is not None
    assert result.randomization_inference is None


def test_a_panel_too_small_for_a_placebo_group_says_so_rather_than_raising():
    # The placebo group is drawn from the never-treated pool and the units it
    # leaves behind are its controls, so it is roughly half the real treated
    # group. Measured on P1 at n_units=60 that is 15 placebo-treated against
    # 16 controls, which clears Requirements' minimum of 10 per arm but is
    # small enough that twfe's own bootstrap refuses -- more than half its
    # replicates come back degenerate. That must surface as a stated absence,
    # not as a ValueError raised three frames down inside the bootstrap.
    result = falsify(uniform_panel(n_units=60), "twfe", FAST, n_placebo_units=3)
    assert result.placebo_units is None
    assert result.placebo_units_reason
    assert "bootstrap" in result.placebo_units_reason
    # The rest of the report still stands: one technique declining does not
    # take the others with it.
    assert result.placebo_dates is not None


def test_a_donor_pool_states_the_real_reason_placebo_dates_are_absent():
    # placebo_leads builds its cells from a within-group variance that needs
    # at least two units a side (group_change returns nan below that), and a
    # donor pool has exactly one treated unit by construction -- so it yields
    # zero cells no matter how long the pre-period is. Measured on S1: 16
    # usable pre-treatment periods, still zero cells. The generic
    # empty-result reason would therefore report "no cohort has two
    # pre-treatment periods to compare", which on this data is simply untrue,
    # and a report whose stated reasons are false is worse than one that says
    # nothing. This pins the honest reason against that regression.
    result = falsify(donor_panel("S1"), "synthetic_control", FAST)
    assert result.placebo_dates is None
    assert result.placebo_dates_reason
    assert "one treated unit" in result.placebo_dates_reason
    assert "two pre-treatment periods" not in result.placebo_dates_reason


def test_every_absence_states_a_reason_that_is_not_boilerplate():
    # A reason that just repeats the field name teaches a reader nothing.
    result = falsify(cross_section(), "difference_in_means", FAST, n_permutations=20)
    for reason in (
        result.placebo_dates_reason,
        result.placebo_units_reason,
        result.leave_one_donor_out_reason,
    ):
        assert reason and len(reason.split()) >= 6


def test_randomization_inference_is_reproducible_under_a_fixed_seed():
    dataset = cross_section()
    first = randomization_inference(
        dataset, "lin_regression", FAST, n_permutations=20, seed=7
    )
    second = randomization_inference(
        dataset, "lin_regression", FAST, n_permutations=20, seed=7
    )
    assert np.array_equal(first.placebo_effects, second.placebo_effects)
    assert first.p_value == second.p_value


def test_a_different_seed_gives_a_different_permutation_draw():
    dataset = cross_section()
    first = randomization_inference(
        dataset, "lin_regression", FAST, n_permutations=20, seed=7
    )
    other = randomization_inference(
        dataset, "lin_regression", FAST, n_permutations=20, seed=8
    )
    assert not np.array_equal(first.placebo_effects, other.placebo_effects)


def test_the_real_effect_is_the_unpermuted_fit():
    dataset = cross_section()
    result = randomization_inference(
        dataset, "lin_regression", FAST, n_permutations=20, seed=0
    )
    direct = all_estimators()["lin_regression"]().fit(dataset, FAST)
    assert result.real_effect == pytest.approx(direct.point)


def test_the_permutation_p_value_cannot_go_below_one_over_n_plus_one():
    # Abadie's construction: the real unit always counts itself.
    result = randomization_inference(
        cross_section(), "lin_regression", FAST, n_permutations=20, seed=0
    )
    assert result.p_value >= 1 / 21
    assert result.placebo_effects.size == 20


def test_randomization_inference_refuses_a_panel():
    with pytest.raises(ValueError, match="panel"):
        randomization_inference(uniform_panel(), "twfe", FAST, n_permutations=5)


def test_an_unknown_estimator_name_is_refused():
    with pytest.raises(KeyError):
        falsify(cross_section(), "lin_regresion", FAST, n_permutations=5)
