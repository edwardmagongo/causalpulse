"""Falsification: does the finding survive tests it should fail if it were spurious?

This file tests routing and shape only -- which technique fires for which
estimator family, and that every absence carries a reason. Where the p-value
actually lands is an empirical claim measured in test_falsification_measured.py
under a protocol that can support it. Asserting a p-value here, on 20
permutations of one seed, would be measuring the rate from a single instance.

A subset is marked `slow`, which in this repo means `make test-fast` skips it
while `make test` and `make check` still run it. The subset is the bsts
assertions that need real posteriors: bsts fills `placebo_dates` from its own
in-time placebo sweep, a NUTS run per window, and no cheaper stand-in tests
what those assertions are for -- that the numbers reported under the bsts name
are bsts's own.

What `test-fast` keeps is deliberately not a reduced version of the contract.
Every routing and shape assertion in this file is made on synthetic_control,
which fits no posterior, and the bsts paths that sample nothing stay unmarked:
family membership, the window arithmetic, and S5 declining without a fit. Only
the assertions that are meaningless without a real posterior move. Measured:
28 tests in 8.3s with the subset skipped, 31 in 45s with it. See
`MCMC_DONOR_POOL` below.
"""

import dataclasses
import functools

import numpy as np
import pytest
from cp_causal._panel import donor_matrix, panel_arrays
from cp_causal.estimators.bsts import FAST_MCMC_CONFIG, bsts_in_time_placebo_sweep
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


def test_a_donor_pool_real_effect_matches_the_estimators_own_ridge_fallback_point():
    # S3 is this repo's own convex-hull-violation scenario: SyntheticControl.fit()
    # switches to the ridge-augmented weight form, and its reported point comes
    # from that form, not the plain simplex fit. An earlier version of falsify()
    # read `real_effect` from `PlaceboRuns.point_estimates[0]`, which is always
    # simplex-based (it exists to build the placebo null distribution, not to
    # report the headline number) -- on S3 that reported a sign-flipped point
    # about 2.5x off from what the estimator itself actually returns. Caught by
    # Task 7's review reproducing exactly this mismatch.
    from cp_causal.estimators.synthetic_control import SyntheticControl

    dataset = donor_panel("S3")
    fitted = SyntheticControl().fit(dataset, FAST)
    result = falsify(dataset, "synthetic_control", FAST)

    assert result.placebo_units is not None
    assert result.placebo_units.real_effect == pytest.approx(fitted.point)
    # Every per-donor leave-one-out estimate should sit near the real,
    # ridge-based point -- not near a value computed from the wrong weight
    # form, which the pre-fix code produced (an outlier an order of
    # magnitude off from every other value).
    assert result.leave_one_donor_out is not None
    for estimate in result.leave_one_donor_out:
        assert estimate == pytest.approx(fitted.point, abs=0.5)


def test_a_donor_pool_placebo_distribution_uses_the_same_form_as_the_real_effect():
    # The discriminating assertion: reconstruct every placebo unit's effect
    # through `fit_donor_pool_point` -- the same function the real effect goes
    # through -- and demand exact equality with what falsify reported.
    #
    # It discriminates because the two estimator forms agree exactly on the
    # 15 of 17 S3 donors whose own hull check passes. A distribution-level
    # assertion (mean, sd, a tolerance band) can therefore pass while the
    # slot still mixes forms. This one fails on the units where the forms
    # differ, which is the whole content of the finding: before the fix,
    # `placebo_effects` came from `placebo_runs`, uniformly simplex, and
    # donor 14 was reported at -1.96713 where its own hull-aware fit gives
    # -0.85951.
    from cp_causal.estimators.synthetic_control import fit_donor_pool_point

    dataset = donor_panel("S3")
    result = falsify(dataset, "synthetic_control", FAST)
    assert result.placebo_units is not None
    reported = np.asarray(result.placebo_units.placebo_effects)

    treated, donors, pre = donor_matrix(panel_arrays(dataset))
    post = ~pre
    assert reported.size == donors.shape[0]

    ridge_units = 0
    for index in range(donors.shape[0]):
        own_fit = fit_donor_pool_point(
            donors[index], np.delete(donors, index, axis=0), pre, post
        )
        assert reported[index] == own_fit.point
        ridge_units += int(own_fit.use_ridge)

    # Without this the test above is vacuous: if no placebo unit ever took
    # the ridge path on S3, hull-aware and simplex would coincide everywhere
    # and the assertion would hold for the wrong reason.
    assert ridge_units > 0

    # And the real effect is index 0 of that same swap, not a separately
    # derived number.
    real_fit = fit_donor_pool_point(treated, donors, pre, post)
    assert real_fit.use_ridge
    assert result.placebo_units.real_effect == real_fit.point


def test_the_donor_pool_p_value_stays_the_rmspe_ratio_rank():
    # Making the effects hull-aware deliberately did not touch inference.
    # Abadie's statistic is the post-to-pre RMSPE ratio's rank over the
    # simplex placebo runs, and `placebo_runs` still supplies it.
    from cp_causal.estimators.synthetic_control import permutation_p_value, placebo_runs

    dataset = donor_panel("S3")
    result = falsify(dataset, "synthetic_control", FAST)
    treated, donors, pre = donor_matrix(panel_arrays(dataset))
    runs = placebo_runs(treated, donors, pre, ~pre)

    assert result.placebo_units is not None
    assert result.placebo_units.p_value == permutation_p_value(runs.rmspe_ratios)


# bsts declares `max_treated_units == 1` exactly as synthetic_control does, so
# both land in the same `_family`. It runs NUTS, so it gets the fast MCMC
# config the language walk uses -- without it a single fit is a production
# posterior and this file would cost minutes.
DONOR_POOL_CONFIGS = {
    "synthetic_control": FAST,
    "bsts": dataclasses.replace(FAST, options={"mcmc_config": FAST_MCMC_CONFIG}),
}

# bsts now fills placebo_dates from its OWN in-time placebo sweep, and every
# window there is a full NUTS run at its own pre-period length. Capped at 3
# windows that is 4 posterior runs per falsify (3 placebo + 1 real), measured
# at about 22s under FAST_MCMC_CONFIG against 87s uncapped. The cap also sets
# the rank p-value's floor at 1/(3 + 1) = 0.25, which is why nothing below
# asserts a p-value smaller than that -- a cutoff the statistic cannot reach
# measures nothing.
BSTS_PLACEBO_DATES = 3

# Which donor-pool estimators pay a posterior refit per falsification, and so
# belong behind `slow` rather than in the default run. Derived from
# DONOR_POOL_CONFIGS below rather than hardcoded into each parametrize, so a
# third donor-pool estimator is routed here by editing one line.
#
# What the default run keeps: every routing and shape contract in this file is
# still asserted on synthetic_control, which fits no posterior, and the three
# bsts tests that fit no posterior either -- the family membership, the window
# arithmetic, and S5's decline-without-sampling -- stay unmarked. What moves to
# nightly is only the assertions that need real bsts posteriors to mean
# anything.
MCMC_DONOR_POOL = frozenset({"bsts"})

DONOR_POOL_PARAMS = [
    pytest.param(name, marks=pytest.mark.slow)
    if name in MCMC_DONOR_POOL
    else pytest.param(name)
    for name in sorted(DONOR_POOL_CONFIGS)
]


@functools.cache
def donor_pool_falsification(estimator_name):
    # Memoised so the whole file pays for one bsts falsification rather than
    # one per test. The tests below only read the result.
    return falsify(
        donor_panel("S1"),
        estimator_name,
        DONOR_POOL_CONFIGS[estimator_name],
        n_placebo_dates=BSTS_PLACEBO_DATES,
    )


@functools.cache
def donor_pool_own_point(estimator_name):
    return all_estimators()[estimator_name]().fit(
        donor_panel("S1"), DONOR_POOL_CONFIGS[estimator_name]
    ).point


def test_the_donor_pool_family_is_exactly_these_two():
    # The reasons bsts declines with name state-space regressors and shrinkage
    # priors, which is a fact about bsts rather than about the family. A third
    # single-treated-unit estimator would inherit that wording and it would be
    # false, so the membership is pinned rather than assumed.
    from cp_causal.sensitivity import _family

    donor_pool = {name for name in all_estimators() if _family(name) == "donor_pool"}
    assert donor_pool == {"bsts", "synthetic_control"}


@pytest.mark.parametrize("estimator_name", DONOR_POOL_PARAMS)
def test_a_donor_pool_result_belongs_to_the_estimator_it_is_reported_under(
    estimator_name,
):
    # `falsify` routes by `_family` almost everywhere, and the donor-pool
    # branch is the one place that is not enough: the machinery it calls is
    # synthetic_control's own and never consults the estimator name. Routed by
    # family alone it fitted a synthetic control for bsts and reported the
    # result with both reason fields None -- which asserts the numbers are
    # bsts's. Either a slot carries the named estimator's own point, or it is
    # absent with a reason.
    own_point = donor_pool_own_point(estimator_name)
    result = donor_pool_falsification(estimator_name)

    if result.placebo_units is None:
        assert result.placebo_units_reason
        assert result.leave_one_donor_out is None
        assert result.leave_one_donor_out_reason
    else:
        assert result.placebo_units.real_effect == pytest.approx(own_point)

    # placebo_dates is filled for BOTH members of the family now -- each from
    # its own machinery -- so the same rule applies to it: either the slot
    # carries the named estimator's own point, or it is absent with a reason.
    if result.placebo_dates is None:
        assert result.placebo_dates_reason
    else:
        assert result.placebo_dates.real_effect == pytest.approx(own_point)


@pytest.mark.slow
def test_bsts_does_not_borrow_synthetic_controls_donor_pool_results():
    bsts_point = donor_pool_own_point("bsts")
    synthetic_point = donor_pool_own_point("synthetic_control")
    # Non-vacuous: on S1 the two disagree (measured 1.17743 against 1.41401),
    # so a slot carrying the wrong one is visible rather than coincidental.
    assert bsts_point != pytest.approx(synthetic_point, rel=1e-3)

    result = donor_pool_falsification("bsts")
    # The two slots defined in terms of the simplex weight vector stay absent:
    # bsts does not fit one, and borrowing synthetic control's was the bug.
    assert result.placebo_units is None
    assert result.leave_one_donor_out is None
    for reason in (result.placebo_units_reason, result.leave_one_donor_out_reason):
        assert reason and "simplex" in reason

    # placebo_dates is different in kind and is filled. An in-time placebo
    # needs only that the estimator can be refitted at an earlier date, which
    # bsts can be -- so the slot carries bsts's own sweep. The discriminating
    # assertion is which estimator's number `real_effect` is: the two points
    # differ by about 0.24 on S1, so borrowing synthetic control's would be
    # visible here rather than a coincidence.
    assert result.placebo_dates is not None
    assert result.placebo_dates_reason is None
    assert result.placebo_dates.real_effect == pytest.approx(bsts_point)
    assert result.placebo_dates.real_effect != pytest.approx(synthetic_point, rel=1e-3)


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


def test_a_donor_pool_gets_placebo_dates_from_an_in_time_placebo_sweep():
    # The technique that fits this family is the in-time placebo: move the
    # treatment date back into the pre-period, refit the donor weights on what
    # comes before it, and measure the gap over what comes after. Sweeping the
    # fake date across the pre-period turns the single number
    # `SyntheticControl.fit` already reports as a diagnostic into a
    # distribution the real effect can be ranked against.
    #
    # It replaces a stated absence. `placebo_leads` genuinely cannot run here
    # -- it builds its cells from a within-group variance needing two units a
    # side and a donor pool has exactly one treated unit -- but "the panel
    # technique does not apply" was never the same claim as "no placebo-date
    # test applies".
    from cp_causal.estimators.synthetic_control import SyntheticControl

    dataset = donor_panel("S1")
    result = falsify(dataset, "synthetic_control", FAST)

    assert result.placebo_dates is not None
    assert result.placebo_dates_reason is None

    # Deterministic arithmetic off the scenario's shape, not a measured
    # quantity: S1 runs 24 periods with adoption at 18, so the pre-period is
    # 17 periods. A window needs at least MIN_IN_TIME_TRAIN_PERIODS = 2
    # periods to train on and at least one period left to measure over, so the
    # training length runs 2..16 -- 15 windows.
    assert result.placebo_dates.placebo_effects.size == 15

    # Same discipline `placebo_units` is already held to: the real effect is
    # the estimator's own reported point, not a separately derived number.
    assert result.placebo_dates.real_effect == pytest.approx(
        SyntheticControl().fit(dataset, FAST).point
    )
    # Abadie's floor, reused rather than recomputed -- 15 placebo effects plus
    # the real one counting itself.
    assert result.placebo_dates.p_value >= 1 / 16


def test_the_in_time_placebo_sweep_reads_only_the_pre_period():
    # The whole claim of an in-time placebo is that its distribution is a null
    # the real effect can be ranked against -- which requires that no window
    # ever sees data from after the real treatment date. Asserted directly:
    # move the treated unit's entire post-period by a large constant and the
    # sweep must not move at all. A window that leaked a single post-treatment
    # period into either its training or its measurement slice would shift.
    from cp_causal.estimators.synthetic_control import in_time_placebo_sweep

    treated, donors, pre = donor_matrix(panel_arrays(donor_panel("S1")))
    baseline = in_time_placebo_sweep(treated, donors, pre)

    shifted = treated.copy()
    shifted[~pre] += 100.0
    after = in_time_placebo_sweep(shifted, donors, pre)

    assert np.array_equal(baseline.point_estimates, after.point_estimates)
    # Non-vacuous only if the sweep produced something to compare.
    assert baseline.point_estimates.size == 15


def test_the_in_time_placebo_distribution_uses_the_same_form_as_the_real_effect():
    # The same finding `placebo_units` was already fixed for, applied to this
    # slot before it can regress: a placebo distribution fitted in one
    # estimator form and a real effect fitted in another are not comparable,
    # and ranking one against the other is the bug rather than the test.
    #
    # S3 is this repo's convex-hull-violation scenario, so the real fit takes
    # the ridge-augmented fallback. Every window is reconstructed through
    # `fit_donor_pool_point` -- the same function the real effect goes through
    # -- and demanded to match exactly.
    from cp_causal.estimators.synthetic_control import (
        fit_donor_pool_point,
        in_time_placebo_sweep,
    )

    dataset = donor_panel("S3")
    result = falsify(dataset, "synthetic_control", FAST)
    assert result.placebo_dates is not None

    treated, donors, pre = donor_matrix(panel_arrays(dataset))
    sweep = in_time_placebo_sweep(treated, donors, pre)
    assert np.array_equal(result.placebo_dates.placebo_effects, sweep.point_estimates)

    pre_positions = np.flatnonzero(pre)
    for offset, n_train in enumerate(sweep.train_lengths):
        fake_pre = np.zeros_like(pre)
        fake_post = np.zeros_like(pre)
        fake_pre[pre_positions[:n_train]] = True
        fake_post[pre_positions[n_train:]] = True
        own_fit = fit_donor_pool_point(treated, donors, fake_pre, fake_post)
        assert sweep.point_estimates[offset] == own_fit.point

    # Without this the assertion above holds for the wrong reason: if no
    # window ever took the ridge path, hull-aware and plain simplex would
    # coincide everywhere. Measured on S3: all 15 windows take it, as the real
    # fit does.
    assert int(sweep.used_ridge.sum()) == 15
    assert fit_donor_pool_point(treated, donors, pre, ~pre).use_ridge


def test_a_donor_pool_with_too_short_a_pre_period_states_that_reason():
    # S5 collapses S1's DGP to 6 periods with adoption at 3, leaving 2
    # pre-treatment periods. A window needs 2 to train on and one more to
    # measure over, so not one can be formed and the slot must go back to a
    # stated absence -- naming the length that actually fired, and not
    # claiming the technique does not exist for this family.
    result = falsify(donor_panel("S5"), "synthetic_control", FAST)
    assert result.placebo_dates is None
    reason = result.placebo_dates_reason
    assert reason is not None
    # The numbers the reason states have to be this design's real ones, not a
    # generic sentence with digits in it: 2 training periods plus one to
    # measure over is 3, and S5 has 2.
    assert "3 in total" in reason
    assert "This design has 2" in reason
    # And it must not fall back to claiming no placebo-date technique exists
    # for this family, which is what the slot used to say for every donor pool
    # however long its pre-period was.
    assert "one treated unit" not in reason
    # The other slots still stand: one technique declining does not take the
    # others with it.
    assert result.placebo_units is not None


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


def test_a_report_carries_all_three_lines_of_attack_for_a_supported_estimator():
    from cp_causal.registry import all_estimators
    from cp_causal.sensitivity import sensitivity_report

    dataset = cross_section()
    estimate = all_estimators()["lin_regression"]().fit(dataset, FAST)
    report = sensitivity_report(
        dataset, "lin_regression", estimate, FAST, n_permutations=20
    )
    assert report.confounding.e_value.e_value >= 1.0
    assert report.confounding.cinelli_hazlett is not None
    assert report.specification_curve is not None
    assert report.falsification.randomization_inference is not None


def test_an_unsupported_curve_becomes_a_stated_absence_not_an_exception():
    # specification_curve raises for a direct caller, which is right. Inside a
    # report the same fact is a stated absence, and this is the one place that
    # translation happens.
    from cp_causal.registry import all_estimators
    from cp_causal.sensitivity import sensitivity_report

    dataset = cross_section()
    estimate = all_estimators()["difference_in_means"]().fit(dataset, FAST)
    report = sensitivity_report(
        dataset, "difference_in_means", estimate, FAST, n_permutations=20
    )
    assert report.specification_curve is None
    assert "difference_in_means" in report.specification_curve_reason


def test_a_non_ols_estimator_gets_an_e_value_but_no_cinelli_hazlett():
    from cp_causal.registry import all_estimators
    from cp_causal.sensitivity import sensitivity_report

    dataset = cross_section()
    estimate = all_estimators()["ipw"]().fit(dataset, FAST)
    report = sensitivity_report(dataset, "ipw", estimate, FAST, n_permutations=20)
    assert report.confounding.cinelli_hazlett is None
    assert report.confounding.cinelli_hazlett_reason
    assert report.confounding.e_value.e_value >= 1.0


def test_a_supported_estimators_curve_failure_propagates_not_absorbed(monkeypatch):
    # sensitivity_report must not relabel a genuine failure from inside a real
    # curve computation as "this estimator is unsupported". specification_curve
    # can raise ValueError for a *supported* estimator too -- e.g. when the
    # base specification cannot be fitted on the dataset as declared, or from
    # inside a candidate refit (aipw's cross-fitting-fold-too-small guard).
    # Arranging either through a real dataset would be a contrived fixture in
    # its own right, so the failure is forced directly by monkeypatching
    # specification_curve in cp_causal.sensitivity's own module namespace --
    # the only name sensitivity_report calls.
    import cp_causal.sensitivity as sensitivity_module
    from cp_causal.registry import all_estimators
    from cp_causal.sensitivity import sensitivity_report

    def _boom(*args, **kwargs):
        raise ValueError(
            "cross-fitting produced a fold with too few units in one arm"
        )

    monkeypatch.setattr(sensitivity_module, "specification_curve", _boom)

    dataset = cross_section()
    estimate = all_estimators()["lin_regression"]().fit(dataset, FAST)
    with pytest.raises(ValueError, match="cross-fitting"):
        sensitivity_report(
            dataset, "lin_regression", estimate, FAST, n_permutations=20
        )


def test_a_degenerate_ols_refit_is_not_reported_as_not_ols_fitted(monkeypatch):
    # cinelli_hazlett returns None for an estimator that IS in
    # OLS_SENSITIVITY_ESTIMATORS when its own refit is numerically degenerate
    # (df < 1, or a zero coefficient standard error) -- a different fact from
    # "this estimator isn't OLS-fitted", which is what _NOT_OLS_REASON would
    # wrongly claim. Forcing that degeneracy through a real dataset would need
    # a contrived fixture (e.g. an n=2 sample), so it's forced directly by
    # monkeypatching cinelli_hazlett in cp_causal.sensitivity's own module
    # namespace -- the only name sensitivity_report calls.
    import cp_causal.sensitivity as sensitivity_module
    from cp_causal.registry import all_estimators
    from cp_causal.sensitivity import sensitivity_report

    monkeypatch.setattr(sensitivity_module, "cinelli_hazlett", lambda *a, **k: None)

    dataset = cross_section()
    estimate = all_estimators()["lin_regression"]().fit(dataset, FAST)
    report = sensitivity_report(
        dataset, "lin_regression", estimate, FAST, n_permutations=20
    )
    assert report.confounding.cinelli_hazlett is None
    reason = report.confounding.cinelli_hazlett_reason
    assert reason is not None
    assert "lin_regression" in reason
    assert "not fitted by least squares" not in reason


def test_a_panel_estimator_gets_no_misclassification_curve_but_a_deferral():
    # The misclassification arm used to gate on the estimate's scale alone, so
    # it fired for every panel and donor-pool estimator reporting a difference
    # of means -- computing a treated share from arm counts that on a panel
    # count units while the regressor is the unit x period indicator, and on
    # the donor-pool estimators are one hardcoded treated unit against the
    # donor count. The README defers panel variants of both of this phase's
    # checks; this is what makes that true for the second of them.
    from cp_causal.sensitivity import sensitivity_report

    dataset = uniform_panel()
    estimate = all_estimators()["twfe"]().fit(dataset, FAST)
    report = sensitivity_report(
        dataset,
        "twfe",
        estimate,
        FAST,
        outcome_sd=1.0,
        n_permutations=20,
        n_placebo_units=2,
    )
    assert report.misclassification is None
    reason = report.misclassification_reason
    assert reason is not None
    assert "twfe" in reason
    assert "deferred" in reason
    # Not the log-ratio refusal wearing a panel estimator's name: twfe reports
    # a difference of means, so the scale was never the disqualifying fact.
    assert estimate.scale != "log_ratio"
    assert "log ratio" not in reason


def test_a_curve_with_only_its_identity_anchor_becomes_a_stated_absence():
    # At a low enough treated share every non-identity pair in the default
    # grid is refused, and the curve that survives holds the analysis as
    # already run and nothing else. Returned as a curve it reads as
    # "mismeasurement cannot move this finding"; the truth is that no assumed
    # rate pair is usable at this share, so the report states that instead.
    # The share is set on the estimate rather than by building a 5%-treated
    # dataset: this arm reads its prevalence off the estimate's arm counts and
    # nothing else, so that is the input under test.
    from cp_causal.sensitivity import sensitivity_report

    dataset = cross_section()
    fitted = all_estimators()["lin_regression"]().fit(dataset, FAST)
    estimate = dataclasses.replace(fitted, n_treated=50, n_control=950)
    report = sensitivity_report(
        dataset, "lin_regression", estimate, FAST, n_permutations=20
    )
    assert report.misclassification is None
    reason = report.misclassification_reason
    assert reason is not None
    assert "0.0500" in reason
    assert "identity" in reason


def test_a_panel_with_no_treated_units_names_its_own_condition():
    # This path used to return the placebo-DATES reason ("the placebo-date
    # cells cannot be formed"), which names the wrong technique and states a
    # condition -- too short a pre-period -- that is not the one that fired.
    # Reached directly: a panel with nothing ever treated does not survive
    # falsify's other branches, and the reason a slot gives is worth pinning
    # on its own.
    from cp_causal.sensitivity import _panel_placebo_units
    from cp_core.dataset import CausalDataset

    dataset = uniform_panel()
    frame = dataset.frame.copy()
    frame[dataset.schema.treatment] = 0
    if dataset.schema.cohort is not None:
        frame[dataset.schema.cohort] = 0
    never_treated = CausalDataset(frame=frame, schema=dataset.schema)

    result, reason = _panel_placebo_units(never_treated, "twfe", FAST, 3, 0)
    assert result is None
    assert reason is not None
    assert "ever treated" in reason
    assert "placebo-date" not in reason


@pytest.mark.slow
def test_the_bsts_placebo_dates_come_from_bsts_own_posterior():
    # The discriminating test for this slot, and the counterpart of
    # `test_the_in_time_placebo_distribution_uses_the_same_form_as_the_real
    # _effect` on the synthetic-control side: reconstruct each window through
    # `fit_bsts_counterfactual` -- the same pipeline `BSTS.fit` reports its
    # point from -- and demand exact equality with what falsify reported.
    #
    # A distribution-level check would not discriminate. What it has to rule
    # out is the slot carrying synthetic control's sweep, so the assertion
    # that carries the finding is that these are bsts posteriors, window by
    # window.
    from cp_causal._panel import split_pre_period
    from cp_causal.estimators.bsts import fit_bsts_counterfactual, resolve_mcmc_config

    config = DONOR_POOL_CONFIGS["bsts"]
    result = donor_pool_falsification("bsts")
    assert result.placebo_dates is not None
    reported = np.asarray(result.placebo_dates.placebo_effects)
    assert reported.size == BSTS_PLACEBO_DATES

    treated, donors, pre = donor_matrix(panel_arrays(donor_panel("S1")))
    sweep = bsts_in_time_placebo_sweep(
        treated,
        donors,
        pre,
        mcmc_config=resolve_mcmc_config(config),
        random_seed=config.random_seed,
        ci_level=config.ci_level,
        max_windows=BSTS_PLACEBO_DATES,
    )
    assert np.array_equal(reported, sweep.point_estimates)

    # Spot-check one window against the pipeline directly, so this does not
    # merely assert the sweep equals itself.
    fake_pre, fake_post = split_pre_period(pre, sweep.train_lengths[0])
    own = fit_bsts_counterfactual(
        treated,
        donors,
        fake_pre,
        fake_post,
        mcmc_config=resolve_mcmc_config(config),
        random_seed=config.random_seed,
        ci_level=config.ci_level,
    )
    assert sweep.point_estimates[0] == own.point

    # And it is NOT synthetic control's sweep, which is the regression this
    # slot is exposed to. Measured on S1: the two agree on no window.
    from cp_causal.estimators.synthetic_control import in_time_placebo_sweep

    sc_sweep = in_time_placebo_sweep(
        treated, donors, pre, max_windows=BSTS_PLACEBO_DATES
    )
    assert sc_sweep.train_lengths == sweep.train_lengths
    assert not np.any(np.isclose(sc_sweep.point_estimates, sweep.point_estimates))


def test_a_capped_sweep_keeps_both_ends_of_the_pre_period():
    # The cap exists to buy time on bsts, and what it must not do is spend
    # that budget on one end of the range: short and long training windows
    # behave differently, so a cap that sampled only short ones would build a
    # reference distribution from the least representative windows.
    from cp_causal._panel import in_time_placebo_windows

    treated, donors, pre = donor_matrix(panel_arrays(donor_panel("S1")))
    full = in_time_placebo_windows(pre)
    assert full == tuple(range(2, 17))

    capped = in_time_placebo_windows(pre, max_windows=BSTS_PLACEBO_DATES)
    assert len(capped) == BSTS_PLACEBO_DATES
    assert capped[0] == full[0]
    assert capped[-1] == full[-1]
    # A cap at or above the available count is a no-op rather than an error.
    assert in_time_placebo_windows(pre, max_windows=99) == full


def test_a_bsts_pre_period_too_short_declines_without_fitting_a_posterior():
    # S5 leaves 2 pre-treatment periods, so no window can be formed. bsts must
    # reach the same stated absence synthetic_control does -- and reach it
    # without sampling, which matters far more here: an estimator that spent
    # a NUTS run per window only to discover it had no windows would cost
    # minutes to say nothing.
    #
    # Asserted structurally rather than by timing it. If the window list is
    # empty then the sweep's comprehension never runs a body, so no posterior
    # can have been fitted; a wall-clock bound would test the same fact more
    # flakily.
    from cp_causal._panel import in_time_placebo_windows

    dataset = donor_panel("S5")
    _, _, pre = donor_matrix(panel_arrays(dataset))
    assert in_time_placebo_windows(pre) == ()

    result = falsify(
        dataset,
        "bsts",
        DONOR_POOL_CONFIGS["bsts"],
        n_placebo_dates=BSTS_PLACEBO_DATES,
    )
    assert result.placebo_dates is None
    reason = result.placebo_dates_reason
    assert reason is not None
    assert "3 in total" in reason
    assert "This design has 2" in reason
