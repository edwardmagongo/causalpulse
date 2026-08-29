"""Does falsification actually discriminate a real effect from a spurious one?

The routing and shape of these results are covered by test_falsification.py.
That the p-value *means* something is a different, empirical claim, and this
project's standard is that empirical claims are measured against known truth.

Spec section 7.3 pre-registered the expectation in both directions: on a
genuinely null scenario the real estimate should sit unremarkably inside the
permutation distribution; on a scenario with a strong true effect it should
sit in the tail. Both directions are asserted below, and both numbers were
measured under exactly this protocol -- same seed count, same permutation
count, same scenarios, same estimator -- not copied from the design document.

Scenario substitution: section 7.3 names P1/P2, which are panel scenarios.
Randomization inference is cross-sectional this phase, so the analogues are
A (declared_effect=0.05, true_ate measured at 0.05) and B (declared_effect=0.0,
a genuine null).

MEASURED 2026-08-28, 20 seeds x 200 permutations, n_units=800, lin_regression:
  Null (B):   share of seeds at p<=0.05 = 0.050 (MCSE 0.049), against a
              nominal 0.05 -- the permutation is well-calibrated.
  Effect (A): share of seeds at p<=0.05 = 0.300 (MCSE 0.102).

**The effect direction did NOT match section 7.3's pre-registered expectation
of "near 1.0", and this is recorded rather than patched (following the
precedent Phase 6 gave TWFE-on-P4 and Phase 7 gave its credible-interval
undercoverage).** This is not a broken permutation: the null direction above
holds almost exactly at nominal 0.05, which is precisely what a miscalibrated
or outcome-correlated permutation would NOT do. It is real, modest power.
Scenario A's declared effect (0.05) is small relative to the permutation
null's own spread at n_units=800 -- the null distribution's standard
deviation, measured directly from five individual seeds' placebo_effects, is
about 0.032, so the effect sits at roughly 1.6x the null's noise. A
one-sided-ish detection rate in the 30-40% range at that signal-to-noise
ratio is squarely what standard power arithmetic predicts; "near 1.0" was
never a supported expectation for this scenario/estimator/n combination, not
a claim this measurement broke.
`test_a_true_effect_is_detected_above_chance_but_not_near_certainty` asserts
what was actually measured instead of the pre-registered "near 1.0".

Leave-one-donor-out spread on S1 (clean donor scenario), 20 seeds: mean 0.2984
(MCSE 0.0494), min 0.0651, max 1.0137.

MEASURED 2026-08-29, in-time placebo sweep (the donor-pool family's
placebo_dates slot), 20 seeds, synthetic_control, S1 against S0 -- the same
DGP at tau=1.0 and tau=0.0:
  Effect (S1): mean p 0.2000 (MCSE 0.0450); share of seeds landing on the
               1/16 = 0.0625 floor 0.4500 (MCSE 0.1141).
  Null (S0):   mean p 0.6312 (MCSE 0.0586); share on the floor 0.0000 --
               0 of 20 seeds, minimum p-value seen 0.2500.
That pair is cleaner to compare than the cross-sectional one above. The sweep
reads only pre-treatment periods and tau enters only after adoption, so S0 and
S1 produce byte-identical placebo distributions at every one of the 20 seeds
(verified directly, not assumed). The entire difference between those two rows
is the real effect moving through a fixed reference distribution, which is
what makes the null direction a genuine guard rather than a second scenario
that happens to read differently.

Note that 0.0625 is this slot's smallest attainable p-value, so a "share at
p <= 0.05" reading of it -- the statistic the randomization-inference tests
above use -- would be identically zero for both scenarios and would measure
nothing. The share landing ON the floor is the analogous statistic here.

A note on the power of the true-effect bracket test below: with SEEDS=20, a
share of seeds is really a count out of 20, so the floor (>= 0.10) is a
break-even of 2-out-of-20 seeds landing at p<=0.05. That gives this bracket
limited power to catch a moderate regression in a single run. Under
Binomial(20, p), if the true detection rate degraded from the measured 0.30
down to 0.10, P(X>=2) ~= 0.61 -- the test would still pass about 61% of the
time even after that much power loss. If it collapsed further to ~0.05
(chance level, the same rate the null direction measures), P(X>=2) ~= 0.26 --
roughly one run in four would still pass. The ceiling added below catches a
rate that has drifted toward certainty, and repeated runs would eventually
show a floor failure if the collapse were severe or persistent, but this
test does not establish that a single run would reliably catch a moderate
regression at n=20 seeds: that would take more seeds, and this file
deliberately does not raise SEEDS because doing so would change the protocol
every other threshold here was measured under. The limitation is recorded
here rather than papered over by a test that implies more assurance than a
20-seed protocol can back up.
"""

import numpy as np
import pytest
from cp_causal.sensitivity import falsify, randomization_inference
from cp_core.values import EstimatorConfig
from cp_simulation.donor_scenarios import DONOR_SCENARIOS, build_donor_scenario
from cp_simulation.scenarios import build_scenario

pytestmark = pytest.mark.slow

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)
SEEDS = 20
N_PERMUTATIONS = 200
N_UNITS = 800


def p_values(scenario_id: str, estimator_name: str) -> np.ndarray:
    values = []
    for seed in range(SEEDS):
        dataset = build_scenario(scenario_id).generate(
            n_units=N_UNITS, rng=np.random.default_rng(seed)
        ).dataset
        values.append(
            randomization_inference(
                dataset, estimator_name, CONFIG,
                n_permutations=N_PERMUTATIONS, seed=seed,
            ).p_value
        )
    return np.asarray(values)


# The sweep's own floor. On the donor protocol S0/S1 run at (24 periods,
# adoption at 18) the pre-period is 17 periods, and a window needs 2 to train
# on plus at least one to measure over, so the fake date runs over training
# lengths 2..16 -- 15 windows. `_rank_p_value` counts the real effect itself,
# so the smallest p it can return is 1/(15 + 1).
IN_TIME_WINDOWS = 15
IN_TIME_FLOOR = 1.0 / (IN_TIME_WINDOWS + 1)


def in_time_p_values(scenario_id: str) -> np.ndarray:
    spec = DONOR_SCENARIOS[scenario_id]
    values = []
    for seed in range(SEEDS):
        dataset = build_donor_scenario(scenario_id).generate(
            spec.n_units, spec.n_periods, np.random.default_rng(seed)
        ).dataset
        placebo_dates = falsify(dataset, "synthetic_control", CONFIG).placebo_dates
        assert placebo_dates is not None
        assert placebo_dates.placebo_effects.size == IN_TIME_WINDOWS
        values.append(placebo_dates.p_value)
    return np.asarray(values)


def test_a_true_effect_is_detected_above_chance_but_not_near_certainty():
    # MEASURED: share of seeds at p <= 0.05 was 0.300 (MCSE 0.102) on scenario
    # A (declared_effect=0.05). Section 7.3 pre-registered "near 1.0"; that
    # did not hold (see module docstring for why -- real, modest power at
    # this effect/noise ratio, not a broken permutation). Floor set with
    # headroom below the measured mean: 0.300 - 2*MCSE = 0.300 - 0.204 =
    # 0.096, rounded up to 0.10 (conservative: raises the floor above the raw
    # two-MCSE bound). Ceiling set with headroom above the measured mean,
    # matching the test's own name ("not near certainty"): 0.300 + 2*MCSE =
    # 0.300 + 0.204 = 0.504, rounded down to 0.50 (conservative: tightens the
    # ceiling below the raw two-MCSE bound so a drift toward certainty is
    # caught rather than waved through). See the module docstring for this
    # bracket's power limitation at SEEDS=20.
    values = p_values("A", "lin_regression")
    rate = float(np.mean(values <= 0.05))
    assert rate >= 0.10
    assert rate <= 0.50


def test_a_true_null_does_not_land_in_the_tail():
    # The other half of the pre-registered expectation, and the half that
    # actually guards against a broken permutation: a test that only fires on
    # real effects would also fire on a permutation correlated with the outcome.
    # MEASURED: share of seeds at p <= 0.05 was 0.050 (MCSE 0.049) on scenario
    # B (a genuine null), against a nominal 0.05 -- matches essentially
    # exactly. Ceiling set with headroom above the measured mean: 0.050 +
    # 2*MCSE = 0.050 + 0.098 = 0.148, rounded to 0.15.
    values = p_values("B", "lin_regression")
    assert float(np.mean(values <= 0.05)) <= 0.15


def test_the_null_scenario_s_p_values_are_not_concentrated_at_the_floor():
    # A permutation distribution that always returns the minimum p-value is
    # broken in a way the rate test above can miss when the rate is small.
    # MEASURED: max p-value on scenario B was 0.9602, far above the floor
    # 1.5/(200+1) = 0.00746 this asserts against.
    values = p_values("B", "lin_regression")
    assert values.max() > 1.5 / (N_PERMUTATIONS + 1)


def test_leave_one_donor_out_stays_tight_on_a_clean_donor_scenario():
    # MEASURED on S1, 20 seeds: mean spread 0.2984 (MCSE 0.0494), max 1.0137.
    # A clean scenario's estimate should not hinge on any single donor.
    # Ceiling set with headroom above the measured mean: 0.2984 + 2*MCSE =
    # 0.2984 + 0.0988 = 0.3972, rounded to 0.40.
    spec = DONOR_SCENARIOS["S1"]
    spreads = []
    for seed in range(SEEDS):
        dataset = build_donor_scenario("S1").generate(
            spec.n_units, spec.n_periods, np.random.default_rng(seed)
        ).dataset
        points = np.asarray(
            falsify(dataset, "synthetic_control", CONFIG).leave_one_donor_out,
            dtype=float,
        )
        if points.size:
            spreads.append(float(points.max() - points.min()))
    assert float(np.mean(spreads)) <= 0.40


def test_the_in_time_placebo_ranks_a_true_effect_in_its_tail():
    # MEASURED on S1 (declared_effect=1.0), 20 seeds: the share of seeds whose
    # real effect ranked most extreme in its own sweep -- p on the 0.0625
    # floor -- was 0.4500 (MCSE 0.1141). Floor set with headroom below the
    # measured mean: 0.4500 - 2*MCSE = 0.4500 - 0.2282 = 0.2218, rounded DOWN
    # to 0.20.
    #
    # Rounded down rather than up, the opposite of the lin_regression bracket
    # above, and for a reason specific to this statistic: a share of 20 seeds
    # moves in steps of 0.05, and 0.2218 sits between two attainable values,
    # so a floor left at the raw bound is really a floor of 0.25 -- one seed
    # of ordinary drift from the measured 0.45 would fail it. 0.20 is the
    # nearest attainable value at or below the bound.
    values = in_time_p_values("S1")
    assert float(np.mean(values <= IN_TIME_FLOOR)) >= 0.20


def test_the_in_time_placebo_leaves_a_true_null_unremarkable():
    # The half that guards against a sweep that would rank anything in the
    # tail. S0 is S1's DGP at tau=0.0, and its placebo distribution is
    # byte-identical to S1's at every seed (see the module docstring), so this
    # measures the real effect's rank against a fixed reference and nothing
    # else.
    #
    # MEASURED on S0, 20 seeds: 0 of 20 seeds reached the 0.0625 floor, the
    # smallest p-value seen was 0.2500, and the mean p-value was 0.6312
    # (MCSE 0.0586).
    #
    # The share's MCSE is exactly 0.0000 at 0-of-20, which is a degenerate
    # estimate rather than a real one: two MCSE of headroom around it gives a
    # ceiling of 0.0 that a single seed's drift breaks. The rule of three is
    # used in its place -- with 0 events in 20 trials the 95% upper bound on
    # the true rate is about 3/20 = 0.15 -- and that is the ceiling asserted.
    values = in_time_p_values("S0")
    assert float(np.mean(values <= IN_TIME_FLOOR)) <= 0.15

    # The mean p-value has a real MCSE, so it gets the usual treatment: floor
    # set with headroom below the measured mean, 0.6312 - 2*MCSE = 0.6312 -
    # 0.1172 = 0.5140, rounded down to 0.50. It catches a sweep drifting
    # toward the tail well before the share above would.
    assert float(np.mean(values)) >= 0.50
