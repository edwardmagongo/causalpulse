"""What each Phase 18 axis does when pushed past the grid that ships it.

Task 10's contract, from the plan: at a degenerate value each axis must give
either **a clear refusal naming the problem, or a correct and describable
dataset -- not a silent degenerate one**. Phase 17's equivalent task found two
real silent-empty defects by asking exactly this, and Phase 16's found three.

What asking it here found, all measured before anything was asserted:

- An intercept extreme enough to empty the **treated** arm refuses, by name,
  out of `oracle_att` -- but the DGP itself encodes the undefined ATT as nan
  and does not raise, so the refusal lives one layer up from where a reader
  might look for it. Both halves are pinned below.
- An intercept that empties the **control** arm does not refuse and should
  not: truth is still defined, and every estimator is routed to `ineligible`
  with a reason naming the exact shortfall.
- `noise_scale` has no degenerate value at all. It saturates against
  `LogisticBaseline`'s `[0.05, 0.85]` clip and stays well-behaved at 1e6.
- The one genuinely silent case is `effect_scale` (and, more slowly,
  `heterogeneity_scale`) large enough that `np.clip(p0 + tau, 0.0, 1.0)`
  bites: measured ground truth stays correct, and `declared_effect` keeps
  advertising the unclipped value. That is a label defect, not a data one --
  `cp_evaluation` scores against the measurement -- and it is documented in
  `_homogeneous_declared_effect` and pinned here.

Every value the shipped grid uses is inside the safe region, which the last
test checks against `full_config()` itself rather than against a copy of its
numbers.
"""

import numpy as np
import pytest
from cp_evaluation.grid import CrossSectionAxes, GridConfig
from cp_evaluation.run import run_grid
from cp_evaluation.tiers import full_config
from cp_simulation.oracle import oracle_att
from cp_simulation.params import AXIS_NAMES, ScenarioParams
from cp_simulation.scenarios import SCENARIOS, build_scenario

# The label contract's own tolerance, from Task 4
# (`test_measured_ate_matches_the_declared_label`).
LABEL_TOLERANCE = 0.003

_SCENARIOS_WITH_AN_EFFECT = ("A", "C", "D", "F", "G", "H")


def _simulate(scenario_id, n_units=20_000, seed=11, **axes):
    params = ScenarioParams(**axes)
    simulated = build_scenario(scenario_id, params).generate(
        n_units=n_units, rng=np.random.default_rng(seed)
    )
    return params, simulated


def _arm_sizes(simulated):
    frame = simulated.dataset.frame
    treatment = frame[simulated.dataset.schema.treatment].to_numpy(dtype=float)
    return int(np.nansum(treatment == 1)), int(np.nansum(treatment == 0))


def _label_gap(scenario_id, **axes):
    params, simulated = _simulate(scenario_id, **axes)
    return float(simulated.truth.true_ate) - float(
        SCENARIOS[scenario_id].declared_effect(params)
    )


def test_an_intercept_that_empties_the_treated_arm_is_refused_by_name():
    # MEASURED: at n=20000 the treated arm still holds 127 units at
    # intercept -5 and 1 unit at -10; -20 is the first round value that
    # empties it completely. The DGP does not raise -- it writes nan for an
    # ATT with nothing to average, which is the honest encoding of undefined
    # -- so the refusal has to come from the oracle, and this pins both
    # halves. If `true_att` ever became 0.0 instead of nan, the oracle would
    # hand a benchmark a confident, wrong truth of zero.
    _, simulated = _simulate("A", treatment_intercept=-20.0)
    treated, control = _arm_sizes(simulated)

    assert treated == 0
    assert control == 20_000
    assert np.isnan(simulated.truth.true_att)
    # true_ate is a population quantity over p1 - p0 and does not depend on
    # who was assigned, so it stays defined and correct even here.
    assert simulated.truth.true_ate == pytest.approx(0.05, abs=1e-12)

    with pytest.raises(ValueError, match="no treated units"):
        oracle_att(simulated)


def test_an_intercept_that_empties_the_control_arm_leaves_every_estimator_ineligible_by_name():
    # The mirror case, and it is deliberately NOT a refusal: with every unit
    # treated, ground truth is still defined, so the honest outcome is a
    # dataset that generates cleanly and estimators that decline it by name.
    # Run end to end rather than against the DGP, because the naming this
    # test is about happens in routing, not in generation.
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",),
            n_units=(500,),
            confounding_strength=(1.0,),
            treatment_intercept=(0.0, 20.0),
        ),
        panel=None,
        replications=1,
    )

    frame = run_grid(config, estimators=["difference_in_means", "ipw", "aipw"])
    emptied = frame[frame["treatment_intercept"] == 20.0]

    assert len(emptied) == 3
    assert (emptied["status"] == "ineligible").all()
    assert (emptied["realised_treated_share"] == 1.0).all()
    # The reason has to name the shortfall, not merely decline: an
    # "ineligible" with an empty reason is the silent-degenerate outcome this
    # task exists to rule out.
    for reason in emptied["reason"]:
        assert "control units" in reason
        assert "has 0" in reason
    # And the baseline cell in the same run is unaffected.
    assert (frame[frame["treatment_intercept"] == 0.0]["status"] == "fitted").all()


@pytest.mark.parametrize("noise_scale", [2.0, 8.0, 1e6])
def test_a_saturating_noise_scale_stays_describable(noise_scale):
    # Spec section 4.2 promised the `[0.05, 0.85]` clip would make a large
    # noise_scale "saturate rather than produce degenerate cells". MEASURED:
    # it holds all the way to 1e6 -- both arms stay populated, the treated
    # share does not move, and because the clip bounds p0 away from 1 the
    # treated probability p0 + 0.05 never reaches the unit interval's edge,
    # so the label stays exact too. There is no degenerate value of this axis
    # to refuse.
    params, simulated = _simulate("A", noise_scale=noise_scale)
    treated, control = _arm_sizes(simulated)

    assert treated > 0
    assert control > 0
    assert simulated.truth.true_ate == pytest.approx(0.05, abs=1e-12)
    assert float(SCENARIOS["A"].declared_effect(params)) == pytest.approx(0.05)


@pytest.mark.parametrize("scenario_id", _SCENARIOS_WITH_AN_EFFECT)
def test_effect_scale_zero_removes_the_effect_exactly(scenario_id):
    # Exact, not approximate: zero times any tau is zero for a constant
    # effect, and scenario F's compensated base_tau is also multiplied by
    # effect_scale, so its mean goes to zero too. An implementation that
    # applied effect_scale to only one of base_tau/increment would leave a
    # residue here.
    params, simulated = _simulate(scenario_id, effect_scale=0.0)

    assert simulated.truth.true_ate == 0.0
    assert np.all(simulated.truth.true_cate == 0.0)
    assert float(SCENARIOS[scenario_id].declared_effect(params)) == 0.0


@pytest.mark.parametrize("scenario_id", _SCENARIOS_WITH_AN_EFFECT)
def test_a_large_effect_scale_clips_and_the_label_stops_agreeing(scenario_id):
    # The sharpest case the plan names. `dgp.py` forms p1 as
    # np.clip(p0 + tau, 0.0, 1.0), and its comment says truth is measured
    # rather than copied from tau *because* clipping would otherwise make the
    # recorded effect disagree with the actual one. That still holds -- and
    # the corollary is that `declared_effect`, which IS computed from tau,
    # goes wrong exactly where the clip bites.
    #
    # MEASURED, n=20000 x 8 seeds: at effect_scale=12 the gap is 0.0405-0.0427
    # (A, D, G, H), 0.0821-0.0845 (C) and 0.1382-0.1423 (F) -- at least 13x
    # the 0.003 label tolerance, so the separation is not seed luck.
    params, simulated = _simulate(scenario_id, effect_scale=12.0)
    declared = float(SCENARIOS[scenario_id].declared_effect(params))
    measured = float(simulated.truth.true_ate)

    assert abs(measured - declared) > 4 * LABEL_TOLERANCE
    # Measured truth stays a real probability difference; the label does not
    # have to. This is the asymmetry that makes the label the broken half.
    assert -1.0 <= measured <= 1.0
    assert abs(measured) < abs(declared)
    # Every per-unit effect stays inside what a probability difference can be,
    # which is what "the clip bit" means.
    assert np.all(np.abs(simulated.truth.true_cate) <= 1.0)


def test_an_absurd_effect_scale_makes_the_label_advertise_an_impossible_effect():
    # Stated as a property rather than a threshold: at effect_scale=100 the
    # label claims a 5.0 average effect on a binary outcome, which no
    # probability difference can be. Measured truth converges instead on
    # mean(1 - p0) -- every treated probability pinned at the ceiling.
    params, simulated = _simulate("A", effect_scale=100.0)

    assert float(SCENARIOS["A"].declared_effect(params)) > 1.0
    assert float(simulated.truth.true_ate) < 1.0
    assert np.all(simulated.truth.true_cate >= 0.0)


def test_heterogeneity_scale_zero_removes_the_spread_and_keeps_the_mean():
    # Spec section 4.4's whole point: heterogeneity_scale moves the spread,
    # not the level. At zero the spread must vanish *exactly* while the mean
    # stays put -- an implementation that scaled `increment` without
    # compensating `base_tau` would move the mean here, which is hazard 3 of
    # this phase.
    params, simulated = _simulate("F", heterogeneity_scale=0.0)

    # Zero to floating point, not literally 0.0: the compensated base_tau is
    # an arithmetic expression, so the per-unit values agree to within an ulp
    # rather than bit-identically (MEASURED: std 2.3e-17). The bound is still
    # ten orders of magnitude below any real spread -- the same scenario at
    # the shipped heterogeneity_scale=1.0 has std 0.0343, and at 2.0, 0.0685
    # -- so this is a tolerance on representation, not a relaxed threshold.
    assert float(np.std(simulated.truth.true_cate)) < 1e-12
    assert float(np.std(_simulate("F", heterogeneity_scale=1.0)[1].truth.true_cate)) > 0.03
    assert float(simulated.truth.true_ate) == pytest.approx(
        float(SCENARIOS["F"].declared_effect(params)), abs=LABEL_TOLERANCE
    )


def test_an_extreme_heterogeneity_scale_erodes_the_mean_it_promises_to_hold():
    # The same clip, reached the other way: a large enough `increment` pushes
    # p0 + tau past a bound for the units at one end of the covariate, so the
    # compensation that keeps the mean fixed stops being able to.
    # MEASURED, n=20000 x 8 seeds: 0.0288-0.0385 at heterogeneity_scale=20,
    # against a 0.003 tolerance. Pinned so the erosion is a known boundary
    # rather than a surprise to whoever first widens this axis.
    assert abs(_label_gap("F", heterogeneity_scale=20.0)) > 4 * LABEL_TOLERANCE
    # And it is genuinely a large-value effect, not a wiring error: the
    # shipped high point holds the mean fine.
    assert abs(_label_gap("F", heterogeneity_scale=2.0)) < LABEL_TOLERANCE


def test_every_value_the_full_tier_ships_is_inside_the_label_contract():
    # The tie-back: the tests above establish where each axis breaks, and
    # this one says the shipped grid does not go there. Read off
    # `full_config()` itself rather than a copy of its numbers, so respacing
    # an axis re-runs this check automatically instead of silently leaving it
    # describing the old grid.
    #
    # MEASURED: worst gap 0.00035 across every (declaring scenario x shipped
    # value) pair, against the 0.003 tolerance -- 8.6x of margin.
    axes = full_config().cross_section
    checked = 0

    for axis_name in AXIS_NAMES:
        for scenario_id in axes.scenarios:
            if axis_name not in SCENARIOS[scenario_id].axes:
                continue
            for value in getattr(axes, axis_name):
                gap = _label_gap(scenario_id, **{axis_name: float(value)})
                assert abs(gap) < LABEL_TOLERANCE, (
                    f"{scenario_id} at {axis_name}={value} has label gap {gap:.6f}, "
                    f"outside the {LABEL_TOLERANCE} the label contract allows"
                )
                checked += 1

    # The loop has to have done real work: a declaration table that stopped
    # matching would silently reduce this to nothing.
    assert checked >= 40
