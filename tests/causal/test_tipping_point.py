"""Worst-case bounds on what the unobserved outcomes could do.

Assumption-free: every missing outcome is filled with the value most opposed
to the finding, then with the value most in its favour. The invariant that
matters is that the pair always contains the complete-data estimate -- an
identity, not a rate, so a single violation means the bounds are wrong.
"""

import numpy as np
import pandas as pd
import pytest
from cp_causal.registry import all_estimators
from cp_causal.sensitivity import _fill_missing_outcomes, _worst_case_bounds
from cp_core.dataset import CausalDataset, CausalSchema, complete_cases
from cp_core.values import EstimatorConfig
from cp_simulation.components import MissingnessMechanism
from cp_simulation.scenarios import build_scenario

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)


def hand_built(n1=500, n0=500, r1=400, r0=420, s1=240, s0=168) -> CausalDataset:
    """A frame with exactly the arm counts the worked example specifies."""
    treated = [1] * n1 + [0] * n0
    outcome: list[float] = (
        [1.0] * s1
        + [0.0] * (r1 - s1)
        + [np.nan] * (n1 - r1)
        + [1.0] * s0
        + [0.0] * (r0 - s0)
        + [np.nan] * (n0 - r0)
    )
    frame = pd.DataFrame(
        {
            "unit_id": np.arange(n1 + n0),
            "x1": np.zeros(n1 + n0),
            "treated": treated,
            "outcome": outcome,
        }
    )
    schema = CausalSchema(
        unit="unit_id",
        treatment="treated",
        outcome="outcome",
        covariates=("x1",),
        outcome_type="binary",
    )
    return CausalDataset(frame=frame, schema=schema)


def test_the_worked_example_reproduces():
    # n1=n0=500, r1=400, r0=420, s1=240, s0=168, so m1=100 and m0=80:
    #   complete-case  = 240/400 - 168/420 = 0.600 - 0.400 = 0.200
    #   adverse bound  = 240/500 - (168+80)/500 = 0.480 - 0.496 = -0.016
    #   favourable     = (240+100)/500 - 168/500 = 0.680 - 0.336 =  0.344
    # Re-derived here rather than copied: the plan's Task 6 header states a
    # favourable bound of 0.400, which its own test contradicts. 0.344 is the
    # value the stated formula produces.
    adverse, favourable = _worst_case_bounds(
        hand_built(), "difference_in_means", CONFIG
    )
    assert adverse == pytest.approx(-0.016, abs=1e-9)
    assert favourable == pytest.approx(0.344, abs=1e-9)


def test_the_bounds_contain_the_complete_case_estimate():
    dataset = hand_built()
    reduced, _ = complete_cases(dataset)
    complete_case = all_estimators()["difference_in_means"]().fit(reduced, CONFIG).point
    adverse, favourable = _worst_case_bounds(dataset, "difference_in_means", CONFIG)
    assert adverse <= complete_case <= favourable


def test_the_bounds_contain_the_complete_data_estimate_on_a_real_scenario():
    """The invariant, against ground truth rather than the observed subset.

    This is the identity the design calls the strongest check in the phase.
    The complete outcome vector is the one the DGP realised before any value
    was hidden, so the estimate below is the one the analyst would have got
    with no missingness at all -- and the bounds must contain it whatever the
    mechanism was, since they assume nothing about why values went missing.
    """
    simulated = build_scenario(
        "D",
        missingness=MissingnessMechanism(
            mechanism="MNAR", rate=0.25, strength=2.0
        ),
    ).generate(n_units=2000, rng=np.random.default_rng(0))
    frame = simulated.dataset.frame.copy()
    frame[simulated.dataset.schema.outcome] = simulated.truth.complete_outcome
    complete = CausalDataset(frame=frame, schema=simulated.dataset.schema)
    complete_estimate = (
        all_estimators()["difference_in_means"]().fit(complete, CONFIG).point
    )

    adverse, favourable = _worst_case_bounds(
        simulated.dataset, "difference_in_means", CONFIG
    )
    assert adverse <= complete_estimate <= favourable


ADMITTED_ESTIMATORS = (
    "difference_in_means",
    "lin_regression",
    "ipw",
    "aipw",
    "causal_forest",
)
"""Every non-panel estimator, which is exactly what `tipping_point`'s gate
admits: it refuses on `needs_panel`, on a non-binary outcome, and on nothing
missing. Enumerated rather than predicated, for the reason Phase 9's bsts
defect and Phase 10's misclassification defect both taught."""

CORNER_EXTREMAL = ("difference_in_means", "ipw")
"""The estimators whose per-unit weights are non-negative within an arm, so a
one-value fill of that arm really is the extreme. See `_worst_case_bounds`."""


@pytest.mark.parametrize("name", ADMITTED_ESTIMATORS)
def test_the_bounds_contain_the_complete_data_estimate_for_every_admitted_estimator(
    name,
):
    """The identity, on every estimator the gate lets through, not just one.

    Phase 11 wrote and tested `_worst_case_bounds` against
    `difference_in_means` alone; Phase 12 widened who could call it without
    revisiting the assumption, and every test in this file still used that one
    estimator. A gate admitting five estimators needs a test per estimator for
    every property its output claims.
    """
    simulated = build_scenario(
        "D",
        missingness=MissingnessMechanism(
            mechanism="MNAR", rate=0.25, strength=2.0
        ),
    ).generate(n_units=600, rng=np.random.default_rng(0))
    frame = simulated.dataset.frame.copy()
    frame[simulated.dataset.schema.outcome] = simulated.truth.complete_outcome
    complete = CausalDataset(frame=frame, schema=simulated.dataset.schema)
    truth = all_estimators()[name]().fit(complete, CONFIG).point

    adverse, favourable = _worst_case_bounds(simulated.dataset, name, CONFIG)
    assert adverse <= truth <= favourable


def _best_single_flip(dataset, name, config):
    """The lowest estimate reachable by flipping one filled unit off the corner."""
    from cp_causal.sensitivity import _adverse_values

    schema = dataset.schema
    treated_value, control_value = _adverse_values(dataset, name, config)
    filled = _fill_missing_outcomes(
        dataset, treated_value=treated_value, control_value=control_value
    )
    estimator = all_estimators()[name]
    best = estimator().fit(filled, config).point
    for row in np.flatnonzero(dataset.frame[schema.outcome].isna().to_numpy()):
        frame = filled.frame.copy()
        frame.loc[row, schema.outcome] = 1.0 - frame.loc[row, schema.outcome]
        point = estimator().fit(
            CausalDataset(frame=frame, schema=schema), config
        ).point
        best = min(best, point)
    return best


@pytest.mark.parametrize("name", ADMITTED_ESTIMATORS[:3])
def test_only_the_documented_estimators_have_an_extremal_corner(name):
    """Which estimators the corner fill is actually the extreme for.

    `TippingPoint` and `_worst_case_bounds` both name `difference_in_means`
    and `ipw` as the two whose per-unit weights keep a constant sign within
    an arm, so that filling the whole arm with one value cannot be beaten.
    This measures that claim instead of trusting it, by exhaustively flipping
    each filled unit off the corner and refitting.

    `lin_regression` is asserted to VIOLATE it -- its Lin-interacted
    coefficient is a linear functional of `y` whose per-unit weight can flip
    sign on units with extreme covariates. That direction is pinned
    deliberately: if a future change makes the corner extremal for it, this
    assertion fails and the field documentation naming the exception must be
    updated in the same commit, rather than quietly rotting into a false
    claim.
    """
    simulated = build_scenario(
        "D",
        missingness=MissingnessMechanism(
            mechanism="MNAR", rate=0.25, strength=2.0
        ),
    ).generate(n_units=400, rng=np.random.default_rng(0))
    adverse, _ = _worst_case_bounds(simulated.dataset, name, CONFIG)
    best = _best_single_flip(simulated.dataset, name, CONFIG)

    if name in CORNER_EXTREMAL:
        assert best >= adverse - 1e-12
    else:
        # MEASURED at bootstrap_reps=25, seed 1: corner -0.228862, best
        # single flip -0.229403. One flip is enough; no search is needed.
        assert best < adverse


@pytest.mark.parametrize("seed", range(25))
def test_the_bounds_contain_the_complete_data_estimate_across_seeds(seed):
    """The same identity, not resting on one lucky draw.

    It is an identity rather than a rate, so every seed must satisfy it --
    which is exactly why parametrizing is cheap insurance rather than
    over-testing. A sign error that only bites when the complete-case
    estimate lands near zero would survive a single-seed check.
    """
    simulated = build_scenario(
        "D",
        missingness=MissingnessMechanism(
            mechanism="MNAR", rate=0.3, strength=2.5
        ),
    ).generate(n_units=600, rng=np.random.default_rng(seed))
    frame = simulated.dataset.frame.copy()
    frame[simulated.dataset.schema.outcome] = simulated.truth.complete_outcome
    complete = CausalDataset(frame=frame, schema=simulated.dataset.schema)
    truth = all_estimators()["difference_in_means"]().fit(complete, CONFIG).point

    adverse, favourable = _worst_case_bounds(
        simulated.dataset, "difference_in_means", CONFIG
    )
    assert adverse <= truth <= favourable


def test_the_adverse_fill_opposes_a_negative_finding_the_other_way():
    """The sign logic, which a positive-only test cannot reach.

    With a negative complete-case estimate the adverse direction flips: the
    treated arm gets 1 and the control arm 0, pulling the estimate back
    toward zero. Measured on this fixture: complete-case -0.3167, bounds
    (-0.4400, -0.0800), so the end an adverse fill produces is -0.0800 --
    the one nearer zero. A sign error would put both bounds on the same side
    and this is where it surfaces.
    """
    dataset = hand_built(s1=140, s0=280)
    reduced, _ = complete_cases(dataset)
    complete_case = all_estimators()["difference_in_means"]().fit(reduced, CONFIG).point
    assert complete_case < 0
    low, high = _worst_case_bounds(dataset, "difference_in_means", CONFIG)
    assert low <= complete_case <= high
    # The adverse end is the one nearer the null. On a negative finding that
    # is the upper bound, which is the whole point of this test.
    assert abs(high) < abs(low)


def test_the_bounds_come_back_low_end_first_whatever_the_sign():
    """`_worst_case_bounds` orders by value, not by which fill is adverse.

    That is deliberate -- it keeps `TippingPoint`'s `adverse_bound <=
    favourable_bound` invariant true by construction -- but it means the
    first element is the *adverse* end only when the finding is positive. On
    a negative finding the adverse end comes back second, so a caller
    wanting it must re-derive it from the sign. Pinned here because the
    field names on the contract say `adverse`/`favourable` and would
    otherwise invite exactly that mistake.
    """
    for kwargs in ({}, {"s1": 140, "s0": 280}):
        low, high = _worst_case_bounds(
            hand_built(**kwargs), "difference_in_means", CONFIG
        )
        assert low <= high


def test_filling_replaces_only_the_missing_entries():
    dataset = hand_built()
    filled = _fill_missing_outcomes(dataset, treated_value=0.0, control_value=1.0)
    assert not filled.frame["outcome"].isna().any()
    observed = dataset.frame["outcome"].notna()
    assert (
        filled.frame.loc[observed, "outcome"].to_numpy()
        == dataset.frame.loc[observed, "outcome"].to_numpy()
    ).all()


def test_filling_a_complete_dataset_changes_nothing():
    dataset = build_scenario("D").generate(
        n_units=200, rng=np.random.default_rng(0)
    ).dataset
    filled = _fill_missing_outcomes(dataset, treated_value=0.0, control_value=1.0)
    assert filled.frame["outcome"].equals(dataset.frame["outcome"])


def test_a_dataset_with_no_missing_outcomes_returns_none():
    from cp_causal.sensitivity import tipping_point

    dataset = build_scenario("D").generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset
    assert tipping_point(dataset, "difference_in_means", CONFIG) is None


def test_a_continuous_outcome_returns_none():
    from cp_causal.sensitivity import tipping_point

    dataset = hand_built()
    continuous = CausalDataset(
        frame=dataset.frame,
        schema=CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=("x1",),
            outcome_type="continuous",
        ),
    )
    assert tipping_point(continuous, "difference_in_means", CONFIG) is None


def test_a_panel_estimator_returns_none():
    from cp_causal.sensitivity import tipping_point

    assert tipping_point(hand_built(), "twfe", CONFIG) is None


def test_an_unknown_estimator_is_refused_rather_than_silently_none():
    from cp_causal.sensitivity import tipping_point

    with pytest.raises(KeyError):
        tipping_point(hand_built(), "difference_in_meanz", CONFIG)


def test_the_result_reports_the_missing_count_and_agreeing_flags():
    from cp_causal.sensitivity import tipping_point

    result = tipping_point(hand_built(), "difference_in_means", CONFIG)
    assert result is not None
    assert result.n_missing_outcomes == 180  # 100 treated + 80 control
    assert result.adverse_bound <= result.favourable_bound
    assert (result.tipping_fraction is None) == result.conclusion_survives_worst_case


def test_the_point_based_bisection_agrees_with_the_closed_form():
    """Cross-check on the fill machinery, using the quantity that has one.

    The shipped fraction is defined on significance and has no closed form.
    This test bisects on the POINT estimate instead, where the design derived
    f* = (p1 - p0) / [m1*p1/n1 + m0*(1-p0)/n0], and checks the two agree. If
    they diverge, the fill or the arm bookkeeping is wrong -- independent of
    anything to do with intervals.
    """
    from cp_causal.sensitivity import _point_at_fraction

    n1 = n0 = 500
    r1, r0, s1, s0 = 400, 420, 240, 168
    m1, m0 = n1 - r1, n0 - r0
    p1, p0 = s1 / r1, s0 / r0
    closed_form = (p1 - p0) / (m1 * p1 / n1 + m0 * (1 - p0) / n0)

    dataset = hand_built()
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if _point_at_fraction(dataset, "difference_in_means", CONFIG, mid) > 0:
            low = mid
        else:
            high = mid
    # Tolerance measured, not guessed. The plan specifies abs=0.01, which is
    # unreachable: `n_adverse` is an integer count, so the treated arm's 100
    # gaps quantise the fraction at 1/100 = 0.01 before any noise at all, and
    # the non-adverse remainder is a random binomial draw on top of that.
    # Measured over 15 seeds: closed form 0.925926, bisection roots mean
    # 0.918333, sd 0.016350, spanning 0.881250 to 0.935000, so the largest
    # deviation seen is 0.0447. Tolerance 0.06 is that plus ~1 sd. The point
    # of this test is that the two agree to about a percent -- which is what
    # shows the arm bookkeeping is right -- not that they agree exactly.
    assert (low + high) / 2.0 == pytest.approx(closed_form, abs=0.06)


def test_sensitivity_report_carries_the_arm_when_outcomes_are_missing():
    from cp_causal.sensitivity import sensitivity_report

    dataset = hand_built()
    reduced, _ = complete_cases(dataset)
    estimate = all_estimators()["difference_in_means"]().fit(reduced, CONFIG)
    report = sensitivity_report(
        dataset, "difference_in_means", estimate, CONFIG, n_permutations=20
    )
    assert report.missing_data is not None
    assert report.missing_data_reason is None


def test_sensitivity_report_states_a_reason_on_complete_data():
    from cp_causal.sensitivity import sensitivity_report

    dataset = build_scenario("D").generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset
    estimate = all_estimators()["difference_in_means"]().fit(dataset, CONFIG)
    report = sensitivity_report(
        dataset, "difference_in_means", estimate, CONFIG, n_permutations=20
    )
    assert report.missing_data is None
    assert report.missing_data_reason
    assert len(report.missing_data_reason.split()) >= 6


def test_a_reversed_finding_is_not_reported_as_surviving():
    """The defect a significance-only rule ships.

    The plan defines the conclusion as `ci.contains(0.0)` and nothing more.
    Measured on scenario D at n_units=2000 under MNAR missingness at rate
    0.25, strength 2.0 (490 of 2000 outcomes missing): the complete-case
    estimate is +0.1002 with CI (0.0629, 0.1374), and filling every missing
    outcome against it gives -0.1176 with CI (-0.1535, -0.0819). That
    interval excludes zero, so a significance-only rule reports the
    conclusion as surviving its own worst case -- at the exact point the
    finding has been driven through zero and out the far side, which is the
    most complete reversal the data admits.

    A reader of that report would be told the finding is robust to any
    pattern of missingness. It is the opposite.
    """
    from cp_causal.sensitivity import tipping_point

    simulated = build_scenario(
        "D",
        missingness=MissingnessMechanism(
            mechanism="MNAR", rate=0.25, strength=2.0
        ),
    ).generate(n_units=2000, rng=np.random.default_rng(0))
    dataset = simulated.dataset
    reduced, _ = complete_cases(dataset)
    complete_case = all_estimators()["difference_in_means"]().fit(reduced, CONFIG)
    assert complete_case.point > 0
    assert not complete_case.ci.contains(0.0)

    result = tipping_point(dataset, "difference_in_means", CONFIG)
    assert result is not None
    # The adverse bound is on the far side of zero, so the finding cannot
    # survive its own worst case however significant that bound looks.
    assert result.adverse_bound < 0.0
    assert result.conclusion_survives_worst_case is False
    assert result.tipping_fraction is not None
    assert 0.0 < result.tipping_fraction < 1.0


def test_the_conclusion_test_rejects_a_flipped_sign_that_stays_significant():
    """The unit-level version of the case above, with no fitting involved."""
    from cp_causal.sensitivity import _conclusion_holds
    from cp_core.values import Interval

    class _Stub:
        def __init__(self, point, lower, upper):
            self.point = point
            self.ci = Interval(lower=lower, upper=upper)

    # Same sign, excludes zero -> holds.
    assert _conclusion_holds(_Stub(0.10, 0.06, 0.14), direction=0.10)
    # Covers zero -> does not hold.
    assert not _conclusion_holds(_Stub(0.02, -0.01, 0.05), direction=0.10)
    # Significant but flipped -> does not hold. This is the one the
    # significance-only rule gets wrong.
    assert not _conclusion_holds(_Stub(-0.12, -0.15, -0.08), direction=0.10)
    # And symmetrically for an originally negative finding.
    assert _conclusion_holds(_Stub(-0.12, -0.15, -0.08), direction=-0.10)
    assert not _conclusion_holds(_Stub(0.12, 0.08, 0.15), direction=-0.10)
