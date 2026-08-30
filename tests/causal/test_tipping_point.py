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
