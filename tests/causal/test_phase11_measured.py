"""Does complete-case analysis go wrong the way theory says, and do the bounds hold?

Spec section 7.3 pre-registered four expectations before this harness existed.
Three are rates and are asserted over seeds. The fourth -- that the worst-case
bounds contain the complete-data estimate -- is an identity, so it is asserted
without tolerance: one violation means the bounds are wrong.

MEASURED 2026-08-30, 20 seeds at n_units=2000, estimator difference_in_means,
scenario D unless noted, missingness MNAR(rate=0.25, strength=2.0):

  Complete-case bias, mean(complete-case point - complete-data point):
    MCAR (strength=0)   mean -0.00189  MCSE 0.00256
    MAR                 mean -0.03932  MCSE 0.00351  (context only, not asserted)
    MNAR                mean -0.07699  MCSE 0.00390
    MNAR, rate=0.05      mean -0.00711  MCSE 0.00135  (context only, not asserted)

  Worst-case bounds (_worst_case_bounds) against the complete-data estimate,
  scenario D: 0 violations of 20, mean bound width 0.4850. The bounds
  routinely span zero for this scenario's own effect size -- expected, not a
  defect: a 25% MNAR missing rate at strength 2.0 is a genuinely severe
  missingness pattern, and the spec's section 9 anticipates that a bound this
  wide is a property of the data, not of the bound.

  Tipping fraction (tipping_point), same MNAR settings:
    scenario B (null)    survived worst case 0/20   fraction mean 0.00675  MCSE 0.00423
    scenario D (+5pp)    survived worst case 0/20   fraction mean 0.24645  MCSE 0.01893
    D minus B difference: 0.23970, combined MCSE sqrt(0.01893^2+0.00423^2) = 0.01940.

Neither scenario's finding survives its own full adverse fill in any of the 20
seeds at this missingness rate -- the tipping fraction is what discriminates
them, not survival, since even D's true +5pp effect is not implausible for a
25%-missing, strength-2.0 MNAR mechanism to erase entirely at full adversity.

All four expectations held as stated; nothing here needed recording as a
negative result the way Phase 9's randomization-inference power did.
"""

import numpy as np
import pytest
from cp_causal.registry import all_estimators
from cp_causal.sensitivity import _worst_case_bounds, tipping_point
from cp_core.dataset import CausalDataset, complete_cases
from cp_core.values import EstimatorConfig
from cp_simulation.components import MissingnessMechanism
from cp_simulation.scenarios import build_scenario

pytestmark = pytest.mark.slow

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)
SEEDS = 20
N_UNITS = 2000
ESTIMATOR = "difference_in_means"


def complete_and_observed(scenario_id: str, mechanism: MissingnessMechanism, seed: int):
    dgp = build_scenario(scenario_id, missingness=mechanism)
    simulated = dgp.generate(n_units=N_UNITS, rng=np.random.default_rng(seed))
    frame = simulated.dataset.frame.copy()
    frame[simulated.dataset.schema.outcome] = simulated.truth.complete_outcome
    return (
        CausalDataset(frame=frame, schema=simulated.dataset.schema),
        simulated.dataset,
    )


def fit(dataset: CausalDataset) -> float:
    return all_estimators()[ESTIMATOR]().fit(dataset, CONFIG).point


def complete_case_gaps(mechanism: MissingnessMechanism) -> np.ndarray:
    gaps = []
    for seed in range(SEEDS):
        complete, observed = complete_and_observed("D", mechanism, seed)
        reduced, _ = complete_cases(observed)
        gaps.append(fit(reduced) - fit(complete))
    return np.asarray(gaps)


def tipping_fractions(scenario_id: str, mechanism: MissingnessMechanism) -> np.ndarray:
    fractions = []
    for seed in range(SEEDS):
        _, observed = complete_and_observed(scenario_id, mechanism, seed)
        result = tipping_point(observed, ESTIMATOR, CONFIG)
        assert result is not None and result.tipping_fraction is not None
        fractions.append(result.tipping_fraction)
    return np.asarray(fractions)


def test_complete_case_is_near_unbiased_when_data_are_missing_at_random():
    # strength=0 collapses the mechanism to MCAR, where complete-case analysis
    # is unbiased. The control: if this is biased, nothing below is readable.
    # MEASURED: mean gap -0.00189 (MCSE 0.00256). Ceiling is
    # |-0.00189| + 3*0.00256 = 0.00957, rounded up to 0.01.
    gaps = complete_case_gaps(
        MissingnessMechanism(mechanism="MNAR", rate=0.25, strength=0.0)
    )
    assert abs(gaps.mean()) <= 0.01


def test_complete_case_is_biased_when_missingness_depends_on_the_outcome():
    # The expectation scenario H exists for. Also the anti-vacuity guard: if
    # MNAR does not bias complete-case analysis, the mechanism is inert.
    # MEASURED: MNAR mean gap -0.07699 (MCSE 0.00390). Floor is
    # 0.07699 - 3*0.00390 = 0.06529, rounded down to 0.06 for real margin.
    gaps = complete_case_gaps(
        MissingnessMechanism(mechanism="MNAR", rate=0.25, strength=2.0)
    )
    assert abs(gaps.mean()) >= 0.06


def test_the_mnar_bias_exceeds_the_mcar_bias_by_a_measured_margin():
    # Asserted as a paired difference rather than two separate thresholds, so
    # the test discriminates the mechanism rather than the noise level.
    # MEASURED: MNAR 0.07699 minus MCAR 0.00189 = 0.07510, combined MCSE
    # sqrt(0.00256^2 + 0.00390^2) = 0.00467. Floor is 0.07510 - 3*0.00467 =
    # 0.06111, rounded down to 0.05 for real margin.
    mcar = abs(complete_case_gaps(
        MissingnessMechanism(mechanism="MNAR", rate=0.25, strength=0.0)
    ).mean())
    mnar = abs(complete_case_gaps(
        MissingnessMechanism(mechanism="MNAR", rate=0.25, strength=2.0)
    ).mean())
    assert mnar - mcar >= 0.05


def test_the_bounds_always_contain_the_complete_data_estimate():
    # An identity, not a rate: asserted with no tolerance and no headroom.
    # MEASURED: 0 violations of 20, mean bound width 0.4850.
    for seed in range(SEEDS):
        complete, observed = complete_and_observed(
            "D", MissingnessMechanism(mechanism="MNAR", rate=0.25, strength=2.0), seed
        )
        target = fit(complete)
        low, high = _worst_case_bounds(observed, ESTIMATOR, CONFIG)
        assert low <= target <= high, seed


def test_a_stronger_finding_takes_more_adverse_filling_to_overturn():
    # MEASURED: D mean fraction 0.24645 (MCSE 0.01893) over 20 seeds; B (null)
    # mean fraction 0.00675 (MCSE 0.00423), 0/20 survived either scenario's
    # own full adverse fill. Floor on D alone is 0.24645 - 3*0.01893 =
    # 0.18965, rounded down to 0.15 for real margin.
    mechanism = MissingnessMechanism(mechanism="MNAR", rate=0.25, strength=2.0)
    fractions = tipping_fractions("D", mechanism)
    assert np.mean(fractions) >= 0.15


def test_a_null_finding_is_overturned_by_a_far_smaller_adverse_share_than_a_real_one():
    # The comparison the test above's name promises but does not itself make:
    # a floor on D alone says D takes a lot of adverse filling to overturn,
    # not that it takes more than a scenario with no real effect to overturn.
    # MEASURED: D minus B = 0.24645 - 0.00675 = 0.23970, combined MCSE
    # sqrt(0.01893^2 + 0.00423^2) = 0.01940. Floor is 0.23970 - 3*0.01940 =
    # 0.18150, rounded down to 0.15 for real margin.
    mechanism = MissingnessMechanism(mechanism="MNAR", rate=0.25, strength=2.0)
    d_fractions = tipping_fractions("D", mechanism)
    b_fractions = tipping_fractions("B", mechanism)
    assert d_fractions.mean() - b_fractions.mean() >= 0.15
