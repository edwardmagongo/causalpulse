"""Do the two new checks discriminate, measured against known truth?

The routing and formulas are covered by the fast tests. That the checks
*mean* something is an empirical claim, and this project's standard is that
empirical claims are measured.

Spec section 7.3 pre-registered three expectations before this harness
existed. Each is a rate, so each is asserted as a rate over seeds rather than
demonstrated on one.

MEASURED 2026-08-29, 20 seeds at n_units=2000:

  Expectation 1 (randomised scenario, control rarely fires):
    A/difference_in_means  0.000 (MCSE 0.000)
    A/lin_regression       0.050 (MCSE 0.049)
    A/ipw                  0.000 (MCSE 0.000)
    A/aipw                 0.050 (MCSE 0.049)

  Expectation 2 (confounded scenario, unadjusted fires, adjusted does not):
    D/difference_in_means  1.000 (MCSE 0.000)   G/difference_in_means  1.000 (MCSE 0.000)
    D/lin_regression       0.000 (MCSE 0.000)   G/lin_regression       0.000 (MCSE 0.000)
    D/ipw                  0.000 (MCSE 0.000)   G/ipw                  0.050 (MCSE 0.049)
    D/aipw                 0.000 (MCSE 0.000)   G/aipw                 0.000 (MCSE 0.000)

  Every rate not at 0.000 or 0.050 held there too; the separation between
  `difference_in_means` and every adjusting estimator on D and G is total
  across all 20 seeds, not a near-miss. Both `D` and `G`'s `1.000`/`0.000`
  pairs are the same shape this project has hit before (a unanimous k-of-N
  share carries MCSE 0.000 by the binomial formula, which is a property of
  the formula at that share, not evidence of certainty) -- thresholds below
  use the rule of three rather than `mean +/- k*MCSE` wherever a rate landed
  at 0 or 1, since `mean +/- 2*0` would assert zero margin against a share
  that a single future miss could break.

  Expectation 3 (misclassification recovery), difference_in_means, true ATE
  0.05000:
    se=0.9 sp=0.9:   observed +0.03504 (bias -0.01496)
                     corrected +0.04381 (bias -0.00619, MCSE 0.00525)
    se=0.8 sp=0.95:  observed +0.03457 (bias -0.01543)
                     corrected +0.04506 (bias -0.00494, MCSE 0.00603)

  Both residual corrected-bias figures are under 1.2x their own MCSE --
  statistically indistinguishable from zero, i.e. genuine recovery, not a
  partial one. Test below uses the se=0.9/sp=0.9 pair.

All three pre-registered expectations held as stated; nothing here needed
recording as a negative result the way Phase 9's randomization-inference
power did.
"""

import numpy as np
import pytest
from cp_causal.registry import all_estimators
from cp_causal.sensitivity import misclassification_curve, negative_control_test
from cp_core.values import EstimatorConfig
from cp_simulation.components import NonDifferentialMisclassification
from cp_simulation.scenarios import DEFAULT_NEGATIVE_CONTROL, build_scenario

pytestmark = pytest.mark.slow

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)
SEEDS = 20
N_UNITS = 2000
SE, SP = 0.9, 0.9


def firing_rate(scenario_id: str, estimator_name: str) -> float:
    fired = []
    for seed in range(SEEDS):
        dataset = build_scenario(
            scenario_id, negative_control=DEFAULT_NEGATIVE_CONTROL
        ).generate(n_units=N_UNITS, rng=np.random.default_rng(seed)).dataset
        result = negative_control_test(dataset, estimator_name, CONFIG)
        assert result is not None
        fired.append(result.excludes_null)
    return float(np.mean(fired))


def misclassified_estimates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(observed, corrected, truth)` over the seeds, at the measured rates."""
    observed, corrected, truth = [], [], []
    for seed in range(SEEDS):
        simulated = build_scenario(
            "A",
            misclassification=NonDifferentialMisclassification(
                sensitivity=SE, specificity=SP
            ),
        ).generate(n_units=N_UNITS, rng=np.random.default_rng(seed))
        estimate = all_estimators()["difference_in_means"]().fit(
            simulated.dataset, CONFIG
        )
        observed.append(estimate.point)
        corrected.append(
            misclassification_curve(estimate, rates=((SE, SP),)).points[0].corrected_point
        )
        truth.append(simulated.truth.true_ate)
    return np.asarray(observed), np.asarray(corrected), np.asarray(truth)


def test_the_negative_control_rarely_fires_on_a_randomised_scenario():
    # Scenario A is randomised, so there is no confounding for the control to
    # detect and the firing rate should sit near the nominal 0.05.
    # MEASURED: 0.050 (MCSE 0.049). Ceiling is 0.050 + 2*MCSE = 0.148, rounded
    # up to 0.15.
    assert firing_rate("A", "lin_regression") <= 0.15


def test_the_negative_control_fires_on_a_confounded_scenario_when_unadjusted():
    # difference_in_means adjusts for nothing, so on D the control shares the
    # confounders with treatment and should fire often.
    # MEASURED: 1.000 (20/20, MCSE 0.000 -- a unanimous share, not certainty).
    # The rule of three puts the true rate's lower bound at roughly
    # 1 - 3/20 = 0.85; floor set below that, at 0.80, for real margin against
    # a single future miss rather than asserting the observed 1.000 itself.
    assert firing_rate("D", "difference_in_means") >= 0.80


def test_an_adjusting_estimator_fires_less_often_than_an_unadjusted_one():
    # The comparison that makes the check informative rather than merely
    # sensitive: a control that fires for everyone would say nothing about
    # whether the adjustment worked.
    # MEASURED: unadjusted 1.000 (20/20), adjusted (aipw) 0.000 (0/20); both
    # unanimous, both MCSE 0.000 by the binomial formula at those shares. The
    # rule of three bounds the true unadjusted rate at >= ~0.85 and the true
    # adjusted rate at <= ~0.15, so the true difference is at least ~0.70 by
    # that same conservative reasoning. Floor set well under that, at 0.50,
    # for real margin.
    unadjusted = firing_rate("D", "difference_in_means")
    adjusted = firing_rate("D", "aipw")
    assert unadjusted - adjusted >= 0.50


def test_the_uncorrected_estimate_is_attenuated_under_misclassification():
    # Guards the next test against passing vacuously. If the observed estimate
    # is not attenuated, the data-generating process is not mismeasuring
    # anything and a "recovery" would be recovering from nothing.
    # MEASURED: observed bias -0.01496 against a true effect of +0.05000 --
    # about 30% attenuation, comfortably real.
    observed, _, truth = misclassified_estimates()
    assert abs(observed.mean()) < abs(truth.mean())


def test_the_correction_recovers_the_measured_true_effect():
    # MEASURED: corrected bias -0.00619 (MCSE 0.00525) against a true effect
    # of +0.05000. Tolerance is |bias| + 3*MCSE = 0.00619 + 0.01575 = 0.02194,
    # rounded up to 0.025.
    _, corrected, truth = misclassified_estimates()
    assert abs((corrected - truth).mean()) <= 0.025
