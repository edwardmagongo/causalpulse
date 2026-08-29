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
    se=0.9 sp=0.9:   observed +0.03504 (bias -0.01496, MCSE 0.00420)
                     corrected +0.04381 (bias -0.00619, MCSE 0.00525)
    se=0.8 sp=0.95:  observed +0.03457 (bias -0.01543, MCSE 0.00462)
                     corrected +0.04506 (bias -0.00494, MCSE 0.00603)

  Both residual corrected-bias figures are under 1.2x their own MCSE --
  statistically indistinguishable from zero, i.e. genuine recovery, not a
  partial one.

  RE-MEASURED 2026-08-29 at the same protocol, and the tests below now assert
  the **se=0.8/sp=0.95** pair. At se=sp=0.9 on scenario A the true and
  observed treated shares coincide (measured share 0.49943), and that is the
  one configuration where the naive `(se + sp - 1)` factor equals the correct
  one -- so the end-to-end claim was being made exactly where a wrong formula
  would also have succeeded. The asymmetric pair puts the observed share at
  0.42510 against a true 0.5, where the two forms differ: attenuation 0.76690,
  against the naive 0.75.

  The bias reduction is the statistic that discriminates a correction from a
  no-op, so it is measured as its own paired quantity. Per seed, the
  correction moves the estimate by `corrected - observed`; over the 20 seeds
  that increment has mean 0.01049 and MCSE 0.00141, and it is exactly the
  amount by which the mean absolute bias falls (0.01543 -> 0.00494). A no-op
  correction would put that reduction at 0.00000.

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
SE, SP = 0.8, 0.95


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
    #
    # Asserted as a magnitude with margin, not as an ordering. A bare
    # `abs(observed.mean()) < abs(truth.mean())` is a strict `<` between two
    # numbers that are equal in expectation under exactly the failure it
    # guards against, which makes it roughly a coin flip rather than a test.
    # MEASURED: |truth| - |observed| = 0.05000 - 0.03457 = 0.01543, MCSE
    # 0.00462 (truth is a constant, so all of it comes from `observed`).
    # Floor set at the measured gap less 2 MCSE, 0.01543 - 0.00924 = 0.00618,
    # rounded down to 0.005 -- which is still about 1.1 MCSE clear of the
    # 0.00000 a DGP that mismeasured nothing would produce, so the guard has
    # margin in both directions rather than only against noise.
    observed, _, truth = misclassified_estimates()
    assert abs(truth.mean()) - abs(observed.mean()) >= 0.005


def test_the_correction_moves_the_estimate_measurably_closer_to_the_truth():
    # The assertion that discriminates a correction from a no-op, and the one
    # carrying the end-to-end claim. A `misclassification_curve` that returned
    # the point unchanged would put this reduction at exactly 0.00000.
    #
    # MEASURED: |observed bias| 0.01543 against |corrected bias| 0.00494, a
    # reduction of 0.01049. Its noise is the per-seed paired increment
    # `corrected - observed`, MCSE 0.00141. Floor is the reduction less
    # 3*MCSE = 0.01049 - 0.00423 = 0.00626, rounded down to 0.006 -- more than
    # four of its own MCSEs above the no-op's 0.00000.
    observed, corrected, truth = misclassified_estimates()
    reduction = abs((observed - truth).mean()) - abs((corrected - truth).mean())
    assert reduction >= 0.006


def test_the_correction_recovers_the_measured_true_effect():
    # The absolute half of the claim: the corrected estimate does not merely
    # move the right way, it lands on the truth. MEASURED: corrected bias
    # -0.00494 (MCSE 0.00603) against a true effect of +0.05000. Tolerance is
    # |bias| + 3*MCSE = 0.00494 + 0.01809 = 0.02303, rounded up to 0.025.
    #
    # This tolerance is wider than the 0.01543 uncorrected bias, so on its own
    # it would not fail a no-op correction -- which is why the paired test
    # above exists and carries the discrimination. Kept because "closer" and
    # "close enough" are different claims and this file asserts both.
    _, corrected, truth = misclassified_estimates()
    assert abs((corrected - truth).mean()) <= 0.025
