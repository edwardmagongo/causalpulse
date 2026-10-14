"""Does imputation help, and where does it stop helping?

MEASURED 2026-09-01 (Task 9), 40 seeds at n_units=2000, m=20, scenario D,
MAR/MNAR(rate=0.25, strength=2.0), OutcomeProxy(noise_sd=...), through
SHIPPED code -- not the discarded prototype, whose figures
are wrong in several places. All numbers below come from that run; "gap" =
arm's point estimate minus the complete-data estimate on the same seed, and
"paired" statistics are per-seed differences, which is the powerful comparison
here because all four arms share seeds, datasets and missing rows.

## Expectation 1 -- MI repairs difference_in_means under MAR (HOLDS)

`difference_in_means`, MAR, noise_sd=0.5, 40 seeds:

  arm            gap        gap MCSE
  complete_case  -0.04011   0.00232
  single         -0.00319   0.00182
  mi_no_aux      -0.00181   0.00256
  mi_aux         -0.00056   0.00150

Complete-case is 17 MCSE from zero; every imputed arm is within 1 MCSE of it.

## Expectation 2 -- FALSIFIED (benignly)

Pre-registered: MI without an auxiliary is *less efficient* than complete-case
(prototype: empirical SD 0.02831 vs 0.02619, "about 8% worse").

Measured, paired, `lin_regression` MAR, 40 seeds:
`mi_no_aux - complete_case` = -7.106e-06, MCSE 2.854e-05, ratio -0.25 --
indistinguishable. RMSE: `mi_no_aux` 0.01485 vs `complete_case` 0.01509
(ratio 0.984).

The prototype's penalty was noise. Indistinguishability is the theoretically
correct answer (Little 1992): imputing outcomes with no auxiliary
information gains nothing and loses nothing.

## Expectation 3 -- HOLDS at every strength this seed count can see (paired)

`lin_regression`, MAR, 40 seeds. Paired `mi_aux - complete_case`:

  noise_sd  paired diff  MCSE       ratio  RMSE mi_aux  RMSE complete_case
  0.25      -2.094e-04   4.725e-05  -4.43  0.00428      0.01509
  0.5       -1.243e-04   4.547e-05  -2.73  0.01017      0.01509

At 100 seeds noise_sd=1.0 also separated (ratio -2.24), but at 40 seeds MCSE
inflates by 1.58x and it would land near 1.4 -- below the bar. This harness
does not assert an efficiency result at noise_sd=1.0.

## Expectation 4 -- HOLDS, and more cleanly than the prototype

SE calibration (mean reported SE / empirical SD; 1.0 is honest),
`lin_regression` MAR, 40 seeds:

  noise_sd  complete_case  single   mi_no_aux  mi_aux
  0.25      0.9377         0.8735*  0.9455     0.9278
  0.5       0.9377         0.7879*  0.9455     0.8946
  1.0       0.9377         0.7530*  0.9455     0.9034

* RE-MEASURED 2026-09-04, same protocol, after `impute_once` began
standardising its design (see that function's docstring). The `single` column
was 0.8406 / 0.7927 / 0.7532 before. Only noise_sd=0.25 moved, and it moved
UP -- toward 1.0, meaning single imputation became slightly MORE honest where
the proxy is most informative. That is the expected direction: standardising
lets the proxy earn its coefficient against the same L2 penalty as every other
column instead of through one scaled by its own units, so the imputation is
better and understates its uncertainty less. `complete_case` never reaches
`impute_once` and is unaffected by construction. `mi_no_aux` and `mi_aux` were
NOT re-measured -- no assertion here reads them, and re-measuring two m=20 arms
across three noise levels costs 4800 fits -- so those two columns still carry
the 2026-09-01 figures and should be read as stale, not as confirmed.

Single imputation is more dishonest as the imputation grows less certain, and
the prototype's 0.959 / 0.995 / 0.843 was non-monotone. But the strict
three-term ordering is NOT well determined at 40 seeds, and the original claim
that this "supports a stronger claim than was pre-registered" overstated
it. Paired bootstrap over the shared seeds (B=4000), which is the right
comparison because all three levels share seeds and datasets:

  cal[0.25] - cal[0.5] = +0.0856  MCSE 0.0458  |t|=1.87  P(<=0)=0.010
  cal[0.5]  - cal[1.0] = +0.0349  MCSE 0.0404  |t|=0.86  P(<=0)=0.153
  P(the full strict ordering holds) = 0.838

The first step separates; the second does not. What this harness can support
at 40 seeds is that single imputation is less honest than complete-case, and
more so at noise_sd=1.0 than at 0.25 -- not each adjacent step in turn.

## Expectation 5 -- FALSIFIED as written

Pre-registered: under MNAR, MI "does not repair the gap ... expected to stay
comparable to complete-case".

Measured, MNAR, noise_sd=0.5, 40 seeds:

  estimator             cc gap    mi_aux gap  paired repair  repair MCSE  mi_aux |t| vs 0
  difference_in_means   -0.07566  -0.02551    +0.05014       0.00157      14.54
  lin_regression         -0.02792  -0.00985    +0.01807       0.00148      5.62

`mi_no_aux` repairs a little too: paired +0.01166 (MCSE 0.00063) and
+0.00366 (MCSE 0.00052).

Mechanism (missed by the pre-registration, real and correct): `OutcomeProxy` is built
from the *complete* outcome, so it carries information about Y exactly
where Y is missing. That makes MNAR partially ignorable given the proxy.
Imputation removes roughly two thirds of the MNAR bias and leaves a residual
that is unambiguously nonzero -- this harness asserts both halves, not just
the repair, so a "fix" that fully erased the gap would fail it exactly as a
"fix" that did nothing would.

## Expectation 6 -- HOLDS

Between-imputation variance, `lin_regression` MAR noise_sd=0.5, 40 seeds:

  rate  mean       MCSE
  0.05  1.097e-05  6.4e-08
  0.10  2.725e-05  1.36e-07
  0.25  7.459e-05  3.71e-07

`MissingnessMechanism` refuses a rate outside (0, 1), so the zero limit
cannot be measured directly; the claim is a monotone trend across these
three rates.
"""

from dataclasses import replace

import cp_causal  # noqa: F401  -- registers the estimators
import numpy as np
import pytest
from cp_causal.imputation import impute_once, multiple_impute
from cp_causal.registry import all_estimators
from cp_core.dataset import CausalDataset, complete_cases
from cp_core.values import EstimatorConfig
from cp_simulation.components import MissingnessMechanism, OutcomeProxy
from cp_simulation.scenarios import build_scenario

pytestmark = pytest.mark.slow

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)
SEEDS = 40
N_UNITS = 2000
M = 20
SCENARIO = "D"


def _simulate(mechanism: MissingnessMechanism, proxy: OutcomeProxy, seed: int):
    dgp = build_scenario(SCENARIO, missingness=mechanism, outcome_proxy=proxy)
    return dgp.generate(n_units=N_UNITS, rng=np.random.default_rng(seed))


def _complete_dataset(simulated) -> CausalDataset:
    frame = simulated.dataset.frame.copy()
    frame[simulated.dataset.schema.outcome] = simulated.truth.complete_outcome
    return CausalDataset(frame=frame, schema=simulated.dataset.schema)


def _without_auxiliary(dataset: CausalDataset) -> CausalDataset:
    if not dataset.schema.auxiliary_columns:
        return dataset
    return CausalDataset(frame=dataset.frame, schema=replace(dataset.schema, auxiliary_columns=()))


def _fit(dataset: CausalDataset, estimator_name: str, seed: int, arm: str):
    """One arm's fit: complete_case, single, mi_no_aux or mi_aux."""
    if arm == "complete_case":
        reduced, _ = complete_cases(dataset)
        return all_estimators()[estimator_name]().fit(reduced, CONFIG)
    if arm == "single":
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        filled = impute_once(dataset, rng, draw_parameters=False)
        return all_estimators()[estimator_name]().fit(filled, CONFIG)
    if arm == "mi_no_aux":
        estimate = multiple_impute(
            _without_auxiliary(dataset), estimator_name, CONFIG, m=M, seed=seed
        )
        assert estimate is not None
        return estimate
    if arm == "mi_aux":
        estimate = multiple_impute(dataset, estimator_name, CONFIG, m=M, seed=seed)
        assert estimate is not None
        return estimate
    raise ValueError(f"unknown arm {arm!r}")


def _gaps(
    estimator_name: str, mechanism: MissingnessMechanism, noise_sd: float, arms: tuple[str, ...]
) -> dict[str, np.ndarray]:
    proxy = OutcomeProxy(noise_sd=noise_sd)
    gaps: dict[str, list[float]] = {arm: [] for arm in arms}
    for seed in range(SEEDS):
        simulated = _simulate(mechanism, proxy, seed)
        target = all_estimators()[estimator_name]().fit(_complete_dataset(simulated), CONFIG).point
        for arm in arms:
            point = _fit(simulated.dataset, estimator_name, seed, arm).point
            gaps[arm].append(point - target)
    return {arm: np.asarray(values) for arm, values in gaps.items()}


MAR = MissingnessMechanism(mechanism="MAR", rate=0.25, strength=2.0)
MNAR = MissingnessMechanism(mechanism="MNAR", rate=0.25, strength=2.0)


def test_expectation_1_multiple_imputation_repairs_the_mar_gap():
    # MEASURED complete_case -0.04011 (MCSE 0.00232), 17 MCSE from zero.
    # Floor = 0.04011 - 4*0.00232 = 0.03083, rounded down to 0.03 for margin
    # (~4.36 MCSE headroom).
    #
    # MEASURED single -0.00319 (0.00182), mi_no_aux -0.00181 (0.00256),
    # mi_aux -0.00056 (0.00150) -- all within 1 MCSE of zero. Ceilings are
    # each |mean| + 4*MCSE, rounded up for real margin (~4-5 MCSE headroom):
    #   single:    0.00319 + 4*0.00182 = 0.01047 -> 0.011
    #   mi_no_aux: 0.00181 + 4*0.00256 = 0.01205 -> 0.013
    #   mi_aux:    0.00056 + 4*0.00150 = 0.00656 -> 0.007
    gaps = _gaps(
        "difference_in_means",
        MAR,
        noise_sd=0.5,
        arms=("complete_case", "single", "mi_no_aux", "mi_aux"),
    )
    assert abs(gaps["complete_case"].mean()) >= 0.03
    assert abs(gaps["single"].mean()) <= 0.011
    assert abs(gaps["mi_no_aux"].mean()) <= 0.013
    assert abs(gaps["mi_aux"].mean()) <= 0.007


def test_expectation_2_mi_without_an_auxiliary_is_no_worse_than_complete_case():
    # Expectation 2 was pre-registered as MI-without-auxiliary being ~8%
    # LESS efficient than complete-case. It measured indistinguishable
    # instead (paired ratio -0.25) -- that is FALSIFIED as written, and is
    # recorded here rather than quietly dropped.
    #
    # A floor on |paired diff| near zero would be a null-hypothesis
    # assertion: it would also pass if this arm silently did nothing at all,
    # which is exactly the failure mode this harness must have teeth
    # against. The RMSE ratio is the more honest and more stable statistic
    # for "no worse than": it is a ratio of two positive, well-estimated
    # quantities rather than a near-zero difference of two small numbers,
    # and it directly answers "how much worse, if at all" rather than only
    # "is the difference nonzero".
    #
    # MEASURED RMSE mi_no_aux 0.01485 vs complete_case 0.01509, ratio 0.984.
    # Ceiling 1.05 gives real headroom above 1.0 (a ratio > 1.05 would mean
    # mi_no_aux is genuinely, not just noisily, worse) while still catching
    # the pre-registered ~1.08 the prototype measured.
    proxy = OutcomeProxy(noise_sd=0.5)
    complete_case_se = []
    mi_no_aux_se = []
    for seed in range(SEEDS):
        simulated = _simulate(MAR, proxy, seed)
        target = (
            all_estimators()["lin_regression"]().fit(_complete_dataset(simulated), CONFIG).point
        )
        cc = _fit(simulated.dataset, "lin_regression", seed, "complete_case").point
        mi = _fit(simulated.dataset, "lin_regression", seed, "mi_no_aux").point
        complete_case_se.append((cc - target) ** 2)
        mi_no_aux_se.append((mi - target) ** 2)
    complete_case_rmse = float(np.sqrt(np.mean(complete_case_se)))
    mi_no_aux_rmse = float(np.sqrt(np.mean(mi_no_aux_se)))
    assert mi_no_aux_rmse / complete_case_rmse <= 1.05


def test_expectation_3_mi_with_an_auxiliary_beats_complete_case():
    # MEASURED paired mi_aux - complete_case, lin_regression MAR:
    #   noise_sd=0.25: -2.094e-04, MCSE 4.725e-05, ratio -4.43
    #   noise_sd=0.5:  -1.243e-04, MCSE 4.547e-05, ratio -2.73
    # Ceilings (the paired diff must stay this far below zero) are
    # mean + 3*MCSE, rounded toward zero for real margin:
    #   0.25: -2.094e-04 + 3*4.725e-05 = -6.765e-05 -> -0.00005 (~3.4 MCSE)
    #   0.5:  -1.243e-04 + 3*4.547e-05 = 1.484e-06  -> -0.00002 (~2.7 MCSE)
    # noise_sd=1.0 is deliberately excluded: at this seed count its ratio
    # would land near 1.4, below the separation this harness can assert.
    #
    # This is the test verification proved has teeth: with
    # `multiple_impute` forced to ignore the auxiliary (a schema whose
    # `auxiliary_columns` is empty), mi_aux collapses to mi_no_aux and this
    # assertion fails (paired diff near zero, not near -2e-4).
    for noise_sd, ceiling in ((0.25, -0.00005), (0.5, -0.00002)):
        proxy = OutcomeProxy(noise_sd=noise_sd)
        complete_case_se = []
        mi_aux_se = []
        for seed in range(SEEDS):
            simulated = _simulate(MAR, proxy, seed)
            target = (
                all_estimators()["lin_regression"]().fit(_complete_dataset(simulated), CONFIG).point
            )
            cc = _fit(simulated.dataset, "lin_regression", seed, "complete_case").point
            mi = _fit(simulated.dataset, "lin_regression", seed, "mi_aux").point
            complete_case_se.append((cc - target) ** 2)
            mi_aux_se.append((mi - target) ** 2)
        diff = np.asarray(mi_aux_se) - np.asarray(complete_case_se)
        assert diff.mean() <= ceiling


def test_expectation_4_single_imputation_grows_more_dishonest_as_noise_rises():
    # RE-MEASURED 2026-09-04 single-imputation calibration 0.8735 / 0.7879 /
    # 0.7530 at noise_sd 0.25 / 0.5 / 1.0, against complete_case at a constant
    # 0.9377. Was 0.8406 / 0.7927 / 0.7532 before `impute_once` began
    # standardising its design; only the 0.25 level moved, and upward.
    #
    # Both assertions below are deterministic tripwires at this fixed 40-seed
    # protocol, NOT MCSE-separated findings, and the comment they replace
    # claimed a margin that was never there. Two measurements say so:
    #
    #   - The calibration ratio's own MCSE is ~0.13 (its denominator is an SD
    #     from 40 points, ~11% relative uncertainty before the numerator's
    #     contribution). The old 0.8406 therefore sat 0.25 MCSE below its 0.87
    #     ceiling, not "about 3 percentage points" of real slack. At this seed
    #     count the ceiling cannot establish separation from complete_case's
    #     0.9377 either -- the two are within 1 MCSE of each other. It is a
    #     guard against a large regression, and should be read as nothing more.
    #   - The strict ordering is better determined than those marginals suggest,
    #     because the three levels share seeds and datasets, but still not well
    #     determined: paired bootstrap gives P(<=0) = 0.010 for the first step
    #     and 0.153 for the second, and P(the whole chain holds) = 0.838.
    #
    # The ceiling is set at 0.90: above the measured 0.8735 with room for the
    # kind of drift a refactor produces, and still below 0.9377 so a change
    # that made single imputation as honest as complete-case would fire it.
    calibrations = []
    for noise_sd in (0.25, 0.5, 1.0):
        proxy = OutcomeProxy(noise_sd=noise_sd)
        points = []
        ses = []
        for seed in range(SEEDS):
            simulated = _simulate(MAR, proxy, seed)
            fit = _fit(simulated.dataset, "lin_regression", seed, "single")
            points.append(fit.point)
            ses.append(fit.se)
        calibrations.append(float(np.mean(ses)) / float(np.std(points, ddof=1)))
    assert calibrations[0] > calibrations[1] > calibrations[2]
    assert calibrations[0] <= 0.90


def test_expectation_5_mi_repairs_most_but_not_all_of_the_mnar_gap():
    # Expectation 5 was pre-registered as MI NOT repairing the MNAR gap.
    # Measured: it repairs roughly two thirds and leaves the rest. Both
    # halves are asserted, so neither a "fix" that erased the whole gap nor
    # one that did nothing would pass this test.
    #
    # MEASURED (MNAR, noise_sd=0.5, difference_in_means): complete_case
    # -0.07566, mi_aux -0.02551, paired repair +0.05014 (MCSE 0.00157,
    # |t|=14.54 vs zero). Repair floor = 0.05014 - 4*0.00157 = 0.04386,
    # rounded down to 0.04 (~6.5 MCSE headroom). Residual floor on |mi_aux
    # gap| = 0.02551 - 6*0.00157*... kept generous: 0.015 (comfortably below
    # the measured 0.02551, ~6 MCSE of margin using the repair MCSE as a
    # stand-in scale). Ratio band brackets "roughly two thirds":
    # 0.05014/0.07566 = 0.663.
    for estimator_name, cc_floor, repair_floor, residual_floor, ratio_bounds in (
        ("difference_in_means", 0.06, 0.04, 0.015, (0.45, 0.85)),
        ("lin_regression", 0.02, 0.012, 0.005, (0.40, 0.90)),
    ):
        gaps = _gaps(estimator_name, MNAR, noise_sd=0.5, arms=("complete_case", "mi_aux"))
        cc_gap = gaps["complete_case"].mean()
        mi_gap = gaps["mi_aux"].mean()
        repair = mi_gap - cc_gap
        assert abs(cc_gap) >= cc_floor
        assert repair >= repair_floor
        assert abs(mi_gap) >= residual_floor
        low, high = ratio_bounds
        assert low <= repair / abs(cc_gap) <= high


def test_expectation_6_between_imputation_variance_rises_with_the_missing_rate():
    # MEASURED lin_regression, MAR noise_sd=0.5: means 1.097e-05 / 2.725e-05
    # / 7.459e-05 at rates 0.05 / 0.10 / 0.25, MCSEs ~0.5% of each mean --
    # generous rounding gives real headroom without needing MCSE arithmetic:
    #   rate 0.05 ceiling 1.3e-05 (measured 1.097e-05)
    #   rate 0.10 floor   2.0e-05 (measured 2.725e-05)
    #   rate 0.25 floor   7.0e-05 (measured 7.459e-05)
    proxy = OutcomeProxy(noise_sd=0.5)
    means = {}
    for rate in (0.05, 0.10, 0.25):
        mechanism = MissingnessMechanism(mechanism="MAR", rate=rate, strength=2.0)
        values = []
        for seed in range(SEEDS):
            simulated = _simulate(mechanism, proxy, seed)
            estimate = multiple_impute(simulated.dataset, "lin_regression", CONFIG, m=M, seed=seed)
            assert estimate is not None
            values.append(estimate.diagnostics["between_variance"].value)
        means[rate] = float(np.mean(values))
    assert means[0.05] <= 1.3e-05
    assert means[0.10] >= 2.0e-05
    assert means[0.25] >= 7.0e-05
    assert means[0.05] < means[0.10] < means[0.25]
