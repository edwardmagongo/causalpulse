"""HierarchicalDiD scored against measured panel ground truth.

Every tolerance below was measured through `HierarchicalDiD.fit` itself, at
this file's own `_config` and seed counts, and each assertion names the number
it was measured at together with its Monte Carlo standard error. Nothing is
carried over from a prototype or from the implementation plan.

Two things this harness is deliberately built to notice.

**Coverage is a rate.** One seed yields one bit, so an interval that covers
truth on a single seed says nothing about whether it covers at its nominal
level. Every coverage number here is a fraction over seeds.

**The estimator was wrong here once, and the fast tests all passed.** Before
the model gained its exposure term, the effect was constant within cohort.
Against a ramping truth that misfit correlates with calendar period, `gamma_t`
absorbed it, and cohorts treated only in late periods lost most of their
effect: P4 measured -17% bias with 0/15 coverage while R-hat and divergences
passed on every seed. Only a multi-seed harness scoring against measured truth
could see it. That is what this file is for.
"""

from __future__ import annotations

import numpy as np
import pytest
from cp_causal.estimators.hierarchical_did import (
    HIER_PRODUCTION_MCMC_CONFIG,
    HierarchicalDiD,
)
from cp_core.values import EstimatorConfig
from cp_simulation.panel_scenarios import (
    HIER_N_PERIODS,
    HIER_N_UNITS,
    PANEL_N_PERIODS,
    PANEL_N_UNITS,
    PANEL_SCENARIOS,
    build_panel_scenario,
)

# This harness's own seed counts, deliberately distinct from
# `panel_scenarios.HIER_SEEDS` (which is P7's ground-truth measurement
# protocol) and from `PANEL_SEEDS` (Phase 3's). They answer different
# questions and are free to move independently.
RECOVERY_SEEDS = 8
# P7 costs roughly 5 minutes per fit against P1-P5's ~2, so it is scored on
# fewer seeds. Its MCSE below is computed at this count, not borrowed.
P7_SEEDS = 6


def _config(seed: int) -> EstimatorConfig:
    return EstimatorConfig(
        random_seed=seed,
        ci_level=0.95,
        options={"mcmc_config": HIER_PRODUCTION_MCMC_CONFIG},
    )


def _generate(scenario_id: str, seed: int):
    """P7 carries its own longer protocol; every other scenario uses the
    repository-wide panel protocol its Phase 3 measurements were taken at."""
    dgp = build_panel_scenario(scenario_id)
    if scenario_id == "P7":
        return dgp.generate(HIER_N_UNITS, HIER_N_PERIODS, np.random.default_rng(seed))
    return dgp.generate(PANEL_N_UNITS, PANEL_N_PERIODS, np.random.default_rng(seed))


class _Scored:
    """One scenario's fits, scored once and shared by every test that reads it.

    Each fit is an MCMC run costing minutes, so re-fitting per assertion would
    multiply this file's runtime by the number of tests rather than the number
    of seeds.
    """

    def __init__(self, scenario_id: str, seeds: int) -> None:
        self.scenario_id = scenario_id
        self.n = seeds
        self.bias: list[float] = []
        self.covered = 0
        self.status: list[dict[str, str]] = []

        for seed in range(seeds):
            simulated = _generate(scenario_id, seed)
            result = HierarchicalDiD().fit(simulated.dataset, _config(seed))
            truth = float(simulated.truth.true_att_overall)
            self.bias.append(result.point - truth)
            self.covered += result.ci.lower <= truth <= result.ci.upper
            self.status.append({c.name: c.status for c in result.assumptions})

    @property
    def mean_bias(self) -> float:
        return float(np.mean(self.bias))

    @property
    def mcse(self) -> float:
        return float(np.std(self.bias, ddof=1) / np.sqrt(self.n))

    @property
    def coverage(self) -> float:
        return self.covered / self.n

    def failing(self, check: str) -> int:
        return sum(1 for s in self.status if s.get(check) == "FAIL")


@pytest.fixture(scope="module")
def p1() -> _Scored:
    return _Scored("P1", RECOVERY_SEEDS)


@pytest.fixture(scope="module")
def p2() -> _Scored:
    return _Scored("P2", RECOVERY_SEEDS)


@pytest.fixture(scope="module")
def p3() -> _Scored:
    return _Scored("P3", RECOVERY_SEEDS)


@pytest.fixture(scope="module")
def p4() -> _Scored:
    return _Scored("P4", RECOVERY_SEEDS)


@pytest.fixture(scope="module")
def p5() -> _Scored:
    return _Scored("P5", RECOVERY_SEEDS)


@pytest.fixture(scope="module")
def p7() -> _Scored:
    return _Scored("P7", P7_SEEDS)


# Measured through `HierarchicalDiD.fit` at this file's own protocol. Bias is
# on the outcome's scale; MCSE is the Monte Carlo standard error of the mean.
#
#   scenario  mean bias   relative      MCSE   coverage   r_hat FAIL
#   P1         -0.0430     -4.30%     0.0307      6/8         0/8
#   P2         -0.0155     (null)     0.0316      6/8         1/8
#   P3         -0.0067     -0.67%     0.0268      7/8         1/8
#   P4         -0.0075     -0.27%     0.0272      7/8         2/8
#   P5         +2.1359   +213.59%     0.0538      0/8         1/8   <- must be biased
#   P7         +0.0123     +0.18%     0.0120      6/6         3/6
#
# Every bias bound below is 0.15. That is not a round number chosen for
# comfort: it sits 4-11 MCSE above each recovering scenario's measured bias,
# and cleanly below what the pre-exposure-term model measured on the two
# scenarios that exposed its defect (P4 -0.4819, P7 -1.3885). A regression to
# that model fails these tests; ordinary seed variance does not.
_BIAS_BOUND = 0.15


@pytest.mark.slow
def test_p1_recovers_truth_on_the_clean_two_by_two(p1: _Scored):
    assert abs(p1.mean_bias) < _BIAS_BOUND, (
        f"P1 mean bias {p1.mean_bias:+.4f} (MCSE {p1.mcse:.4f}) over {p1.n} seeds"
    )


@pytest.mark.slow
def test_p2_null_scenario_produces_no_systematic_effect(p2: _Scored):
    """The null's truth is exactly zero, so mean bias IS the estimate.

    A systematic finding here would mean the estimator invents an effect
    where none exists -- the falsification direction that recovery on the
    effect scenarios cannot test.
    """
    assert abs(p2.mean_bias) < _BIAS_BOUND, (
        f"P2 mean estimate {p2.mean_bias:+.4f} (MCSE {p2.mcse:.4f}) against a true zero"
    )


@pytest.mark.slow
def test_p3_recovers_truth_under_homogeneous_staggered_adoption(p3: _Scored):
    assert abs(p3.mean_bias) < _BIAS_BOUND, (
        f"P3 mean bias {p3.mean_bias:+.4f} (MCSE {p3.mcse:.4f}) over {p3.n} seeds"
    )


@pytest.mark.slow
def test_p4_recovers_truth_when_effects_ramp_with_exposure(p4: _Scored):
    """The scenario that separates this estimator from a biased TWFE, and the
    one that caught the model's own missing exposure term.

    Two-way fixed effects is badly biased on P4 by construction. Before the
    model let effects grow with exposure, it was too: -0.4819 mean bias with
    0/15 coverage, R-hat clean on every seed. Measured here at -0.0075
    (MCSE 0.0272), which is indistinguishable from zero.
    """
    assert abs(p4.mean_bias) < _BIAS_BOUND, (
        f"P4 mean bias {p4.mean_bias:+.4f} (MCSE {p4.mcse:.4f}) over {p4.n} seeds -- "
        f"the pre-exposure-term model measured -0.4819 here"
    )


@pytest.mark.slow
def test_p7_recovers_truth_with_many_cohorts(p7: _Scored):
    """Seven cohorts, the scenario built so the across-cohort variance is
    identifiable at all. The pre-exposure-term model measured -1.3885 here
    (-20.6%) with 0/15 coverage; measured now at +0.0123 (MCSE 0.0120).
    """
    assert abs(p7.mean_bias) < _BIAS_BOUND, (
        f"P7 mean bias {p7.mean_bias:+.4f} (MCSE {p7.mcse:.4f}) over {p7.n} seeds -- "
        f"the pre-exposure-term model measured -1.3885 here"
    )


@pytest.mark.slow
def test_p5_is_biased_and_its_parallel_trends_check_says_so(p5: _Scored):
    """P5 violates parallel trends by construction, so the estimator SHOULD be
    badly biased here -- every DiD estimator in this repository is,
    Callaway-Sant'Anna included, because they all assume it.

    What must not happen is being biased quietly. Both halves are asserted:
    the bias is real (+2.1359, over 200% of truth), and `parallel_trends`
    FAILs on 8 of 8 seeds. Asserting only recovery elsewhere would let this
    guard rot unnoticed.
    """
    assert abs(p5.mean_bias) > 1.0, (
        f"P5 mean bias {p5.mean_bias:+.4f} -- expected a large bias on a scenario "
        f"that violates parallel trends; a small one means the scenario stopped "
        f"violating it"
    )
    failing = p5.failing("parallel_trends")
    assert failing >= 7, (
        f"parallel_trends FAILed on only {failing}/{p5.n} P5 seeds (measured 8/8) -- "
        f"the check that is supposed to catch this violation has weakened"
    )


@pytest.mark.slow
def test_credible_intervals_cover_truth_at_a_rate_far_above_the_broken_model(
    p1: _Scored, p2: _Scored, p3: _Scored, p4: _Scored, p7: _Scored
):
    """Coverage is a RATE. One seed is one bit and says nothing.

    Measured per scenario: 6/8, 6/8, 7/8, 7/8, 6/6 -- pooled 32/38 = 0.842.

    **This is below the nominal 0.95, and the shortfall is larger than seed
    noise explains**: if true coverage were 0.95, 38 fits carry a standard
    error of 0.035, putting the observed 0.842 about 3 standard errors low.
    Recorded here as a measured property rather than smoothed over. A
    posterior credible interval carries no frequentist coverage guarantee, so
    this is not a contradiction, but it does mean these intervals should not
    be read as 95% frequentist confidence intervals -- which is also why
    `ci_basis` is `"posterior"` and never described as a confidence interval.

    The floor is 0.5, chosen to catch collapse rather than to police the gap
    between 0.84 and 0.95: the pre-exposure-term model measured 0/15 on both
    P4 and P7, and any regression of that kind fails here immediately. At 8
    seeds a tighter floor would trip on ordinary variance -- a scenario at
    true 0.95 coverage still shows 6/8 about 6% of the time.
    """
    scored = [p1, p2, p3, p4, p7]
    for s in scored:
        assert s.coverage >= 0.5, (
            f"{s.scenario_id} covered truth on {s.covered}/{s.n} seeds "
            f"({s.coverage:.3f}) -- the pre-exposure-term model measured 0.000"
        )
    pooled = sum(s.covered for s in scored) / sum(s.n for s in scored)
    assert pooled >= 0.6, f"pooled coverage {pooled:.3f} across {len(scored)} scenarios"


@pytest.mark.slow
def test_mcmc_diagnostics_are_reported_per_seed_not_assumed_clean(
    p4: _Scored, p7: _Scored
):
    """The production config does not converge cleanly on every seed, and the
    harness says so rather than assuming otherwise.

    Measured R-hat FAILs: 2/8 on P4, 3/6 on P7. This is the seed-sensitive
    funnel `HIER_PRODUCTION_MCMC_CONFIG`'s own comment warns about -- at 1000
    warmup a P3 seed produced a genuinely stuck chain (R-hat 5.50), which is
    why warmup is 1500. What survives at 1500 is mild overshoot, not stuck
    chains, and bias stays tight regardless of which seeds trip it (P7's
    bias sd is 0.0293 across seeds where half fail R-hat).

    So this is a diagnostic-cleanliness property, not a bias one, and it is
    asserted as a rate. The bound is deliberately loose: it exists to catch
    every seed failing, which would mean the sampler had stopped working.
    """
    assert p4.n - p4.failing("r_hat") >= 4, (
        f"P4 R-hat passed on only {p4.n - p4.failing('r_hat')}/{p4.n} seeds (measured 6/8)"
    )
    assert p7.n - p7.failing("r_hat") >= 2, (
        f"P7 R-hat passed on only {p7.n - p7.failing('r_hat')}/{p7.n} seeds (measured 3/6)"
    )


def _compare(scenario_id: str, other, seeds: int) -> dict[str, float]:
    """Fit HierarchicalDiD and one other estimator on identical panels."""
    diffs: list[float] = []
    inside = 0
    overlap = 0
    for seed in range(seeds):
        simulated = _generate(scenario_id, seed)
        hier = HierarchicalDiD().fit(simulated.dataset, _config(seed))
        rival = other().fit(
            simulated.dataset, EstimatorConfig(random_seed=seed, ci_level=0.95)
        )
        diffs.append(hier.point - rival.point)
        inside += hier.ci.lower <= rival.point <= hier.ci.upper
        overlap += (hier.ci.lower <= rival.ci.upper) and (rival.ci.lower <= hier.ci.upper)
    return {
        "abs_max_diff": float(np.abs(diffs).max()),
        "mean_diff": float(np.mean(diffs)),
        "inside": inside,
        "overlap": overlap,
        "n": seeds,
    }


@pytest.mark.slow
def test_single_cohort_fit_collapses_onto_two_way_fixed_effects():
    """With one adoption cohort the hierarchy has nothing to pool over, so
    the estimate must reduce to the ordinary two-way fixed-effects answer.

    P1 is exactly that case -- uniform adoption, a single cohort. This is the
    cheapest real check available and it uses only code that already exists:
    it catches cohort-indexing errors, weighting errors and sign errors that
    a dispersed-cohort recovery test can absorb into partial pooling.

    Measured over 8 seeds: mean difference +0.0093, sd 0.0121, largest
    absolute difference 0.0277, and TWFE's point estimate falls inside the
    posterior credible interval on 8 of 8 seeds. Both estimators carry
    essentially the same bias here (-0.0430 and -0.0523), which is what
    "collapses onto" should mean.

    The bound is 0.10, about 8 sd above the measured mean difference and
    roughly 4x the worst seed, so a genuine divergence between the two
    parameterisations fails while MCMC noise does not.
    """
    from cp_causal.estimators.did import TwoWayFixedEffects

    m = _compare("P1", TwoWayFixedEffects, RECOVERY_SEEDS)
    assert m["abs_max_diff"] < 0.10, (
        f"P1 hierarchical vs TWFE: largest difference {m['abs_max_diff']:.4f} "
        f"(measured 0.0277 over 8 seeds) -- with one cohort these must agree"
    )
    assert m["inside"] >= 7, (
        f"TWFE's point fell inside the credible interval on only "
        f"{m['inside']}/{m['n']} seeds (measured 8/8)"
    )


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["P3", "P4"])
def test_agrees_with_callaway_santanna_on_staggered_adoption(scenario: str):
    """Two genuinely different methods targeting the same estimand.

    P3 (homogeneous) and P4 (effects ramping with exposure) are the scenarios
    where Callaway-Sant'Anna recovers truth and two-way fixed effects does
    not, so agreement here is evidence this estimator belongs in the former
    group. Checked as interval overlap and rough magnitude, NOT exact match:
    these solve the same statistical problem by different means -- posterior
    partial pooling against a group-time ATT aggregation -- the way
    CausalForest and econml do, not the same optimisation the way
    SyntheticControl and pysyncon do.

    Measured over 8 seeds per scenario: largest absolute difference 0.1902
    (P3) and 0.1954 (P4); intervals overlap on 8 of 8 seeds in both;
    Callaway-Sant'Anna's point falls inside the credible interval on 7 of 8.
    Mean bias against measured truth: -0.0067 (this estimator) versus +0.0471
    (Callaway-Sant'Anna) on P3, and -0.0075 versus +0.0471 on P4.

    An aside worth recording, since it looks like a bug and is not:
    Callaway-Sant'Anna's per-seed ERROR is identical on P3 and P4
    (-0.0811, +0.0200, +0.1362, +0.1526, ...). The two scenarios share units,
    noise and adoption timing at a given seed and differ only in the
    deterministic effect added on top; a linear estimator's error does not
    depend on that, so the errors coincide exactly. This estimator's do not,
    because MCMC is stochastic.

    The bound is 0.35, roughly 1.8x the worst measured seed.
    """
    from cp_causal.estimators.callaway_santanna import CallawaySantAnna

    m = _compare(scenario, CallawaySantAnna, RECOVERY_SEEDS)
    assert m["abs_max_diff"] < 0.35, (
        f"{scenario} hierarchical vs Callaway-Sant'Anna: largest difference "
        f"{m['abs_max_diff']:.4f} (measured 0.19)"
    )
    assert m["overlap"] >= 7, (
        f"{scenario}: intervals overlapped on only {m['overlap']}/{m['n']} seeds "
        f"(measured 8/8) -- two methods targeting one estimand should not be disjoint"
    )


def test_every_scenario_this_harness_scores_is_a_registered_panel_scenario():
    # A scenario renamed or removed upstream should break loudly here rather
    # than silently drop out of this file's coverage.
    assert {"P1", "P2", "P3", "P4", "P5", "P7"} <= set(PANEL_SCENARIOS)
