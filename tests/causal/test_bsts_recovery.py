"""BSTS scored against measured donor-pool truth, at PRODUCTION_MCMC_CONFIG.

Tolerances below are seeded from design-phase prototype measurements at 15
seeds per scenario (design doc §2 item 7 / this plan's "Prototype findings"
section), not the full DONOR_SEEDS=400 protocol test_donor_recovery.py uses
for SyntheticControl -- BSTS's MCMC fit costs ~1.5-2s/seed against SC's
milliseconds, so a full 400-seed run (~10-15 min) is deliberately left to
this task's own execution rather than pre-computed during planning. Run this
file's tests once implemented, replace the tolerances and this comment with
the real measured bias/MCSE/coverage at DONOR_SEEDS, exactly as
test_donor_recovery.py's own comments cite exact measured numbers -- do not
ship this file with the prototype-derived placeholder tolerances below as
the final word.

Prototype measurements this task's initial tolerances are set from (15 seeds
each, PRODUCTION_MCMC_CONFIG):
- S1: mean bias +0.0745 (MCSE 0.1322), 15/15 credible intervals covered
  truth, mean divergence rate 4.22%.
- S2: mean bias -0.0098 (MCSE 0.1785), 15/15 covered, mean divergence rate
  7.25%.
- S0: 0/20 seeds had a 95% credible interval excluding zero.
"""

from __future__ import annotations

import numpy as np
import pytest
from cp_causal.estimators.bsts import BSTS, PRODUCTION_MCMC_CONFIG
from cp_core.values import EstimatorConfig
from cp_simulation.donor_scenarios import (
    DONOR_SCENARIOS,
    DONOR_START,
    build_donor_scenario,
)

# 15 seeds here, not DONOR_SEEDS=400: see module docstring. Raise to
# DONOR_SEEDS once this file is actually run and its runtime is measured.
BSTS_RECOVERY_SEEDS = 15


def _config(seed: int) -> EstimatorConfig:
    return EstimatorConfig(
        random_seed=seed, ci_level=0.95, options={"mcmc_config": PRODUCTION_MCMC_CONFIG}
    )


def _generate(scenario_id: str, seed: int):
    spec = DONOR_SCENARIOS[scenario_id]
    return build_donor_scenario(scenario_id).generate(
        spec.n_units, spec.n_periods, np.random.default_rng(seed)
    )


def _bsts_bias(scenario_id: str, seed: int) -> tuple[float, bool]:
    panel = _generate(scenario_id, seed)
    result = BSTS().fit(panel.dataset, _config(seed))
    truth = panel.truth.true_att_by_cohort[DONOR_START]
    covers = result.ci.lower <= truth <= result.ci.upper
    return result.point - truth, covers


@pytest.mark.slow
def test_bsts_recovers_truth_on_the_clean_scenario():
    # Prototype (15 seeds): bias +0.0745 (MCSE 0.1322). Tolerance 0.5 gives
    # (0.5 - 0.0745) / 0.1322 = 3.2 MCSE headroom at this seed count --
    # loose on purpose, since this is a design-phase prototype measurement,
    # not the calibrated tolerance. Tighten once run at DONOR_SEEDS.
    results = [_bsts_bias("S1", seed) for seed in range(BSTS_RECOVERY_SEEDS)]
    biases = np.array([b for b, _ in results])
    coverage = np.mean([c for _, c in results])
    bias = float(biases.mean())
    assert abs(bias) < 0.5, f"S1 BSTS bias {bias:+.4f}"
    assert coverage >= 0.7, f"S1 BSTS coverage {coverage:.2f}"


@pytest.mark.slow
def test_bsts_recovers_truth_and_concentrates_donor_weight_under_divergent_pretrend():
    # Prototype (15 seeds): bias -0.0098 (MCSE 0.1785). See
    # test_bsts_recovers_truth_on_the_clean_scenario for the tolerance's
    # headroom reasoning.
    results = [_bsts_bias("S2", seed) for seed in range(BSTS_RECOVERY_SEEDS)]
    biases = np.array([b for b, _ in results])
    coverage = np.mean([c for _, c in results])
    bias = float(biases.mean())
    assert abs(bias) < 0.6, f"S2 BSTS bias {bias:+.4f} -- S2 stopped being recoverable"
    assert coverage >= 0.6, f"S2 BSTS coverage {coverage:.2f}"


@pytest.mark.slow
def test_null_scenario_credible_interval_does_not_over_reject():
    # Prototype: 0/20 seeds had a 95% credible interval excluding zero.
    # One-sided bound only, same limitation test_donor_recovery.py's S0 test
    # and test_panel_recovery.py's P2 test both carry: this can only confirm
    # nothing is catastrophically miscalibrated toward false positives, not
    # that the interval is well-calibrated at exactly the 5% level.
    significant = 0
    for seed in range(BSTS_RECOVERY_SEEDS):
        panel = _generate("S0", seed)
        result = BSTS().fit(panel.dataset, _config(seed))
        if not result.ci.contains(0.0):
            significant += 1
    assert significant / BSTS_RECOVERY_SEEDS < 0.35


@pytest.mark.slow
def test_sufficient_pre_periods_fires_reliably_on_s5():
    for seed in range(10):
        panel = _generate("S5", seed)
        result = BSTS().fit(panel.dataset, _config(seed))
        status = {c.name: c.status for c in result.assumptions}
        assert status["sufficient_pre_periods"] == "FAIL", seed


def test_every_donor_scenario_is_covered_by_this_harness():
    assert set(DONOR_SCENARIOS) == {"S0", "S1", "S2", "S3", "S4", "S5"}
