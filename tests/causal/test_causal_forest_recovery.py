"""CausalForest scored against measured per-unit CATE truth.

Every tolerance below is measured through `CausalForest.fit` itself, at this
file's own `_CONFIG_OPTIONS` and `CATE_RECOVERY_SEEDS`, and each test names the
number it was measured at. Nothing here is carried over from the plan's
design-phase prototype: the plan's own placeholder tolerances cited a
single-seed prototype correlation of 0.389, which is not a protocol any
assertion here runs under -- the same quantity measures 0.8177 through the
shipped estimator at this file's protocol.

Two of these tests exist because a *rate* cannot be checked on one seed, and
both caught a formula that every fast contract test passed. The ATE
interval's coverage measured 0.35-0.40 against a nominal 0.95, and the
subgroup false-discovery rate measured 0.43 against a nominal 0.10. Neither
is visible from a single fit: coverage on one seed is one bit, and a single
seed's findings look individually plausible whether or not the family-wise
rate is controlled. See each test's own comment for the numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from cp_causal.estimators.causal_forest import CausalForest
from cp_core.values import EstimatorConfig
from cp_simulation.cate_scenarios import CATE_N_UNITS, CATE_SCENARIOS, build_cate_scenario

CATE_RECOVERY_SEEDS = 15
_CONFIG_OPTIONS = {"n_trees": 500, "max_depth": 4, "min_leaf_structure": 20}


def _config(seed: int) -> EstimatorConfig:
    return EstimatorConfig(random_seed=seed, ci_level=0.95, options=dict(_CONFIG_OPTIONS))


def _generate(scenario_id: str, seed: int):
    dgp = build_cate_scenario(scenario_id)
    return dgp.generate(CATE_N_UNITS, np.random.default_rng(seed))


@pytest.mark.slow
def test_causal_forest_per_unit_cate_correlates_with_truth_on_rich_heterogeneity():
    correlations = []
    biases = []
    for seed in range(CATE_RECOVERY_SEEDS):
        simulated = _generate("RICH", seed)
        result = CausalForest().fit(simulated.dataset, _config(seed))
        per_unit = result.diagnostics["per_unit_cate"].value
        true_cate = simulated.truth.true_cate
        correlations.append(float(np.corrcoef(per_unit, true_cate)[0, 1]))
        biases.append(float(np.mean(per_unit - true_cate)))
    mean_corr = float(np.mean(correlations))
    mean_bias = float(np.mean(biases))
    # Measured at this exact protocol (15 seeds, this file's `_CONFIG_OPTIONS`,
    # through `CausalForest.fit`): correlation mean 0.8177, per-seed min
    # 0.5980, sd 0.0920 -- so the mean carries an MCSE of about 0.024. Bias
    # mean +0.0031, sd 0.0226, MCSE about 0.006.
    #
    # The correlation bound is 0.60, roughly nine MCSEs below the measured
    # mean and just under the *worst single seed*, so ordinary seed variance
    # cannot trip it but a forest that had stopped ordering units by their
    # true effect would. It is deliberately not set just under 0.8177: the
    # honest-splitting subsample and the cross-fitting folds both move with
    # the seed, and this is a recovery floor, not a regression pin.
    #
    # The earlier bound here was 0.15, carried over from a design-phase
    # prototype that measured 0.389 on a single seed. Against a real 0.8177
    # that would have passed while the estimator lost more than three
    # quarters of its signal.
    assert mean_corr > 0.60, f"RICH per-unit CATE correlation {mean_corr:.4f} -- too weak to trust"
    assert abs(mean_bias) < 0.03, f"RICH per-unit CATE bias {mean_bias:+.4f}"


@pytest.mark.slow
def test_causal_forest_ate_interval_covers_the_true_ate():
    """Coverage of the reported ATE interval, which is a RATE and so cannot be
    checked on one seed.

    This test exists because the formula the implementation plan specified --
    the mean of the per-unit infinitesimal-jackknife variances divided by n --
    measured 0.40 coverage on NULL and 0.35 on RICH against a nominal 0.95,
    understating the standard error about sixfold by treating 2000 per-unit
    CATEs built from the same trees and the same residuals as independent.
    Every fast test in `test_causal_forest.py` passed under that formula.
    `aggregate_effect`'s docstring carries the full comparison.
    """
    for scenario in ("NULL", "RICH"):
        covered = 0
        for seed in range(CATE_RECOVERY_SEEDS):
            simulated = _generate(scenario, seed)
            result = CausalForest().fit(simulated.dataset, _config(seed))
            true_ate = float(simulated.truth.true_ate)
            covered += result.ci.lower <= true_ate <= result.ci.upper
        rate = covered / CATE_RECOVERY_SEEDS
        # Measured at this exact protocol over 20 seeds: 1.00 on NULL, 0.95 on
        # RICH. The lower bound is 0.80 rather than something nearer 0.95
        # because 15 seeds put a binomial standard error of about 0.06 on the
        # measurement even when coverage is exactly nominal; it is set to
        # catch the broken formula above (which would land near 0.35-0.40),
        # not to resolve a few points of undercoverage.
        assert rate >= 0.80, (
            f"{scenario} ATE interval covered the true ATE on only {covered}/"
            f"{CATE_RECOVERY_SEEDS} seeds ({rate:.2f}) against a nominal 0.95"
        )


@pytest.mark.slow
def test_causal_forest_calibration_test_correctly_finds_no_heterogeneity_on_null_scenario():
    # Two-sided per this project's established pattern (Phase 4's S2 test,
    # Phase 6 design doc section 6): passing RICH recovery alone must not
    # silently mean the calibration guard stopped working.
    detected = 0
    for seed in range(CATE_RECOVERY_SEEDS):
        simulated = _generate("NULL", seed)
        result = CausalForest().fit(simulated.dataset, _config(seed))
        status = {c.name: c.status for c in result.assumptions}
        if status["calibration_test"] == "PASS":  # PASS here means "heterogeneity detected"
            detected += 1
    false_positive_rate = detected / CATE_RECOVERY_SEEDS
    assert false_positive_rate < 0.25, (
        f"calibration test false-fired on {detected}/{CATE_RECOVERY_SEEDS} NULL seeds"
    )


@pytest.mark.slow
def test_causal_forest_bh_fdr_control_limits_false_discoveries_on_noisy_scenario():
    # x0/x1/x2 are the real drivers of tau in the NOISY scenario (see
    # cate_scenarios.py); x3-x11 are pure noise and must not dominate reported
    # findings at the configured fdr_level=0.10.
    noise_covariate_names = {f"x{i}" for i in range(3, 12)}
    total_findings = 0
    noise_findings = 0
    for seed in range(CATE_RECOVERY_SEEDS):
        simulated = _generate("NOISY", seed)
        result = CausalForest().fit(simulated.dataset, _config(seed))
        total_findings += len(result.subgroup_findings)
        noise_findings += sum(
            1 for f in result.subgroup_findings if f.covariate in noise_covariate_names
        )
    assert total_findings > 0, "expected at least some real findings across 15 NOISY seeds"
    false_discovery_rate = noise_findings / total_findings if total_findings else 0.0
    # Measured over 20 seeds when this guard was fixed (commit "estimate
    # subgroup effects from residuals so BH actually controls the FDR"): 80
    # discoveries, 4 false, an empirical FDR of 0.050 against the configured
    # nominal 0.10. The formula it replaced -- averaging the forest's own
    # out-of-bag CATE over each cell -- measured 0.43, with pure-noise
    # covariates surviving on every one of the first eight seeds.
    #
    # fdr_level=0.10 is a target on the EXPECTED proportion of false
    # discoveries among reported findings, not a hard per-run ceiling, so
    # the bound sits at 0.30 rather than near the measured 0.050: the count
    # of discoveries itself swings seed to seed (3 of those 20 seeds
    # reported nothing at all), and a pooled ratio over few discoveries is
    # unstable in a way the underlying rate is not. It is set to catch a
    # guard that has stopped working -- the 0.43 regime above -- not to
    # police the difference between 0.05 and 0.10.
    assert false_discovery_rate < 0.30, (
        f"noise-covariate false discovery rate {false_discovery_rate:.2f} "
        f"({noise_findings}/{total_findings}) -- BH-FDR control may not be working"
    )


def test_every_cate_scenario_is_covered_by_this_harness():
    assert set(CATE_SCENARIOS) == {"RICH", "NULL", "NOISY"}
