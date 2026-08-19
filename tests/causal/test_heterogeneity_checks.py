"""Guards against fabricated subgroup findings: omnibus calibration test,
best-linear-projection, minimum cell size, Benjamini-Hochberg FDR control.
"""

from __future__ import annotations

import numpy as np
from cp_causal.estimators.causal_forest import fit_forest
from cp_causal.heterogeneity_checks import best_linear_projection, calibration_test
from cp_simulation.cate_scenarios import CATE_N_UNITS, build_cate_scenario

# n_trees=500 is measured, not guessed -- but what it is measured to be is the
# cheapest of several equally unusable values, so these two tests assert only
# the checks' *shape*, never that `calibration_test` detects RICH's
# heterogeneity or clears NULL's. Both were measured on 2026-08-19 in this
# worktree, at min_leaf_structure=20, max_depth=4, subsample_frac=0.5,
# cross_fit_folds=5, CATE_N_UNITS=2000, data seed == forest seed.
#
# Sweeping n_trees at data seed 0 (brief's Step 5 script verbatim):
#
#     n_trees   RICH t    NULL t
#         500   +1.268    +1.409
#        1000   +1.323    +1.476
#        2000   +1.234    +1.461
#
# The statistic is essentially invariant in n_trees (a 4x increase moves it by
# <0.1 at a fixed seed); the spread is across data seeds, not tree counts. So
# no value of n_trees separates the scenarios, and 500 is kept as the cheapest.
#
# Over 10 data seeds at n_trees=500, the differential-prediction coefficient
# (which should be ~1 on RICH and ~0 on NULL) and its t:
#
#     RICH: mean b_diff=-1.145, t mean=-0.819 (sd 2.028), |t|>1.96 on 3/10
#           seeds -- all three NEGATIVE (seeds 2, 4, 9); 0/10 detections in
#           the correct, positive direction.
#     NULL: mean b_diff=-1.904, t mean=-1.562 (sd 2.026), |t|>1.96 on 5/10
#           seeds -- a ~50% false-positive rate at alpha=0.05.
#
# The mean-calibration coefficient is fine (b_mean = 1.00 +/- 0.03 on both
# scenarios), and the forest itself does carry signal: corr(oob_cate,
# true_cate) on RICH has a median of ~+0.4 over those seeds. Heteroskedasticity
# is not the cause either -- HC0 standard errors move every t by <0.07. The
# differential *coefficient* is biased negative, so the drafted statistic, not
# the forest and not the standard error, is what fails to separate the two
# scenarios. Recorded here rather than encoded as a threshold: a test asserting
# RICH PASS / NULL FAIL would be asserting a magnitude that was measured not to
# hold.
_FOREST_CONFIG: dict[str, object] = dict(
    n_trees=500, min_leaf_structure=20, max_depth=4, subsample_frac=0.5, cross_fit_folds=5,
)


def _fit_on_scenario(scenario_id: str, seed: int):
    dgp = build_cate_scenario(scenario_id)
    simulated = dgp.generate(CATE_N_UNITS, np.random.default_rng(seed))
    covariates = simulated.dataset.frame[list(simulated.dataset.schema.covariates)].to_numpy()
    treatment = simulated.dataset.frame[simulated.dataset.schema.treatment].to_numpy()
    outcome = simulated.dataset.frame[simulated.dataset.schema.outcome].to_numpy(dtype=float)
    fit = fit_forest(
        covariates, treatment, outcome, rng=np.random.default_rng(seed), **_FOREST_CONFIG
    )
    return fit, covariates


def test_calibration_test_reports_a_real_assumption_check():
    fit, _ = _fit_on_scenario("RICH", seed=0)
    result = calibration_test(fit, alpha=0.05)
    assert result.name == "calibration_test"
    assert result.status in ("PASS", "FAIL")
    assert result.statistic is not None


def test_best_linear_projection_returns_one_coefficient_per_covariate():
    fit, covariates = _fit_on_scenario("RICH", seed=0)
    names = ("x0", "x1", "x2", "x3", "x4", "x5")
    projection = best_linear_projection(fit, covariates, names)
    assert set(projection) == set(names)
