"""Shared placebo-lead machinery in `panel_checks.py`.

`HierarchicalDiD` and `CallawaySantAnna` both need pre-treatment placebo
leads for their `parallel_trends_check`, and until this refactor each
estimator carried its own private copy of the exact same 2x2/variance
formula (`_unit_means` + `_group_change` + the placebo loop). That formula
has already been corrected once in this repository's history -- a variance
pooled across placebo cells was replaced with each cell's own analytic
variance, after the pooled form produced 16/20 spurious `parallel_trends`
failures on a clean scenario (see `panel_checks.placebo_leads`'s docstring)
-- so a second, silently-diverging copy of it is a real, previously-realised
hazard, not a hypothetical one.

These tests guard against the duplicate ever coming back: both estimator
modules must resolve `placebo_leads` to the very same function object (not
merely an equivalent one), and running both estimators on the same panel
with the same control-group choice must produce bit-identical
parallel-trends statistics -- not merely "close", because they are computing
the same thing from the same inputs through the same code path, and any
future edit that reintroduces a private copy in only one file would show up
here as soon as that copy drifted, without needing to wait for a numerical
tolerance to be crossed.
"""

from __future__ import annotations

import numpy as np
from cp_causal import panel_checks
from cp_causal.estimators import callaway_santanna, hierarchical_did
from cp_causal.estimators.callaway_santanna import CallawaySantAnna
from cp_causal.estimators.hierarchical_did import (
    HIER_FAST_MCMC_CONFIG,
    HierarchicalDiD,
)
from cp_core.values import EstimatorConfig
from cp_simulation.panel_scenarios import build_panel_scenario


def _panel_fixture(scenario_id: str = "P3", seed: int = 3, n_units: int = 200, n_periods: int = 8):
    return build_panel_scenario(scenario_id).generate(
        n_units, n_periods, np.random.default_rng(seed)
    )


def test_both_estimators_import_the_same_placebo_leads_function():
    """There is exactly one implementation to drift, not two equivalent ones.

    Checking function identity (`is`, not just "produces the same numbers on
    this fixture") is what actually rules out a reintroduced private copy: an
    accidental reimplementation could easily match on today's test panels and
    still diverge later, the same way the original pooled-variance bug was
    only found by Task 13's recovery harness, not by a fixed-panel unit test.
    """
    assert callaway_santanna.placebo_leads is panel_checks.placebo_leads
    assert hierarchical_did.placebo_leads is panel_checks.placebo_leads


def test_callaway_santanna_and_hierarchical_did_agree_on_the_parallel_trends_statistic():
    """Fit both estimators on the identical panel with the identical control
    group and check their `parallel_trends` chi-square statistics match
    bit-for-bit.

    This is only possible because both ultimately call
    `panel_checks.placebo_leads(panel, "not_yet_treated")` on the same
    `PanelArrays` derived from the same dataset. If either estimator ever
    grew its own private copy of the placebo-lead formula again, the two
    paths could silently diverge -- exactly the kind of drift the
    pooled-vs-per-cell-variance history already produced once -- and this
    test would catch it as a statistic or threshold mismatch, not as a
    vague tolerance failure.
    """
    panel = _panel_fixture()

    cs_config = EstimatorConfig(
        bootstrap_reps=20,
        random_seed=1,
        options={"control_group": "not_yet_treated"},
    )
    cs_result = CallawaySantAnna().fit(panel.dataset, cs_config)
    cs_trends = next(a for a in cs_result.assumptions if a.name == "parallel_trends")

    hier_config = EstimatorConfig(
        random_seed=0, options={"mcmc_config": HIER_FAST_MCMC_CONFIG}
    )
    hier_result = HierarchicalDiD().fit(panel.dataset, hier_config)
    hier_trends = next(a for a in hier_result.assumptions if a.name == "parallel_trends")

    assert cs_trends.status != "NOT_TESTABLE"
    assert hier_trends.status != "NOT_TESTABLE"
    assert cs_trends.statistic == hier_trends.statistic
    assert cs_trends.threshold == hier_trends.threshold


def test_placebo_leads_returns_keys_coefficients_and_covariance_of_matching_shape():
    """`placebo_leads`'s contract: `keys` identifies each row of
    `coefficients` and of `covariance`'s diagonal, in the same order --
    `HierarchicalDiD.pre_treatment_leads` relies on this to find the
    "nearest" lead after the fact, something `CallawaySantAnna` never needed
    when this computation was still two separate, key-less implementations.
    """
    from cp_causal._panel import panel_arrays

    arrays = panel_arrays(_panel_fixture().dataset)
    keys, coefficients, covariance = panel_checks.placebo_leads(arrays, "not_yet_treated")

    assert coefficients.size > 0
    assert keys.shape == (coefficients.size, 2)
    assert covariance.shape == (coefficients.size, coefficients.size)
    # Each cell's variance is independent of every other cell's by
    # construction, so the covariance is exactly diagonal, not merely
    # diagonal-dominant.
    off_diagonal = covariance - np.diag(np.diag(covariance))
    assert np.all(off_diagonal == 0)


def test_parallel_trends_check_survives_a_pinv_that_cannot_converge(monkeypatch):
    # panel_checks.py:parallel_trends_check inverts lead_covariance via
    # cp_statistics.linalg_safety.safe_symmetric_pinv, not np.linalg.pinv
    # directly, for the same reason ols_fit's own pinv call does: this project
    # has twice measured np.linalg.pinv raising `LinAlgError: SVD did not
    # converge` on a genuinely well-posed matrix, on this platform (see
    # linalg_safety's module docstring). This is a preventive fix -- no real
    # cell has yet been observed hitting the bug at this specific call site --
    # so what this test pins is the wiring: that a raising np.linalg.pinv does
    # not crash the check, not a real historical failure.
    def always_fails(A, *args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    rng = np.random.default_rng(4)
    lead_coefficients = rng.normal(size=3)
    M = rng.normal(size=(3, 3))
    lead_covariance = M @ M.T + 3 * np.eye(3)

    monkeypatch.setattr(np.linalg, "pinv", always_fails)
    check = panel_checks.parallel_trends_check(lead_coefficients, lead_covariance)

    assert check.status in ("PASS", "FAIL")
    assert check.statistic is not None and np.isfinite(check.statistic)
