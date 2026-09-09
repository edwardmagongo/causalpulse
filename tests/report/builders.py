"""Shared test builders for `tests/report/`.

`_check`, `_score` and `_estimate` build the three `cp_core.results` types
every report-section test needs, with defaults that make a valid, passing
instance so a test only has to override what it cares about. They started
inside `test_sections.py`; this is the third test file to need them
(`test_markdown.py` after it, `test_html.py`, `test_degenerate.py` and
`test_lint_coverage.py` in later tasks), so they live here once and every
test file imports them explicitly (`from tests.report.builders import
_check, _score, _estimate`) rather than each copying its own.

Plain functions, not fixtures: `conftest.py` only auto-injects fixtures, and
every call site here needs to pass its own overrides inline (e.g.
`_check("unconfoundedness", status="NOT_TESTABLE", statistic=None,
threshold=None)`), which a plain factory fixture would need an extra layer
of indirection to support. An explicit import also makes each test file's
dependency on these builders visible at the top of the file, the same way
the section data functions and `VALIDITY_SCORE_LIMITS` are imported
explicitly rather than injected.
"""

from __future__ import annotations

from cp_core.results import AssumptionCheck, CausalValidityScore, EffectEstimate, FitMetadata
from cp_core.values import Interval


def _check(
    name="overlap",
    kind="identification",
    status="PASS",
    statistic=0.04,
    threshold=0.1,
    interpretation="Standardised mean differences are within threshold.",
    remedy=None,
):
    return AssumptionCheck(
        name=name,
        kind=kind,
        status=status,
        statistic=statistic,
        threshold=threshold,
        interpretation=interpretation,
        remedy=remedy,
    )


def _score(band="MODERATE", ceiling="STRONG"):
    return CausalValidityScore(
        band=band,
        ceiling=ceiling,
        rationale=("Design provenance is observational, so the band cannot exceed STRONG.",),
        checks=(_check(),),
    )


def _estimate(method="ipw", estimand="ATE", point=0.25, lo=0.05, hi=0.45, basis="analytic"):
    return EffectEstimate(
        estimand=estimand,
        method=method,
        point=point,
        se=0.1,
        ci=Interval(lower=lo, upper=hi, level=0.95),
        ci_basis=basis,
        bootstrap_ci=None,
        p_value=0.01,
        n_treated=100,
        n_control=100,
        scale="percentage_point",
        diagnostics={},
        assumptions=(),
        counterfactual=None,
        fit_metadata=FitMetadata.capture(
            estimator=method, config_hash="h", runtime_seconds=0.1, random_seed=0
        ),
        subgroup_findings=(),
    )
