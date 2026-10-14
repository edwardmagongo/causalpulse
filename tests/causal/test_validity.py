"""The validity rubric: every rule, and the property that guards all of them."""

import pytest
from cp_causal.validity import score_validity
from cp_core.language import causal_language_lint
from cp_core.results import BAND_ORDER, AssumptionCheck, EffectEstimate, FitMetadata
from cp_core.values import Interval
from hypothesis import given, settings
from hypothesis import strategies as st

KINDS = ("identification", "estimation", "data_adequacy")
STATUSES = ("PASS", "WARN", "FAIL", "NOT_TESTABLE")
PROVENANCES = ("randomised", "natural_experiment", "observational")

# Captured once at import time and reused by every `estimate()` call below.
# `FitMetadata.capture` forks a `git rev-parse HEAD` subprocess and does
# several `importlib.metadata.version()` lookups (~13ms combined), and this
# module's property test alone builds two `EffectEstimate`s per Hypothesis
# example. None of these tests inspect `fit_metadata`, so a single shared
# instance is equivalent and removes real, repeated subprocess/IO cost.
_FIT_METADATA = FitMetadata.capture(
    estimator="twfe", config_hash="abc", runtime_seconds=0.1, random_seed=0
)


def check(kind: str, status: str, name: str = "parallel_trends") -> AssumptionCheck:
    return AssumptionCheck(
        name=name,
        kind=kind,
        status=status,
        statistic=0.1,
        threshold=0.2,
        interpretation="Pre-treatment leads are jointly indistinguishable from zero.",
        remedy=None,
    )


def estimate(*checks: AssumptionCheck) -> EffectEstimate:
    return EffectEstimate(
        estimand="ATT",
        method="twfe",
        point=2.0,
        se=0.5,
        ci=Interval(1.0, 3.0),
        ci_basis="analytic",
        bootstrap_ci=None,
        p_value=0.01,
        n_treated=100,
        n_control=100,
        scale="absolute",
        diagnostics={},
        assumptions=checks,
        counterfactual=None,
        fit_metadata=_FIT_METADATA,
    )


@pytest.mark.parametrize(
    ("provenance", "expected"),
    [
        ("randomised", "STRONG"),
        ("natural_experiment", "STRONG"),
        ("observational", "MODERATE"),
    ],
)
def test_a_clean_fit_lands_on_its_design_ceiling(provenance: str, expected: str):
    scored = score_validity(estimate(check("identification", "PASS")), provenance=provenance)
    assert scored.band == expected
    assert scored.ceiling == expected


def test_an_observational_design_cannot_reach_strong():
    # The design-based ceiling. Every check passing does not lift it.
    clean = estimate(*(check(kind, "PASS") for kind in KINDS))
    assert score_validity(clean, provenance="observational").band == "MODERATE"


def test_an_estimation_failure_is_insufficient_whatever_the_design():
    # An unconverged sampler does not produce a weakly-identified estimate. It
    # produces no usable estimate, so the design ceiling is irrelevant.
    broken = estimate(check("estimation", "FAIL", name="divergent_transitions"))
    assert score_validity(broken, provenance="randomised").band == "INSUFFICIENT"


@pytest.mark.parametrize("kind", ["identification", "data_adequacy"])
def test_an_identification_or_adequacy_failure_caps_at_weak(kind: str):
    failed = estimate(check(kind, "FAIL"))
    assert score_validity(failed, provenance="randomised").band == "WEAK"


def test_a_warning_drops_one_step_from_the_ceiling():
    warned = estimate(check("identification", "WARN"))
    assert score_validity(warned, provenance="randomised").band == "MODERATE"
    assert score_validity(warned, provenance="observational").band == "WEAK"


def test_the_lowest_cap_wins_when_several_apply():
    # A FAIL caps at WEAK and a WARN caps one step below STRONG (MODERATE).
    # WEAK is lower, so WEAK is the answer -- the caps compose by minimum, they
    # do not chain into INSUFFICIENT.
    both = estimate(check("identification", "FAIL"), check("data_adequacy", "WARN"))
    assert score_validity(both, provenance="randomised").band == "WEAK"


def test_not_testable_is_named_but_does_not_move_the_band():
    # Already priced in by the ceiling: an untestable identifying assumption is
    # what makes a design observational in the first place. Subtracting for it
    # as well would count the same fact twice.
    untestable = estimate(check("identification", "NOT_TESTABLE", name="unconfoundedness"))
    scored = score_validity(untestable, provenance="observational")
    assert scored.band == "MODERATE"
    assert any("unconfoundedness" in line for line in scored.rationale)


def test_an_estimate_with_no_checks_lands_on_its_ceiling():
    scored = score_validity(estimate(), provenance="observational")
    assert scored.band == "MODERATE"
    assert scored.rationale


def test_the_score_carries_the_checks_that_produced_it():
    checks = (check("identification", "PASS"), check("estimation", "PASS", name="r_hat"))
    assert score_validity(estimate(*checks), provenance="randomised").checks == checks


def test_every_rationale_line_passes_the_causal_language_lint():
    scored = score_validity(
        estimate(*(check(kind, status) for kind in KINDS for status in STATUSES)),
        provenance="observational",
    )
    for line in scored.rationale:
        assert causal_language_lint(line) == (), line


# Hypothesis's default 200ms-per-example deadline exists to catch accidentally
# quadratic code, not to be a timing SLA -- and score_validity here does none
# of that; it's a handful of dict lookups and a min() over a short tuple. The
# flakiness this guards against was never about score_validity's own speed
# (measured at ~0.0013ms) or JAX/numpyro thread-pool jitter -- it was every
# `estimate()` call in this file constructing a fresh `EffectEstimate` via
# `FitMetadata.capture()`, which forks a `git rev-parse HEAD` subprocess and
# does several `importlib.metadata.version()` lookups (~13ms combined) on
# every call, and this property builds two `EffectEstimate`s per example.
# That real subprocess/IO cost is now eliminated by the shared
# `_FIT_METADATA` module constant above, but the deadline stays off: wall
# clock was never a meaningful thing to assert about this correctness
# property, so there is nothing to gain by re-enabling it.
@settings(deadline=None)
@given(
    provenance=st.sampled_from(PROVENANCES),
    kinds=st.lists(st.sampled_from(KINDS), min_size=1, max_size=6),
    statuses=st.lists(st.sampled_from(STATUSES), min_size=6, max_size=6),
    index=st.integers(min_value=0, max_value=5),
)
def test_worsening_any_check_never_raises_the_band(provenance, kinds, statuses, index):
    """The monotonicity guarantee the whole rubric rests on.

    Written as a property rather than a table because the failure it guards
    against -- some combination of caps that lets a worse fit score higher --
    is exactly the kind a hand-written table would not think to include.
    """
    severity = {"PASS": 0, "NOT_TESTABLE": 0, "WARN": 1, "FAIL": 2}
    position = index % len(kinds)
    before = [statuses[i % len(statuses)] for i in range(len(kinds))]
    worse = list(before)
    worse[position] = "FAIL"
    if severity[worse[position]] <= severity[before[position]]:
        return
    band_before = score_validity(
        estimate(*(check(k, s) for k, s in zip(kinds, before, strict=True))),
        provenance=provenance,
    ).band
    band_after = score_validity(
        estimate(*(check(k, s) for k, s in zip(kinds, worse, strict=True))),
        provenance=provenance,
    ).band
    assert BAND_ORDER[band_after] <= BAND_ORDER[band_before]
