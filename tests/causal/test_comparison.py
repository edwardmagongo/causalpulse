"""Cross-method comparison: what it reports, and what it refuses to do."""

import numpy as np
import pytest
from cp_causal.comparison import compare
from cp_causal.validity import score_validity
from cp_core.language import causal_language_lint
from cp_core.results import AssumptionCheck, EffectEstimate, FitMetadata
from cp_core.values import Interval

# Captured once at import time and reused by every `estimate()` call below.
# `FitMetadata.capture` forks a `git rev-parse HEAD` subprocess and does
# several `importlib.metadata.version()` lookups (~13ms combined) on every
# call. Nothing in this file inspects `fit_metadata` (it is required plumbing
# for the `EffectEstimate` constructor, not something `compare()` or any
# assertion here reads), so a single shared instance is equivalent and
# removes real, repeated subprocess/IO cost across this file's tests.
_FIT_METADATA = FitMetadata.capture(
    estimator="twfe", config_hash="abc", runtime_seconds=0.1, random_seed=0
)


def check(name: str, kind: str = "identification", status: str = "PASS") -> AssumptionCheck:
    return AssumptionCheck(
        name=name,
        kind=kind,
        status=status,
        statistic=None,
        threshold=None,
        interpretation="Recorded for comparison.",
        remedy=None,
    )


def estimate(
    method: str,
    point: float,
    ci: tuple[float, float],
    *checks: AssumptionCheck,
    estimand: str = "ATT",
    scale: str = "absolute",
    ci_basis: str = "analytic",
) -> EffectEstimate:
    return EffectEstimate(
        estimand=estimand,
        method=method,
        point=point,
        se=0.5,
        ci=Interval(*ci),
        ci_basis=ci_basis,
        bootstrap_ci=None,
        p_value=None,
        n_treated=100,
        n_control=100,
        scale=scale,
        diagnostics={},
        assumptions=checks,
        counterfactual=None,
        fit_metadata=_FIT_METADATA,
    )


def scored(*estimates: EffectEstimate, provenance: str = "natural_experiment"):
    return estimates, tuple(
        score_validity(e, provenance=provenance) for e in estimates
    )


def test_overlapping_intervals_are_consistent():
    result = compare(*scored(
        estimate("twfe", 2.0, (1.0, 3.0), check("parallel_trends")),
        estimate("callaway_santanna", 2.4, (1.5, 3.3), check("parallel_trends")),
    ))
    assert result.agreement == "consistent"
    assert result.point_range == (2.0, 2.4)


def test_disjoint_intervals_are_divergent():
    result = compare(*scored(
        estimate("twfe", 2.0, (1.8, 2.2), check("parallel_trends")),
        estimate("callaway_santanna", 5.0, (4.5, 5.5), check("parallel_trends")),
    ))
    assert result.agreement == "divergent"


def test_comparison_exposes_no_pooled_estimate():
    # Estimates are not averaged. A test asserting no averaging happens is
    # weak; a type off which no pooled number can be read is not.
    result = compare(*scored(
        estimate("twfe", 2.0, (1.0, 3.0)),
        estimate("callaway_santanna", 2.4, (1.5, 3.3)),
    ))
    fields = set(vars(result))
    assert not fields & {"combined", "pooled", "mean", "meta_estimate", "average"}


def test_comparison_refuses_a_mixed_estimand():
    with pytest.raises(ValueError, match="estimand"):
        compare(*scored(
            estimate("twfe", 2.0, (1.0, 3.0)),
            estimate("aipw", 2.4, (1.5, 3.3), estimand="ATE"),
        ))


def test_comparison_refuses_a_mixed_scale():
    with pytest.raises(ValueError, match="scale"):
        compare(*scored(
            estimate("twfe", 2.0, (1.0, 3.0)),
            estimate("did_2x2", 2.4, (1.5, 3.3), scale="log_ratio"),
        ))


def test_comparison_refuses_fewer_than_two_methods():
    with pytest.raises(ValueError, match="at least two"):
        compare(*scored(estimate("twfe", 2.0, (1.0, 3.0))))


def test_comparison_refuses_duplicate_method_names():
    # by_method is keyed by method name, so two fits of the same estimator
    # would silently overwrite each other in the assumption diff.
    with pytest.raises(ValueError, match="duplicate method"):
        compare(*scored(
            estimate("twfe", 2.0, (1.0, 3.0)),
            estimate("twfe", 2.4, (1.5, 3.3)),
        ))


def test_insufficient_methods_are_excluded_from_the_verdict():
    usable_a = estimate("twfe", 2.0, (1.8, 2.2), check("parallel_trends"))
    usable_b = estimate("callaway_santanna", 2.1, (1.9, 2.3), check("parallel_trends"))
    broken = estimate(
        "bsts", 9.0, (8.0, 10.0), check("r_hat", kind="estimation", status="FAIL")
    )
    result = compare(*scored(usable_a, usable_b, broken))
    # The unusable fit is far outside the others and would make the set look
    # divergent; a band the rubric just called INSUFFICIENT must not drive the
    # verdict.
    assert result.agreement == "consistent"
    assert "bsts" in result.verdict


def test_agreement_is_not_assessable_below_two_usable_methods():
    usable = estimate("twfe", 2.0, (1.8, 2.2), check("parallel_trends"))
    broken = estimate(
        "bsts", 9.0, (8.0, 10.0), check("r_hat", kind="estimation", status="FAIL")
    )
    result = compare(*scored(usable, broken))
    assert result.agreement == "not_assessable"


def test_assumption_diff_reports_a_check_only_one_method_relies_on():
    result = compare(*scored(
        estimate("twfe", 2.0, (1.0, 3.0), check("parallel_trends")),
        estimate("aipw", 2.4, (1.5, 3.3), check("positivity")),
    ))
    by_name = {d.check_name: d for d in result.assumption_diff}
    assert by_name["parallel_trends"].by_method == {"twfe": "PASS", "aipw": None}
    assert by_name["positivity"].by_method == {"twfe": None, "aipw": "PASS"}


def test_assumption_diff_omits_checks_every_method_agrees_on():
    result = compare(*scored(
        estimate("twfe", 2.0, (1.0, 3.0), check("parallel_trends")),
        estimate("callaway_santanna", 2.4, (1.5, 3.3), check("parallel_trends")),
    ))
    assert result.assumption_diff == ()


def test_overlap_matrix_is_square_symmetric_and_true_on_the_diagonal():
    result = compare(*scored(
        estimate("twfe", 2.0, (1.8, 2.2)),
        estimate("callaway_santanna", 5.0, (4.5, 5.5)),
        estimate("did_2x2", 2.1, (1.9, 2.3)),
    ))
    assert result.overlap.shape == (3, 3)
    assert np.array_equal(result.overlap, result.overlap.T)
    assert result.overlap.diagonal().all()


def test_overlap_matrix_values_reflect_pairwise_interval_overlap():
    # A value-level check, not just shape/symmetry/diagonal: touching
    # endpoints must still count as overlap (these are closed intervals), and
    # a pair with no shared point must read False in both directions -- a
    # matrix that were merely symmetric with a true diagonal could still get
    # every off-diagonal entry backwards.
    result = compare(*scored(
        estimate("twfe", 2.0, (1.0, 2.0)),
        estimate("callaway_santanna", 2.5, (2.0, 3.0)),  # touches twfe at 2.0
        estimate("did_2x2", 9.0, (8.0, 10.0)),  # disjoint from both
    ))
    expected = np.array(
        [
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ]
    )
    assert np.array_equal(result.overlap, expected)


def test_excluded_method_point_still_counted_in_the_descriptive_spread():
    # point_range and dispersion are descriptive fields over every estimate
    # handed in, not just the usable ones. Restricting them to the usable
    # subset would divide by zero (sample standard deviation on a single
    # point) whenever exactly one method survives the WEAK-or-better filter --
    # which is exactly the not_assessable case just above. It is the
    # agreement judgment, not these fields, that excludes the broken fit.
    usable_a = estimate("twfe", 2.0, (1.8, 2.2), check("parallel_trends"))
    usable_b = estimate("callaway_santanna", 2.1, (1.9, 2.3), check("parallel_trends"))
    broken = estimate(
        "bsts", 9.0, (8.0, 10.0), check("r_hat", kind="estimation", status="FAIL")
    )
    result = compare(*scored(usable_a, usable_b, broken))
    assert result.point_range == (2.0, 9.0)


def test_divergent_verdict_with_no_assumption_differences_does_not_point_at_an_empty_diff():
    # All three methods carry the identical check, so assumption_diff is empty
    # even though the intervals themselves are divergent. The verdict must not
    # tell the reader to "read assumption_diff" for insight an empty tuple
    # cannot contain.
    result = compare(*scored(
        estimate("twfe", 2.0, (1.8, 2.2), check("parallel_trends")),
        estimate("callaway_santanna", 5.0, (4.5, 5.5), check("parallel_trends")),
        estimate("did_2x2", 9.0, (8.5, 9.5), check("parallel_trends")),
    ))
    assert result.agreement == "divergent"
    assert result.assumption_diff == ()
    assert "0 assumption(s) differ" not in result.verdict
    assert "read assumption_diff" not in result.verdict
    assert "not mutually consistent" in result.verdict


def test_verdict_records_the_interval_bases_it_mixed():
    result = compare(*scored(
        estimate("twfe", 2.0, (1.0, 3.0)),
        estimate("bsts", 2.2, (1.2, 3.2), ci_basis="posterior"),
    ))
    assert "posterior" in result.verdict and "analytic" in result.verdict


def test_verdict_passes_the_causal_language_lint():
    for result in (
        compare(*scored(
            estimate("twfe", 2.0, (1.0, 3.0), check("parallel_trends")),
            estimate("callaway_santanna", 2.4, (1.5, 3.3), check("parallel_trends")),
        )),
        compare(*scored(
            estimate("twfe", 2.0, (1.8, 2.2), check("parallel_trends")),
            estimate("aipw", 9.0, (8.5, 9.5), check("positivity")),
        )),
        compare(*scored(
            estimate("twfe", 2.0, (1.8, 2.2), check("parallel_trends")),
            estimate("callaway_santanna", 5.0, (4.5, 5.5), check("parallel_trends")),
            estimate("did_2x2", 9.0, (8.5, 9.5), check("parallel_trends")),
        )),
    ):
        assert causal_language_lint(result.verdict) == (), result.verdict
