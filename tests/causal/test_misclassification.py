"""The misclassification correction, checked against the arithmetic that
motivated it.

The formula's whole point is that the widely quoted `(se + sp - 1)` factor is
incomplete: it holds only when the true and observed treated shares coincide.
The parametrised case below is the one that discriminates the two forms, and
it is taken from the design doc's own verified table.
"""

import pytest
from cp_causal.sensitivity import MISCLASSIFICATION_RATES, misclassification_curve
from cp_core.results import EffectEstimate, FitMetadata
from cp_core.values import Interval


def an_estimate(point=1.6, lower=1.4, upper=1.8, n_treated=400, n_control=600,
                scale="absolute") -> EffectEstimate:
    return EffectEstimate(
        estimand="ATE", method="stub", point=point, se=None,
        ci=Interval(lower=lower, upper=upper), ci_basis="analytic",
        bootstrap_ci=None, p_value=None,
        n_treated=n_treated, n_control=n_control, scale=scale,
        diagnostics={}, assumptions=(), counterfactual=None,
        fit_metadata=FitMetadata.capture(
            estimator="stub", config_hash="0" * 16, runtime_seconds=0.0,
            random_seed=None,
        ),
    )


def test_perfect_classification_leaves_the_estimate_alone():
    # se = sp = 1 gives attenuation exactly 1, so the curve contains the
    # analysis as actually run as its own anchor.
    curve = misclassification_curve(an_estimate(), rates=((1.0, 1.0),))
    assert curve.points[0].attenuation_factor == pytest.approx(1.0)
    assert curve.points[0].corrected_point == pytest.approx(1.6)


def test_the_default_grid_contains_the_identity_pair():
    assert (1.0, 1.0) in MISCLASSIFICATION_RATES


def test_correction_always_moves_away_from_the_null():
    # Non-differential misclassification attenuates, so every correction is
    # larger in magnitude than the observed estimate. This is the property the
    # report states structurally, and the one a reader is most likely to have
    # backwards.
    curve = misclassification_curve(an_estimate(point=1.6))
    for point in curve.points:
        assert abs(point.corrected_point) >= abs(curve.observed_point) - 1e-12


def test_the_balanced_symmetric_case_matches_the_naive_factor():
    # p = q = 0.5 with se = sp: the only configuration where the naive
    # (se + sp - 1) form is correct. 400/400 makes q = 0.5.
    curve = misclassification_curve(
        an_estimate(n_treated=400, n_control=400), rates=((0.9, 0.9),)
    )
    assert curve.points[0].attenuation_factor == pytest.approx(0.8, rel=1e-9)


def test_the_asymmetric_case_does_not_match_the_naive_factor():
    # se=0.8, sp=1.0 at an observed share of 0.4 implies a true share of 0.5,
    # giving attenuation 0.8 * 0.25 / 0.24 = 0.8333..., not 0.8. This is the
    # case that discriminates the correct formula from the naive one; if it
    # reads 0.8, the prevalence terms have been dropped.
    curve = misclassification_curve(
        an_estimate(n_treated=400, n_control=600), rates=((0.8, 1.0),)
    )
    factor = curve.points[0].attenuation_factor
    assert factor == pytest.approx(0.8 * 0.25 / 0.24, rel=1e-9)
    assert factor != pytest.approx(0.8, rel=1e-6)


def test_the_design_docs_verified_table_reproduces():
    # Spec section 4.3, verified by simulation at 4,000,000 units per row.
    # Given an observed estimate and share, the correction must return the
    # true effect of 2.0. `q` is supplied through the arm counts.
    for q, se, sp, b_obs in [
        (0.5000, 0.9, 0.9, 1.5975),
        (0.4000, 0.8, 1.0, 1.6644),
        (0.2902, 0.85, 0.95, 1.6308),
        (0.6896, 0.9, 0.8, 1.3757),
    ]:
        n = 100_000
        n_treated = int(round(q * n))
        estimate = an_estimate(
            point=b_obs, lower=b_obs - 0.1, upper=b_obs + 0.1,
            n_treated=n_treated, n_control=n - n_treated,
        )
        corrected = misclassification_curve(estimate, rates=((se, sp),)).points[0]
        assert corrected.corrected_point == pytest.approx(2.0, abs=0.01), (q, se, sp)


def test_the_interval_is_scaled_by_the_same_factor():
    curve = misclassification_curve(
        an_estimate(point=1.6, lower=1.4, upper=1.8), rates=((0.9, 0.9),)
    )
    point = curve.points[0]
    assert point.corrected_ci.lower == pytest.approx(1.4 / point.attenuation_factor)
    assert point.corrected_ci.upper == pytest.approx(1.8 / point.attenuation_factor)


def test_rates_summing_to_one_are_skipped_not_returned_as_infinity():
    # se + sp = 1 means the observed label carries no information about the
    # true one. The correction does not exist there; the pair is dropped
    # rather than returned as an infinity.
    curve = misclassification_curve(an_estimate(), rates=((1.0, 1.0), (0.5, 0.5)))
    assert len(curve.points) == 1
    assert curve.points[0].sensitivity == 1.0


def test_rates_incompatible_with_the_observed_share_are_skipped():
    # sp = 1.0 means no false positives, so the observed treated share cannot
    # exceed the true one; se = 0.2 with q = 0.4 implies a true share above 1.
    curve = misclassification_curve(
        an_estimate(n_treated=400, n_control=600), rates=((1.0, 1.0), (0.2, 1.0))
    )
    assert all(p.sensitivity == 1.0 for p in curve.points)


def test_a_grid_with_no_usable_pair_is_refused():
    with pytest.raises(ValueError, match="no usable"):
        misclassification_curve(an_estimate(), rates=((0.5, 0.5),))


def test_a_log_ratio_estimate_is_refused():
    # The derivation is for a difference of means. Applying it to a log ratio
    # would divide a ratio by an attenuation derived for a difference.
    with pytest.raises(ValueError, match="log_ratio"):
        misclassification_curve(an_estimate(scale="log_ratio"))


def test_the_observed_share_is_read_off_the_estimate():
    curve = misclassification_curve(an_estimate(n_treated=250, n_control=750))
    assert curve.observed_treated_share == pytest.approx(0.25)
