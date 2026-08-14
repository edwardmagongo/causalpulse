import numpy as np
import pytest
from cp_core.results import (
    AssumptionCheck,
    CounterfactualPoint,
    EffectEstimate,
    FitMetadata,
)
from cp_core.values import Interval


def check(status: str, name: str = "overlap") -> AssumptionCheck:
    return AssumptionCheck(
        name=name,
        status=status,
        statistic=0.02,
        threshold=0.1,
        interpretation="Propensity distributions overlap adequately.",
        remedy=None,
    )


def estimate(**overrides: object) -> EffectEstimate:
    defaults: dict[str, object] = {
        "estimand": "ATE",
        "method": "difference_in_means",
        "point": 0.047,
        "se": 0.013,
        "ci": Interval(lower=0.021, upper=0.073),
        "ci_basis": "analytic",
        "bootstrap_ci": Interval(lower=0.020, upper=0.074),
        "p_value": 0.008,
        "n_treated": 500,
        "n_control": 500,
        "scale": "percentage_point",
        "diagnostics": {},
        "assumptions": (check("PASS"),),
        "counterfactual": None,
        "fit_metadata": FitMetadata.capture(
            estimator="difference_in_means",
            config_hash="abc123",
            runtime_seconds=0.01,
            random_seed=7,
        ),
    }
    defaults.update(overrides)
    return EffectEstimate(**defaults)  # type: ignore[arg-type]


def test_assumption_check_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="status"):
        check("MAYBE")


def test_not_testable_is_a_permitted_status() -> None:
    # Unconfoundedness is not testable from data. The system must be able to
    # say so rather than implying a balance table proved it.
    assert check("NOT_TESTABLE").status == "NOT_TESTABLE"


def test_fit_metadata_captures_environment() -> None:
    metadata = FitMetadata.capture(
        estimator="ipw", config_hash="deadbeef", runtime_seconds=1.5, random_seed=3
    )

    assert metadata.estimator == "ipw"
    assert metadata.random_seed == 3
    assert "numpy" in metadata.package_versions
    assert metadata.platform != ""


def test_effect_estimate_reports_worst_assumption_status() -> None:
    result = estimate(
        assumptions=(check("PASS"), check("WARN", "positivity"), check("FAIL", "balance"))
    )

    assert result.worst_assumption_status == "FAIL"


def test_worst_assumption_status_ignores_not_testable() -> None:
    # NOT_TESTABLE is an honesty marker, not a severity level; it must not
    # outrank a real FAIL nor be treated as worse than PASS.
    result = estimate(assumptions=(check("PASS"), check("NOT_TESTABLE", "unconfoundedness")))

    assert result.worst_assumption_status == "PASS"


def test_significance_uses_the_interval_not_the_p_value() -> None:
    assert estimate().is_significant is True
    assert estimate(ci=Interval(lower=-0.01, upper=0.09)).is_significant is False


def test_estimate_rejects_mismatched_arm_counts() -> None:
    with pytest.raises(ValueError, match="n_treated"):
        estimate(n_treated=0)


def test_estimate_rejects_negative_standard_error() -> None:
    with pytest.raises(ValueError, match="se"):
        estimate(se=-0.1)


def test_credible_basis_is_permitted_and_distinct() -> None:
    result = estimate(ci_basis="posterior")

    assert result.ci_basis == "posterior"


def test_estimate_rejects_unknown_ci_basis() -> None:
    with pytest.raises(ValueError, match="ci_basis"):
        estimate(ci_basis="vibes")


def test_counterfactual_point_is_frozen() -> None:
    point = CounterfactualPoint(
        observed_mean=0.25, predicted_absent_treatment=0.207, lower=0.19, upper=0.22
    )
    with pytest.raises(AttributeError):
        point.observed_mean = 0.3  # type: ignore[misc]


def test_fit_metadata_hash_succeeds() -> None:
    metadata = FitMetadata.capture(
        estimator="x",
        config_hash="a",
        runtime_seconds=0.1,
        random_seed=1,
    )
    # This should not raise TypeError: unhashable type: 'dict'
    assert isinstance(hash(metadata), int)


def test_estimate_rejects_p_value_above_one() -> None:
    with pytest.raises(ValueError, match="p_value"):
        estimate(p_value=1.1)


def test_estimate_rejects_p_value_below_zero() -> None:
    with pytest.raises(ValueError, match="p_value"):
        estimate(p_value=-0.1)


def test_estimate_accepts_p_value_at_boundary_zero() -> None:
    result = estimate(p_value=0.0)
    assert result.p_value == 0.0


def test_estimate_accepts_p_value_at_boundary_one() -> None:
    result = estimate(p_value=1.0)
    assert result.p_value == 1.0


def test_estimate_accepts_p_value_none() -> None:
    result = estimate(p_value=None)
    assert result.p_value is None


def test_counterfactual_path_accepts_equal_length_arrays():
    from cp_core.results import CounterfactualPath

    path = CounterfactualPath(
        periods=np.array([1, 2, 3]),
        observed=np.array([1.0, 2.0, 3.5]),
        predicted_absent_treatment=np.array([1.0, 2.0, 2.5]),
        lower=np.array([0.8, 1.8, 2.3]),
        upper=np.array([1.2, 2.2, 2.7]),
        pointwise_effect=np.array([0.0, 0.0, 1.0]),
        cumulative_effect=np.array([0.0, 0.0, 1.0]),
    )

    assert path.periods.size == 3


def test_counterfactual_path_rejects_a_short_bound_array():
    # A path whose bounds are shorter than its periods renders as a silently
    # wrong chart rather than raising, which is the failure this guards.
    from cp_core.results import CounterfactualPath

    with pytest.raises(ValueError, match="same length"):
        CounterfactualPath(
            periods=np.array([1, 2, 3]),
            observed=np.array([1.0, 2.0, 3.5]),
            predicted_absent_treatment=np.array([1.0, 2.0, 2.5]),
            lower=np.array([0.8, 1.8]),
            upper=np.array([1.2, 2.2, 2.7]),
            pointwise_effect=np.array([0.0, 0.0, 1.0]),
            cumulative_effect=np.array([0.0, 0.0, 1.0]),
        )


def test_effect_estimate_accepts_a_counterfactual_path():
    from cp_core.results import CounterfactualPath

    path = CounterfactualPath(
        periods=np.array([1, 2]),
        observed=np.array([1.0, 2.0]),
        predicted_absent_treatment=np.array([1.0, 1.5]),
        lower=np.array([0.9, 1.4]),
        upper=np.array([1.1, 1.6]),
        pointwise_effect=np.array([0.0, 0.5]),
        cumulative_effect=np.array([0.0, 0.5]),
    )

    result = estimate(counterfactual=path)

    assert result.counterfactual is path
