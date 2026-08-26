import numpy as np
import pytest
from cp_core.results import (
    VALIDITY_SCORE_LIMITS,
    AssumptionCheck,
    CausalValidityScore,
    CounterfactualPoint,
    EffectEstimate,
    FitMetadata,
)
from cp_core.values import Interval


def check(status: str, name: str = "overlap", kind: str = "identification") -> AssumptionCheck:
    return AssumptionCheck(
        name=name,
        kind=kind,
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


def test_assumption_check_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        check("PASS", kind="vibes")


@pytest.mark.parametrize("kind", ["identification", "estimation", "data_adequacy"])
def test_assumption_check_accepts_every_declared_kind(kind: str):
    assert check("PASS", kind=kind).kind == kind


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


def test_subgroup_finding_holds_a_named_condition_and_a_credible_or_confidence_interval():
    from cp_core.results import SubgroupFinding
    from cp_core.values import Interval

    finding = SubgroupFinding(
        covariate="x1",
        condition="x1 == 1",
        n_units=340,
        cate_estimate=0.09,
        ci=Interval(lower=0.04, upper=0.14),
        q_value=0.01,
    )
    assert finding.covariate == "x1"
    assert finding.q_value == 0.01


def test_subgroup_finding_rejects_a_q_value_outside_zero_one():
    from cp_core.results import SubgroupFinding
    from cp_core.values import Interval

    try:
        SubgroupFinding(
            covariate="x1", condition="x1 == 1", n_units=10,
            cate_estimate=0.0, ci=Interval(lower=-0.1, upper=0.1), q_value=1.5,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_effect_estimate_subgroup_findings_defaults_to_empty():
    # Every existing estimator's EffectEstimate(...) construction call omits
    # subgroup_findings entirely -- this must stay valid unchanged, or every
    # other estimator's tests break on this task's contract change. Reuse
    # an existing estimator's real fit() output rather than hand-constructing
    # an EffectEstimate here, since the point is to prove real call sites
    # aren't broken, not just that the dataclass accepts a default.
    import numpy as np
    import pandas as pd
    from cp_causal.estimators.experimental import DifferenceInMeans  # any existing estimator
    from cp_core.dataset import CausalDataset, CausalSchema
    from cp_core.values import EstimatorConfig

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "unit_id": np.arange(200),
            "treated": rng.binomial(1, 0.5, 200),
            "outcome": rng.normal(size=200),
        }
    )
    schema = CausalSchema(
        unit="unit_id", treatment="treated", outcome="outcome",
        covariates=(), outcome_type="continuous",
    )
    result = DifferenceInMeans().fit(
        CausalDataset(frame=frame, schema=schema), EstimatorConfig()
    )
    assert result.subgroup_findings == ()


def score(band: str = "MODERATE", ceiling: str = "MODERATE") -> CausalValidityScore:
    return CausalValidityScore(
        band=band,
        ceiling=ceiling,
        rationale=("Design provenance is observational.",),
        checks=(check("PASS"),),
    )


def test_validity_score_rejects_an_unknown_band():
    with pytest.raises(ValueError, match="band must be one of"):
        score(band="EXCELLENT")


def test_validity_score_rejects_a_band_above_its_own_ceiling():
    # The ceiling is a cap, so this state is unreachable through the rubric.
    # Rejecting it here means a hand-built or deserialised score cannot express
    # the one thing section 6.2 forbids -- an observational design scoring
    # STRONG -- however it was constructed.
    with pytest.raises(ValueError, match="band STRONG exceeds ceiling MODERATE"):
        score(band="STRONG", ceiling="MODERATE")


def test_validity_score_requires_a_rationale():
    with pytest.raises(ValueError, match="rationale must not be empty"):
        CausalValidityScore(
            band="MODERATE", ceiling="MODERATE", rationale=(), checks=(check("PASS"),)
        )


def test_validity_score_always_carries_its_stated_limits():
    assert score().limits == VALIDITY_SCORE_LIMITS
    assert "not a probability" in VALIDITY_SCORE_LIMITS


def test_validity_score_carries_the_table_that_produced_it():
    # Section 6.2 requires the band to be printed alongside its check table.
    # Carrying the checks on the score makes that structural: there is no way
    # to hold a band without the table, so no rendering can omit it.
    assert score().checks == (check("PASS"),)
