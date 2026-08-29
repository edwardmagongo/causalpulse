"""The negative-control outcome is declared on the schema, and validated there.

Declared rather than inferred, following `assignment_provenance`'s precedent:
which column is a negative control is a claim about how the data came to
exist, not something a heuristic can read off the values.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from cp_core.dataset import CausalDataset, CausalSchema
from cp_core.results import (
    MisclassificationCurve,
    MisclassificationPoint,
    NegativeControlResult,
)
from cp_core.values import Interval


def frame(with_nc: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {
        "unit_id": np.arange(50),
        "x1": rng.normal(size=50),
        "treated": rng.integers(0, 2, size=50),
        "outcome": rng.integers(0, 2, size=50),
    }
    if with_nc:
        data["outcome_nc"] = rng.integers(0, 2, size=50)
    return pd.DataFrame(data)


def schema(nc: str | None) -> CausalSchema:
    return CausalSchema(
        unit="unit_id",
        treatment="treated",
        outcome="outcome",
        covariates=("x1",),
        outcome_type="binary",
        negative_control_outcome=nc,
    )


def test_the_field_defaults_to_none():
    assert schema(None).negative_control_outcome is None


def test_required_columns_includes_the_negative_control_when_declared():
    assert "outcome_nc" in schema("outcome_nc").required_columns


def test_required_columns_omits_it_when_absent():
    assert all(not c.endswith("_nc") for c in schema(None).required_columns)


def test_a_declared_column_missing_from_the_frame_is_refused_at_construction():
    # Refused when the dataset is built, not when a check later tries to read
    # it -- an absent column is a structural problem, and the existing
    # `_validate_columns_present` is already the place that says so.
    with pytest.raises(ValueError, match="outcome_nc"):
        CausalDataset(frame=frame(with_nc=False), schema=schema("outcome_nc"))


def test_a_declared_column_present_in_the_frame_constructs():
    dataset = CausalDataset(frame=frame(), schema=schema("outcome_nc"))
    assert dataset.schema.negative_control_outcome == "outcome_nc"


def test_the_negative_control_may_not_also_be_the_outcome():
    # Swapping the outcome for itself would make the check compare an
    # estimate against itself and always read as clean.
    with pytest.raises(ValueError, match="negative_control_outcome"):
        CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=("x1",),
            outcome_type="binary",
            negative_control_outcome="outcome",
        )


def test_the_negative_control_may_not_be_a_covariate():
    # A negative-control outcome is downstream of the covariates; adjusting
    # for it would be a different and much worse analysis.
    with pytest.raises(ValueError, match="negative_control_outcome"):
        CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=("x1",),
            outcome_type="binary",
            negative_control_outcome="x1",
        )


def test_the_negative_control_may_not_collide_with_treatment():
    # A negative-control refit swaps this column in as the outcome and refits
    # the same estimator; colliding with treatment would regress treatment on
    # itself rather than test anything.
    with pytest.raises(ValueError, match="negative_control_outcome"):
        CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=("x1",),
            outcome_type="binary",
            negative_control_outcome="treated",
        )


def test_the_negative_control_may_not_collide_with_unit():
    with pytest.raises(ValueError, match="negative_control_outcome"):
        CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=("x1",),
            outcome_type="binary",
            negative_control_outcome="unit_id",
        )


def test_the_negative_control_may_not_collide_with_time_or_cohort():
    with pytest.raises(ValueError, match="negative_control_outcome"):
        CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=("x1",),
            outcome_type="binary",
            time="period",
            negative_control_outcome="period",
        )
    with pytest.raises(ValueError, match="negative_control_outcome"):
        CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=("x1",),
            outcome_type="binary",
            cohort="adoption_cohort",
            negative_control_outcome="adoption_cohort",
        )


def a_negative_control() -> NegativeControlResult:
    return NegativeControlResult(
        outcome_column="outcome_nc",
        estimate=0.004,
        ci=Interval(lower=-0.02, upper=0.03),
        excludes_null=False,
        estimator="lin_regression",
    )


def a_point() -> MisclassificationPoint:
    return MisclassificationPoint(
        sensitivity=0.9,
        specificity=0.9,
        corrected_point=0.125,
        corrected_ci=Interval(lower=0.05, upper=0.20),
        attenuation_factor=0.8,
    )


def test_the_two_new_slots_are_pinned_as_reason_carrying():
    # Exact, matching test_sensitivity_contracts.py's style: a new optional
    # slot added later without a reason fails here rather than shipping.
    from cp_core.results import FalsificationResult, SensitivityReport

    falsification = {f.name for f in dataclasses.fields(FalsificationResult)}
    report = {f.name for f in dataclasses.fields(SensitivityReport)}
    assert {"negative_control", "negative_control_reason"} <= falsification
    assert {"misclassification", "misclassification_reason"} <= report


def test_excludes_null_must_agree_with_the_interval():
    # Two fields that must agree, so the contract refuses a disagreement
    # rather than letting a reader pick which to believe.
    with pytest.raises(ValueError, match="excludes_null"):
        NegativeControlResult(
            outcome_column="outcome_nc",
            estimate=0.004,
            ci=Interval(lower=-0.02, upper=0.03),
            excludes_null=True,
            estimator="lin_regression",
        )


def test_a_negative_control_result_constructs_when_they_agree():
    assert a_negative_control().excludes_null is False


def test_a_misclassification_point_refuses_a_non_positive_attenuation():
    # attenuation <= 0 means the assumed rates carry no information about the
    # true label; the correction is undefined and must not be constructed.
    with pytest.raises(ValueError, match="attenuation_factor"):
        MisclassificationPoint(
            sensitivity=0.5,
            specificity=0.5,
            corrected_point=1.0,
            corrected_ci=Interval(lower=0.0, upper=2.0),
            attenuation_factor=0.0,
        )


@pytest.mark.parametrize(("se", "sp"), [(-0.1, 0.9), (1.1, 0.9), (0.9, -0.1), (0.9, 1.1)])
def test_a_misclassification_point_refuses_rates_outside_the_unit_interval(se, sp):
    with pytest.raises(ValueError, match="must lie in"):
        MisclassificationPoint(
            sensitivity=se,
            specificity=sp,
            corrected_point=1.0,
            corrected_ci=Interval(lower=0.0, upper=2.0),
            attenuation_factor=0.8,
        )


def test_a_curve_must_hold_at_least_one_point():
    with pytest.raises(ValueError, match="at least one"):
        MisclassificationCurve(
            observed_point=0.1, points=(), observed_treated_share=0.5
        )


def test_a_curve_refuses_an_impossible_treated_share():
    with pytest.raises(ValueError, match="observed_treated_share"):
        MisclassificationCurve(
            observed_point=0.1, points=(a_point(),), observed_treated_share=1.0
        )


def test_a_curve_constructs_and_keeps_its_points():
    curve = MisclassificationCurve(
        observed_point=0.1, points=(a_point(),), observed_treated_share=0.5
    )
    assert curve.points[0].attenuation_factor == 0.8
