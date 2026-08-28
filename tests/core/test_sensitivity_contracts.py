"""The sensitivity contract's absences are always explained.

The invariant under test is not "these fields exist" -- it is that no slot
can be None without a sibling saying why. That is the shape Phase 5 shipped
wrong, and it is worth a structural test rather than a spot check: the exact
set of optional slots is pinned here, so a new one added later without a
reason fails this file rather than shipping as a silent gap.
"""

import dataclasses

import numpy as np
import pytest
from cp_core.results import (
    CinelliHazlettResult,
    ConfoundingSensitivity,
    EValueResult,
    FalsificationResult,
    PlaceboTestResult,
    SensitivityReport,
    SpecificationCurve,
    SpecificationPoint,
)
from cp_core.values import Interval

# Every (type, value field, reason field) triple this contract carries.
# Exact, not illustrative: a new optional slot must be added here, which is
# the point -- the test fails until someone states what its absence means.
PINNED_OPTIONAL_SLOTS = {
    (ConfoundingSensitivity, "cinelli_hazlett", "cinelli_hazlett_reason"),
    (FalsificationResult, "placebo_dates", "placebo_dates_reason"),
    (FalsificationResult, "placebo_units", "placebo_units_reason"),
    (FalsificationResult, "leave_one_donor_out", "leave_one_donor_out_reason"),
    (
        FalsificationResult,
        "randomization_inference",
        "randomization_inference_reason",
    ),
    (SensitivityReport, "specification_curve", "specification_curve_reason"),
}


def an_e_value() -> EValueResult:
    return EValueResult(
        e_value=2.0,
        e_value_ci_bound=1.5,
        approximated=False,
        approximation_note=None,
    )


def a_placebo() -> PlaceboTestResult:
    return PlaceboTestResult(
        placebo_effects=np.array([0.1, -0.2, 0.05]),
        real_effect=0.9,
        p_value=0.25,
    )


def test_every_reason_field_names_a_value_field_that_exists():
    # Catches a typo'd sibling: `foo_reason` guarding nothing at all would
    # otherwise look like a satisfied invariant.
    for owner in (ConfoundingSensitivity, FalsificationResult, SensitivityReport):
        names = {f.name for f in dataclasses.fields(owner)}
        for name in names:
            if name.endswith("_reason"):
                assert name.removesuffix("_reason") in names, (owner, name)


def test_the_pinned_optional_slots_are_exactly_the_reason_carrying_fields():
    found = set()
    for owner in (ConfoundingSensitivity, FalsificationResult, SensitivityReport):
        names = {f.name for f in dataclasses.fields(owner)}
        for name in sorted(names):
            if name.endswith("_reason"):
                found.add((owner, name.removesuffix("_reason"), name))
    assert found == PINNED_OPTIONAL_SLOTS


def test_a_missing_value_without_a_reason_is_refused():
    with pytest.raises(ValueError, match="cinelli_hazlett_reason"):
        ConfoundingSensitivity(
            e_value=an_e_value(), cinelli_hazlett=None, cinelli_hazlett_reason=None
        )


def test_a_present_value_with_a_reason_is_refused():
    # The other half of the exclusive-or. A reason sitting next to a real
    # result is a stale reason, and a reader cannot tell which one to believe.
    result = CinelliHazlettResult(
        robustness_value=0.14,
        robustness_value_alpha=0.08,
        partial_r2_treatment_outcome=0.02,
        degrees_of_freedom=783,
    )
    with pytest.raises(ValueError, match="cinelli_hazlett_reason"):
        ConfoundingSensitivity(
            e_value=an_e_value(),
            cinelli_hazlett=result,
            cinelli_hazlett_reason="not applicable",
        )


def test_an_empty_reason_string_does_not_count_as_a_reason():
    with pytest.raises(ValueError, match="cinelli_hazlett_reason"):
        ConfoundingSensitivity(
            e_value=an_e_value(), cinelli_hazlett=None, cinelli_hazlett_reason=""
        )


def test_an_e_value_below_one_is_refused():
    # E = RR + sqrt(RR*(RR-1)) with RR >= 1 is at least 1 by construction, so
    # a value below it means the conversion or the formula is wrong.
    with pytest.raises(ValueError, match="at least 1"):
        EValueResult(
            e_value=0.8, e_value_ci_bound=1.0, approximated=False,
            approximation_note=None,
        )


def test_an_approximated_e_value_must_carry_its_note():
    with pytest.raises(ValueError, match="approximation_note"):
        EValueResult(
            e_value=2.0, e_value_ci_bound=1.5, approximated=True,
            approximation_note=None,
        )


def test_specification_point_is_not_hashable_but_constructs():
    # eq=False is what keeps this from raising TypeError on a Mapping field.
    point = SpecificationPoint(
        specification={"covariates": ("x1",), "outliers_excluded": False},
        point=0.1,
        ci=Interval(lower=0.0, upper=0.2),
    )
    assert point.specification["covariates"] == ("x1",)


def test_specification_curve_refuses_an_out_of_range_original_index():
    point = SpecificationPoint(
        specification={"covariates": ()}, point=0.1, ci=Interval(lower=0.0, upper=0.2)
    )
    with pytest.raises(ValueError, match="original_index"):
        SpecificationCurve(specifications=(point,), original_index=3, capped=False)


def test_placebo_test_result_holds_an_array_without_raising_on_equality():
    first, second = a_placebo(), a_placebo()
    # eq=False: identity semantics, so this compares False rather than raising
    # on the ambiguous truth value of an array.
    assert first != second
    assert first == first
