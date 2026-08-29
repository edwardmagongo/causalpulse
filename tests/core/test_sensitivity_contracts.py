"""The sensitivity contract's absences are always explained.

The invariant under test is not "these fields exist" -- it is that no slot
can be None without a sibling saying why. That is the shape Phase 5 shipped
wrong, and it is worth a structural test rather than a spot check: the exact
set of optional slots is pinned here, so a new one added later without a
reason fails this file rather than shipping as a silent gap.
"""

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from cp_core.results import (
    CinelliHazlettResult,
    ConfoundingSensitivity,
    EValueResult,
    FalsificationResult,
    MisclassificationCurve,
    MisclassificationPoint,
    NegativeControlResult,
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
    (FalsificationResult, "negative_control", "negative_control_reason"),
    (SensitivityReport, "specification_curve", "specification_curve_reason"),
    (SensitivityReport, "misclassification", "misclassification_reason"),
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


def a_negative_control() -> NegativeControlResult:
    # excludes_null must agree with whether the interval contains zero: this
    # interval spans zero, so excludes_null is False.
    return NegativeControlResult(
        outcome_column="outcome_nc",
        estimate=0.004,
        ci=Interval(lower=-0.02, upper=0.03),
        excludes_null=False,
        estimator="lin_regression",
    )


def a_misclassification_curve() -> MisclassificationCurve:
    point = MisclassificationPoint(
        sensitivity=0.9,
        specificity=0.9,
        corrected_point=0.125,
        corrected_ci=Interval(lower=0.05, upper=0.20),
        attenuation_factor=0.8,
    )
    return MisclassificationCurve(
        observed_point=0.1, points=(point,), observed_treated_share=0.5
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
    # Matched on the field name, not on the shared "at least 1" phrasing: both
    # bounds raise with that phrase, so the looser regex passed whichever
    # branch fired and left the other one unverified.
    with pytest.raises(ValueError, match="e_value must be at least 1"):
        EValueResult(
            e_value=0.8, e_value_ci_bound=1.0, approximated=False,
            approximation_note=None,
        )


def test_an_e_value_ci_bound_below_one_is_refused():
    # The other branch of the same guard. e_value is valid here, so only the
    # CI-bound check can fire -- otherwise this would pass on the message
    # above.
    with pytest.raises(ValueError, match="e_value_ci_bound must be at least 1"):
        EValueResult(
            e_value=2.0, e_value_ci_bound=0.8, approximated=False,
            approximation_note=None,
        )


CINELLI_HAZLETT_UNIT_FIELDS = (
    "robustness_value",
    "robustness_value_alpha",
    "partial_r2_treatment_outcome",
)


def a_cinelli_hazlett(**overrides) -> CinelliHazlettResult:
    fields = {
        "robustness_value": 0.14,
        "robustness_value_alpha": 0.08,
        "partial_r2_treatment_outcome": 0.02,
        "degrees_of_freedom": 783,
    }
    return CinelliHazlettResult(**{**fields, **overrides})


@pytest.mark.parametrize("field", CINELLI_HAZLETT_UNIT_FIELDS)
@pytest.mark.parametrize("value", (-0.01, 1.01))
def test_a_cinelli_hazlett_share_outside_the_unit_interval_is_refused(field, value):
    # All three are shares of residual variation, so all three are bounded by
    # [0, 1] on both sides. Every field is exercised by name: a guard that
    # checked one of them twice and another not at all would still satisfy a
    # spot check on the first.
    with pytest.raises(ValueError, match=field):
        a_cinelli_hazlett(**{field: value})


def test_a_cinelli_hazlett_boundary_share_is_accepted():
    # The complement: 0 and 1 are inside the interval, so a strict comparison
    # in the guard would fail here rather than shipping.
    result = a_cinelli_hazlett(robustness_value=0.0, partial_r2_treatment_outcome=1.0)
    assert result.robustness_value == 0.0
    assert result.partial_r2_treatment_outcome == 1.0


@pytest.mark.parametrize("degrees_of_freedom", (0, -5))
def test_a_cinelli_hazlett_without_residual_degrees_of_freedom_is_refused(
    degrees_of_freedom,
):
    # The bounds are read through a t-statistic on df residual degrees of
    # freedom, and zero of them is not a small number -- it is no regression.
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        a_cinelli_hazlett(degrees_of_freedom=degrees_of_freedom)


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


def a_confounding_sensitivity() -> ConfoundingSensitivity:
    return ConfoundingSensitivity(e_value=an_e_value(), cinelli_hazlett=None,
                                  cinelli_hazlett_reason="not applicable")


def test_falsification_result_fully_populated_constructs():
    # All four results present, all four reasons None. This exercises the
    # "present value with reason=None" path of all four _require_reason calls.
    FalsificationResult(
        placebo_dates=a_placebo(),
        placebo_dates_reason=None,
        placebo_units=a_placebo(),
        placebo_units_reason=None,
        leave_one_donor_out=(0.1, 0.2),
        leave_one_donor_out_reason=None,
        randomization_inference=a_placebo(),
        randomization_inference_reason=None,
        negative_control=None,
        negative_control_reason="not applicable",
    )


def test_falsification_result_fully_absent_constructs():
    # All four results None, all four reasons non-empty. This exercises the
    # "absent value with non-empty reason" path of all four _require_reason calls.
    FalsificationResult(
        placebo_dates=None,
        placebo_dates_reason="not applicable",
        placebo_units=None,
        placebo_units_reason="not applicable",
        leave_one_donor_out=None,
        leave_one_donor_out_reason="not applicable",
        randomization_inference=None,
        randomization_inference_reason="not applicable",
        negative_control=None,
        negative_control_reason="not applicable",
    )


# Every FalsificationResult slot, paired with a factory for its "present"
# value. leave_one_donor_out is a bare tuple, not a dataclass, so it gets a
# lambda rather than a_placebo; note an empty tuple is falsy but is NOT the
# same as absent (None), so the factory must return a non-empty one.
_FALSIFICATION_SLOTS: tuple[tuple[str, Callable[[], object]], ...] = (
    ("placebo_dates", a_placebo),
    ("placebo_units", a_placebo),
    ("leave_one_donor_out", lambda: (0.1, 0.2)),
    ("randomization_inference", a_placebo),
    ("negative_control", a_negative_control),
)
_FALSIFICATION_SLOT_IDS = [slot for slot, _ in _FALSIFICATION_SLOTS]


def _absent_falsification_kwargs() -> dict[str, object]:
    # Every slot absent with a valid reason: the baseline each parametrized
    # case perturbs at exactly one slot, so __post_init__'s sequential checks
    # can only raise on the slot under test.
    kwargs: dict[str, object] = {}
    for slot, _ in _FALSIFICATION_SLOTS:
        kwargs[slot] = None
        kwargs[f"{slot}_reason"] = "not applicable"
    return kwargs


@pytest.mark.parametrize(
    ("slot", "make_present"), _FALSIFICATION_SLOTS, ids=_FALSIFICATION_SLOT_IDS
)
def test_falsification_result_slot_absent_without_reason_raises(
    slot: str, make_present: Callable[[], object]
):
    # Omitting the reason on an absent slot raises ValueError naming that
    # slot's own reason field, not a generic message -- this is what makes a
    # swapped-field-name bug in __post_init__ actually fail the test.
    kwargs = _absent_falsification_kwargs()
    kwargs[f"{slot}_reason"] = None
    with pytest.raises(ValueError, match=f"{slot}_reason"):
        FalsificationResult(**kwargs)


@pytest.mark.parametrize(
    ("slot", "make_present"), _FALSIFICATION_SLOTS, ids=_FALSIFICATION_SLOT_IDS
)
def test_falsification_result_slot_present_with_reason_raises(
    slot: str, make_present: Callable[[], object]
):
    # Supplying a reason alongside a present slot raises ValueError naming
    # that slot's own reason field, not a generic message.
    kwargs = _absent_falsification_kwargs()
    kwargs[slot] = make_present()
    kwargs[f"{slot}_reason"] = "some reason"
    with pytest.raises(ValueError, match=f"{slot}_reason"):
        FalsificationResult(**kwargs)


def test_sensitivity_report_with_specification_curve_constructs():
    # specification_curve present, reason None.
    spec_point = SpecificationPoint(
        specification={"covariates": ()}, point=0.1, ci=Interval(lower=0.0, upper=0.2)
    )
    SensitivityReport(
        confounding=a_confounding_sensitivity(),
        specification_curve=SpecificationCurve(
            specifications=(spec_point,), original_index=0, capped=False
        ),
        specification_curve_reason=None,
        falsification=FalsificationResult(
            placebo_dates=None,
            placebo_dates_reason="not applicable",
            placebo_units=None,
            placebo_units_reason="not applicable",
            leave_one_donor_out=None,
            leave_one_donor_out_reason="not applicable",
            randomization_inference=None,
            randomization_inference_reason="not applicable",
            negative_control=None,
            negative_control_reason="not applicable",
        ),
        misclassification=None,
        misclassification_reason="not applicable",
    )


def test_sensitivity_report_without_specification_curve_constructs():
    # specification_curve None, reason non-empty.
    SensitivityReport(
        confounding=a_confounding_sensitivity(),
        specification_curve=None,
        specification_curve_reason="not applicable",
        falsification=FalsificationResult(
            placebo_dates=None,
            placebo_dates_reason="not applicable",
            placebo_units=None,
            placebo_units_reason="not applicable",
            leave_one_donor_out=None,
            leave_one_donor_out_reason="not applicable",
            randomization_inference=None,
            randomization_inference_reason="not applicable",
            negative_control=None,
            negative_control_reason="not applicable",
        ),
        misclassification=None,
        misclassification_reason="not applicable",
    )


def test_sensitivity_report_absent_curve_without_reason_raises():
    # specification_curve None but reason is None -> should raise.
    with pytest.raises(ValueError, match="specification_curve_reason"):
        SensitivityReport(
            confounding=a_confounding_sensitivity(),
            specification_curve=None,
            specification_curve_reason=None,
            falsification=FalsificationResult(
                placebo_dates=None,
                placebo_dates_reason="not applicable",
                placebo_units=None,
                placebo_units_reason="not applicable",
                leave_one_donor_out=None,
                leave_one_donor_out_reason="not applicable",
                randomization_inference=None,
                randomization_inference_reason="not applicable",
                negative_control=None,
                negative_control_reason="not applicable",
            ),
            misclassification=None,
            misclassification_reason="not applicable",
        )


def test_sensitivity_report_present_curve_with_reason_raises():
    # specification_curve present but reason is non-None -> should raise.
    spec_point = SpecificationPoint(
        specification={"covariates": ()}, point=0.1, ci=Interval(lower=0.0, upper=0.2)
    )
    with pytest.raises(ValueError, match="specification_curve_reason"):
        SensitivityReport(
            confounding=a_confounding_sensitivity(),
            specification_curve=SpecificationCurve(
                specifications=(spec_point,), original_index=0, capped=False
            ),
            specification_curve_reason="some reason",
            falsification=FalsificationResult(
                placebo_dates=None,
                placebo_dates_reason="not applicable",
                placebo_units=None,
                placebo_units_reason="not applicable",
                leave_one_donor_out=None,
                leave_one_donor_out_reason="not applicable",
                randomization_inference=None,
                randomization_inference_reason="not applicable",
                negative_control=None,
                negative_control_reason="not applicable",
            ),
            misclassification=None,
            misclassification_reason="not applicable",
        )


def test_sensitivity_report_with_misclassification_constructs():
    # misclassification present, reason None -- the other slot's "present"
    # direction, mirroring the specification_curve pair above.
    SensitivityReport(
        confounding=a_confounding_sensitivity(),
        specification_curve=None,
        specification_curve_reason="not applicable",
        falsification=FalsificationResult(
            placebo_dates=None,
            placebo_dates_reason="not applicable",
            placebo_units=None,
            placebo_units_reason="not applicable",
            leave_one_donor_out=None,
            leave_one_donor_out_reason="not applicable",
            randomization_inference=None,
            randomization_inference_reason="not applicable",
            negative_control=None,
            negative_control_reason="not applicable",
        ),
        misclassification=a_misclassification_curve(),
        misclassification_reason=None,
    )


def test_sensitivity_report_without_misclassification_constructs():
    # misclassification None, reason non-empty.
    SensitivityReport(
        confounding=a_confounding_sensitivity(),
        specification_curve=None,
        specification_curve_reason="not applicable",
        falsification=FalsificationResult(
            placebo_dates=None,
            placebo_dates_reason="not applicable",
            placebo_units=None,
            placebo_units_reason="not applicable",
            leave_one_donor_out=None,
            leave_one_donor_out_reason="not applicable",
            randomization_inference=None,
            randomization_inference_reason="not applicable",
            negative_control=None,
            negative_control_reason="not applicable",
        ),
        misclassification=None,
        misclassification_reason="not applicable",
    )


def test_sensitivity_report_absent_misclassification_without_reason_raises():
    # misclassification None but reason is None -> should raise.
    with pytest.raises(ValueError, match="misclassification_reason"):
        SensitivityReport(
            confounding=a_confounding_sensitivity(),
            specification_curve=None,
            specification_curve_reason="not applicable",
            falsification=FalsificationResult(
                placebo_dates=None,
                placebo_dates_reason="not applicable",
                placebo_units=None,
                placebo_units_reason="not applicable",
                leave_one_donor_out=None,
                leave_one_donor_out_reason="not applicable",
                randomization_inference=None,
                randomization_inference_reason="not applicable",
                negative_control=None,
                negative_control_reason="not applicable",
            ),
            misclassification=None,
            misclassification_reason=None,
        )


def test_sensitivity_report_present_misclassification_with_reason_raises():
    # misclassification present but reason is non-None -> should raise.
    with pytest.raises(ValueError, match="misclassification_reason"):
        SensitivityReport(
            confounding=a_confounding_sensitivity(),
            specification_curve=None,
            specification_curve_reason="not applicable",
            falsification=FalsificationResult(
                placebo_dates=None,
                placebo_dates_reason="not applicable",
                placebo_units=None,
                placebo_units_reason="not applicable",
                leave_one_donor_out=None,
                leave_one_donor_out_reason="not applicable",
                randomization_inference=None,
                randomization_inference_reason="not applicable",
                negative_control=None,
                negative_control_reason="not applicable",
            ),
            misclassification=a_misclassification_curve(),
            misclassification_reason="some reason",
        )


def test_an_empty_negative_control_reason_string_does_not_count_as_a_reason():
    # Mirrors test_an_empty_reason_string_does_not_count_as_a_reason for the
    # negative_control slot: an empty string is falsy but not None, and the
    # guard must still refuse it as "no reason given".
    kwargs = _absent_falsification_kwargs()
    kwargs["negative_control_reason"] = ""
    with pytest.raises(ValueError, match="negative_control_reason"):
        FalsificationResult(**kwargs)


def test_an_empty_misclassification_reason_string_does_not_count_as_a_reason():
    with pytest.raises(ValueError, match="misclassification_reason"):
        SensitivityReport(
            confounding=a_confounding_sensitivity(),
            specification_curve=None,
            specification_curve_reason="not applicable",
            falsification=FalsificationResult(
                placebo_dates=None,
                placebo_dates_reason="not applicable",
                placebo_units=None,
                placebo_units_reason="not applicable",
                leave_one_donor_out=None,
                leave_one_donor_out_reason="not applicable",
                randomization_inference=None,
                randomization_inference_reason="not applicable",
                negative_control=None,
                negative_control_reason="not applicable",
            ),
            misclassification=None,
            misclassification_reason="",
        )
