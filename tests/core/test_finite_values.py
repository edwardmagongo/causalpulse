"""A dataset holding an infinite value is refused as a dataset.

A missing value is different, and stays constructible: `complete_cases` and
imputation take a dataset with gaps. Nothing in the engine can use an
infinity, and before this check one in a panel's time column was fitted.
"""

import numpy as np
import pandas as pd
import pytest
from cp_core.dataset import CausalDataset, CausalSchema


def frame() -> pd.DataFrame:
    """Two periods of four units, with a column for every schema role."""
    rows = []
    for unit in range(4):
        cohort = 2 if unit >= 2 else 0
        for period in (1, 2):
            rows.append(
                {
                    "unit_id": unit,
                    "period": period,
                    "cohort": cohort,
                    "treated": int(cohort > 0 and period >= cohort),
                    "outcome": float(unit + period),
                    "placebo": float(unit - period),
                    "x1": float(unit),
                    "x2": float(period),
                    "proxy": float(unit * period),
                }
            )
    return pd.DataFrame(rows)


SCHEMA = CausalSchema(
    unit="unit_id",
    treatment="treated",
    outcome="outcome",
    covariates=("x1", "x2"),
    outcome_type="continuous",
    time="period",
    cohort="cohort",
    negative_control_outcome="placebo",
    auxiliary_columns=("proxy",),
)
ROLES = ("unit_id", "treated", "outcome", "x1", "period", "cohort", "placebo", "proxy")


def with_value(column: str, value: float) -> pd.DataFrame:
    changed = frame()
    changed[column] = changed[column].astype(float)
    changed.loc[0, column] = value
    return changed


def test_the_unchanged_frame_constructs():
    # Without this, every refusal below could be the frame's own fault.
    CausalDataset(frame=frame(), schema=SCHEMA)


@pytest.mark.parametrize("column", ROLES)
def test_an_infinite_value_is_refused_in_every_role(column):
    with pytest.raises(ValueError, match="infinite values") as refusal:
        CausalDataset(frame=with_value(column, np.inf), schema=SCHEMA)
    assert f"columns ['{column}'] contain" in str(refusal.value)


@pytest.mark.parametrize("column", ["outcome", "x1"])
def test_a_negative_infinity_is_refused_too(column):
    with pytest.raises(ValueError, match="infinite values") as refusal:
        CausalDataset(frame=with_value(column, -np.inf), schema=SCHEMA)
    assert f"columns ['{column}'] contain" in str(refusal.value)


def test_every_offending_column_is_named_once_and_sorted():
    # `x2` is declared before `period` (covariates come before time), so the
    # order asserted here is the sorted one and not the schema's.
    changed = with_value("x2", np.inf)
    changed.loc[2, "x2"] = np.inf
    changed["period"] = changed["period"].astype(float)
    changed.loc[1, "period"] = -np.inf
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=changed, schema=SCHEMA)
    assert "columns ['period', 'x2'] contain infinite values" in str(refusal.value)


def test_the_message_is_the_one_the_design_gives():
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=with_value("x1", np.inf), schema=SCHEMA)
    assert str(refusal.value) == (
        "columns ['x1'] contain infinite values (inf or -inf), and no estimator "
        "here can use them. Correct those values or remove those rows, and "
        "supply the data again."
    )


def test_an_infinite_treatment_is_not_reported_as_non_binary():
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=with_value("treated", np.inf), schema=SCHEMA)
    assert "binary" not in str(refusal.value)


def test_the_refusal_does_not_offer_complete_cases():
    # `complete_cases` drops rows with missing values and would keep these.
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=with_value("outcome", np.inf), schema=SCHEMA)
    assert "complete_cases" not in str(refusal.value)


@pytest.mark.parametrize("column", ["outcome", "x1", "placebo", "proxy"])
def test_a_missing_value_still_constructs(column):
    dataset = CausalDataset(frame=with_value(column, np.nan), schema=SCHEMA)
    assert dataset.frame[column].isna().sum() == 1


def test_a_missing_value_beside_an_infinite_one_does_not_hide_it():
    changed = with_value("x1", np.nan)
    changed.loc[1, "x1"] = np.inf
    with pytest.raises(ValueError, match="infinite values"):
        CausalDataset(frame=changed, schema=SCHEMA)


def test_a_nullable_integer_column_with_a_gap_constructs():
    # A masked dtype marks a gap with `pd.NA`, not `nan`; it is still a gap.
    changed = frame()
    changed["x1"] = pd.array([1, None, 3, 4, 5, 6, 7, 8], dtype="Int64")
    CausalDataset(frame=changed, schema=SCHEMA)


def test_a_nullable_float_column_holding_an_infinity_is_refused():
    # The masked float dtype can hold both a gap and an infinity; the gap
    # must not hide the infinity.
    changed = frame()
    changed["x1"] = pd.array([1.0, None, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0], dtype="Float64")
    with pytest.raises(ValueError, match="infinite values") as refusal:
        CausalDataset(frame=changed, schema=SCHEMA)
    assert "columns ['x1'] contain" in str(refusal.value)


def test_a_text_unit_column_constructs():
    changed = frame()
    changed["unit_id"] = changed["unit_id"].map(lambda unit: f"unit-{unit}")
    CausalDataset(frame=changed, schema=SCHEMA)


def test_a_declared_auxiliary_column_absent_from_the_frame_is_not_required():
    CausalDataset(frame=frame().drop(columns="proxy"), schema=SCHEMA)


def test_an_infinite_value_in_an_undeclared_column_is_not_this_checks_business():
    changed = frame()
    changed["notes"] = np.inf
    CausalDataset(frame=changed, schema=SCHEMA)
