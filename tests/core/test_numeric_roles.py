"""A column in a role that must be numeric is refused unless pandas read it as numbers.

`read_csv` and `float()` disagree about what a number is: `1_000`, or `inf`
followed by a space, makes pandas read a column as text (or, in a large file,
as a mix of numbers and text), and yet every cell converts when an estimator
asks for floats. Before this rule such a column was analysed as numbers, and an
infinity in it slipped past the finite check.
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
NUMERIC_ROLES = ("treated", "outcome", "x1", "cohort", "placebo", "proxy")


def as_text(column: str) -> pd.DataFrame:
    """The frame with `column` holding the same values as text."""
    changed = frame()
    changed[column] = changed[column].astype(str)
    return changed


def test_the_unchanged_frame_constructs():
    # Without this, every refusal below could be the frame's own fault.
    CausalDataset(frame=frame(), schema=SCHEMA)


@pytest.mark.parametrize("column", NUMERIC_ROLES)
def test_a_text_column_is_refused_in_every_numeric_role(column):
    with pytest.raises(ValueError, match="must hold numbers") as refusal:
        CausalDataset(frame=as_text(column), schema=SCHEMA)
    message = str(refusal.value)
    assert f"columns ['{column}'] must hold numbers" in message
    assert f"({column}: str)" in message


@pytest.mark.parametrize("column", ["outcome", "x1"])
def test_an_object_column_of_floats_is_refused(column):
    changed = frame()
    changed[column] = changed[column].astype(object)
    with pytest.raises(ValueError, match="must hold numbers") as refusal:
        CausalDataset(frame=changed, schema=SCHEMA)
    assert f"({column}: object)" in str(refusal.value)


def test_an_object_column_holding_an_infinity_is_refused_by_this_rule():
    # The case the finite check passes over: it looks only at numeric dtypes.
    changed = frame()
    changed["x1"] = changed["x1"].astype(object)
    changed.loc[0, "x1"] = np.inf
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=changed, schema=SCHEMA)
    assert "columns ['x1'] must hold numbers" in str(refusal.value)
    assert "infinite" not in str(refusal.value)


def test_every_offending_column_is_named_with_its_dtype_and_sorted():
    # `x2` is declared before `cohort` (covariates come before the cohort), so
    # the order asserted here is the sorted one and not the schema's.
    changed = as_text("x2")
    changed["cohort"] = changed["cohort"].astype(object)
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=changed, schema=SCHEMA)
    message = str(refusal.value)
    assert "columns ['cohort', 'x2'] must hold numbers" in message
    assert "(cohort: object, x2: str)" in message


def test_the_message_is_the_one_the_design_gives():
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=as_text("x1"), schema=SCHEMA)
    assert str(refusal.value) == (
        "columns ['x1'] must hold numbers, and were not read as numbers "
        "(x1: str). Every value has to be a plain number: look for stray text, "
        "a space after a value, or digit grouping such as 1_000 or 1,000."
    )


def test_a_text_treatment_is_not_reported_as_non_binary():
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=as_text("treated"), schema=SCHEMA)
    assert "binary" not in str(refusal.value)


def test_a_text_unit_column_constructs():
    changed = frame()
    changed["unit_id"] = changed["unit_id"].map(lambda unit: f"unit-{unit}")
    CausalDataset(frame=changed, schema=SCHEMA)


def test_a_time_column_of_dates_is_not_this_rules_business():
    # The time column has its own rule, which refuses dates with its own
    # message; this rule must not claim it as a column that must hold numbers.
    changed = frame().drop(columns="cohort")
    changed["period"] = pd.to_datetime(changed["period"].map({1: "2024-01-01", 2: "2024-02-01"}))
    schema = CausalSchema(
        unit="unit_id",
        treatment="treated",
        outcome="outcome",
        covariates=("x1", "x2"),
        outcome_type="continuous",
        time="period",
    )
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=changed, schema=schema)
    assert str(refusal.value).startswith("time column 'period' must hold whole-numbered periods")
    assert "must hold numbers" not in str(refusal.value)


def test_a_boolean_treatment_constructs():
    changed = frame()
    changed["treated"] = changed["treated"].astype(bool)
    CausalDataset(frame=changed, schema=SCHEMA)


def test_a_nullable_integer_covariate_with_a_gap_constructs():
    changed = frame()
    changed["x1"] = pd.array([1, None, 3, 4, 5, 6, 7, 8], dtype="Int64")
    CausalDataset(frame=changed, schema=SCHEMA)


def test_a_numeric_outcome_with_a_missing_value_constructs():
    changed = frame()
    changed.loc[0, "outcome"] = np.nan
    dataset = CausalDataset(frame=changed, schema=SCHEMA)
    assert dataset.frame["outcome"].isna().sum() == 1


def test_a_declared_auxiliary_column_absent_from_the_frame_is_not_required():
    CausalDataset(frame=frame().drop(columns="proxy"), schema=SCHEMA)


def test_a_text_column_the_schema_does_not_name_is_not_this_rules_business():
    changed = frame()
    changed["notes"] = "free text"
    CausalDataset(frame=changed, schema=SCHEMA)
