"""Structural refusals: duplicated labels, the unit column, and exact periods.

Each of these used to construct and then go wrong somewhere else: an infinity
unseen under a duplicated label, a crash while routing a panel with a missing
unit, a cohort truncated to a period that is not in the data.
"""

import numpy as np
import pandas as pd
import pytest
from cp_core.dataset import CausalDataset, CausalSchema
from cp_core.reading import read_csv


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
CROSS_SECTION = CausalSchema(
    unit="unit_id",
    treatment="treated",
    outcome="outcome",
    covariates=("x1", "x2"),
    outcome_type="continuous",
)
NAMED = ("unit_id", "treated", "outcome", "x1", "period", "cohort", "placebo", "proxy")


def cross_section() -> pd.DataFrame:
    return frame()[lambda rows: rows["period"] == 2].reset_index(drop=True)


def with_second_copy(column: str) -> pd.DataFrame:
    changed = frame()
    return pd.concat([changed, changed[[column]]], axis=1)


def with_periods(values) -> pd.DataFrame:
    """The frame with its two periods replaced by `values`, in order."""
    changed = frame()
    first, second = values
    changed["period"] = pd.Series(
        [first if period == 1 else second for period in changed["period"]],
        dtype=np.asarray(values).dtype,
    )
    changed["cohort"] = 0
    return changed


def test_the_unchanged_frame_constructs():
    # Without this, every refusal below could be the frame's own fault.
    CausalDataset(frame=frame(), schema=SCHEMA)
    CausalDataset(frame=cross_section(), schema=CROSS_SECTION)


@pytest.mark.parametrize("column", NAMED)
def test_a_duplicated_label_is_refused_for_every_column_the_schema_names(column):
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=with_second_copy(column), schema=SCHEMA)
    assert str(refusal.value) == (
        f"columns ['{column}'] appear more than once in the frame; each column "
        "the schema names must appear once."
    )


def test_an_infinity_under_a_duplicated_label_no_longer_goes_unseen():
    changed = frame()
    extra = pd.DataFrame({"x1": [np.inf] + [1.0] * 7}, index=changed.index)
    with pytest.raises(ValueError, match="appear more than once"):
        CausalDataset(frame=pd.concat([changed, extra], axis=1), schema=SCHEMA)


def test_two_duplicated_labels_are_named_together_and_sorted():
    changed = pd.concat([frame(), frame()[["x2", "outcome"]]], axis=1)
    with pytest.raises(ValueError, match=r"columns \['outcome', 'x2'\] appear more than once"):
        CausalDataset(frame=changed, schema=SCHEMA)


def test_a_duplicated_label_the_schema_does_not_name_is_left_alone():
    changed = frame()
    changed["notes"] = "a"
    CausalDataset(frame=pd.concat([changed, changed[["notes"]]], axis=1), schema=SCHEMA)


@pytest.mark.parametrize("build", [cross_section, frame], ids=["cross-section", "panel"])
def test_a_missing_unit_is_refused(build):
    changed = build()
    changed["unit_id"] = changed["unit_id"].astype(float)
    changed.loc[0, "unit_id"] = np.nan
    schema = CROSS_SECTION if build is cross_section else SCHEMA
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=changed, schema=schema)
    assert str(refusal.value) == (
        "unit column 'unit_id' contains missing values. pandas reads NA, N/A, "
        "n/a, NaN, null, NULL, None and an empty cell as missing, so a unit "
        "with one of those names has to be renamed."
    )


@pytest.mark.parametrize("spelling", ["NA", "N/A", "n/a", "NaN", "null", "NULL", "None", ""])
def test_each_spelling_the_message_lists_is_read_as_a_missing_unit(spelling):
    # The message tells the user which names to avoid; this holds it to what
    # the reader does.
    text = f"unit_id,treated,outcome,x1,x2\nalpha,1,1.0,0.1,0.2\n{spelling},0,2.0,0.3,0.4\n"
    read = read_csv(text.encode())
    assert read["unit_id"].isna().tolist() == [False, True]
    with pytest.raises(ValueError, match="unit column 'unit_id' contains missing values"):
        CausalDataset(frame=read, schema=CROSS_SECTION)


def test_a_unit_named_like_a_missing_value_but_not_read_as_one_constructs():
    text = "unit_id,treated,outcome,x1,x2\nalpha,1,1.0,0.1,0.2\nNAM,0,2.0,0.3,0.4\n"
    CausalDataset(frame=read_csv(text.encode()), schema=CROSS_SECTION)


def test_a_unit_column_mixing_numbers_and_text_is_refused():
    changed = cross_section()
    changed["unit_id"] = pd.Series([5, "5", 7, "abc"], dtype=object)
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=changed, schema=CROSS_SECTION)
    assert str(refusal.value) == (
        "unit column 'unit_id' mixes numbers and text, so its values cannot be "
        "ordered or compared. Make every unit a number or every unit text."
    )


@pytest.mark.parametrize(
    "units",
    [
        pd.Series(["a", "b", "c", "d"], dtype=object),
        pd.Series([1, 2, 3, 4], dtype=object),
        pd.Series([1, 2.5, 3, 4], dtype=object),
        pd.Series(["a", "b", "c", "d"], dtype="str"),
    ],
    ids=["object text", "object integers", "object integers and floats", "text"],
)
def test_a_unit_column_of_one_kind_constructs(units):
    changed = cross_section()
    changed["unit_id"] = units
    CausalDataset(frame=changed, schema=CROSS_SECTION)


@pytest.mark.parametrize(
    ("periods", "found"),
    [
        (np.array([1989.5, 1990.5]), "1989.5"),
        (np.array([1.0, 2.0**53 + 2]), "9007199254740994.0"),
        (np.array([1, 2**53 + 1], dtype=np.int64), "9007199254740993"),
        (np.array([1, 2**63], dtype=np.uint64), "9223372036854775808"),
        (np.array([-(2**53) - 1, 1], dtype=np.int64), "-9007199254740993"),
        (np.array([1.0, 1e300]), "1e+300"),
    ],
    ids=["fractional", "float above", "integer just above", "beyond int64", "below", "huge"],
)
def test_a_period_that_is_not_an_exact_whole_number_is_refused(periods, found):
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=with_periods(periods), schema=SCHEMA)
    assert str(refusal.value) == (
        f"time column 'period' must hold whole-numbered periods no larger than 2**53; "
        f"found {found}. Periods are used as integers, so a fractional or very large "
        "value would be changed."
    )


@pytest.mark.parametrize(
    "periods",
    [
        np.array([1970.0, 1971.0]),
        np.array([1.0, 2.0**53]),
        np.array([1, 2**53], dtype=np.int64),
        np.array([-3, 0], dtype=np.int64),
        np.array([1, 2], dtype=np.uint8),
    ],
    ids=[
        "whole-numbered floats", "the limit as a float", "the limit itself", "zero and below",
        "small unsigned",
    ],
)
def test_whole_numbered_periods_construct(periods):
    # Zero and negative periods are refused later, by the panel code, with
    # its own message; they are exact, so this rule lets them through.
    CausalDataset(frame=with_periods(periods), schema=SCHEMA)


def with_time_column(values) -> pd.DataFrame:
    """The frame with its period column replaced by `values`, one per row."""
    changed = frame().drop(columns="cohort")
    changed["period"] = values
    return changed


NO_COHORT = CausalSchema(
    unit="unit_id",
    treatment="treated",
    outcome="outcome",
    covariates=("x1", "x2"),
    outcome_type="continuous",
    time="period",
)


@pytest.mark.parametrize(
    ("make", "dtype"),
    [
        (lambda p: pd.to_datetime(p.map({1: "2024-01-01", 2: "2024-02-01"})), "datetime64"),
        (
            lambda p: pd.to_datetime(p.map({1: "2024-01-01", 2: "2024-02-01"})).dt.tz_localize(
                "UTC"
            ),
            "datetime64",
        ),
        (lambda p: pd.to_timedelta(p, unit="D"), "timedelta64"),
        (lambda p: p == 2, "bool"),
        (lambda p: p.astype(complex), "complex128"),
        (lambda p: p.astype(str) + "-label", "str"),
    ],
    ids=["dates", "dates with a time zone", "durations", "booleans", "complex", "text"],
)
def test_a_time_column_that_is_not_integer_or_float_is_refused(make, dtype):
    # Every panel estimator numbers its periods. A column of dates used to
    # construct and then be refused by each of them when fitted.
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=with_time_column(make(frame()["period"])), schema=NO_COHORT)
    message = str(refusal.value)
    assert message.startswith(
        f"time column 'period' must hold whole-numbered periods, and holds {dtype}"
    )
    assert message.endswith(
        "values. Number the periods (1, 2, 3, ... or the year) before building the dataset."
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda p: p,
        lambda p: p.astype(float),
        lambda p: p.astype("float32"),
        lambda p: p.astype("Int64"),
        lambda p: p.astype("uint8"),
        lambda p: p.astype("float16"),
        lambda p: p.astype("Float64"),
    ],
    ids=["int64", "float64", "float32", "nullable Int64", "uint8", "float16", "nullable Float64"],
)
def test_an_integer_or_float_time_column_constructs(make):
    CausalDataset(frame=with_time_column(make(frame()["period"])), schema=NO_COHORT)
