"""A value too large in magnitude to compute with is refused as a dataset.

Float64 holds up to about 1.8e308, and a variance or a sum of squares adds n
squares of the data. At 1e150 a square is 1e300, so up to about 1e8 rows stay
finite; above it the estimators failed with a library's own message naming
no column (measured 2026-10-14: at 1e155 in LaLonde's outcome all four
cross-sectional estimators failed; in a covariate, `aipw` did).
"""

import numpy as np
import pandas as pd
import pytest
from cp_core.dataset import MAGNITUDE_BOUND, CausalDataset, CausalSchema

SCHEMA = CausalSchema(
    unit="unit_id",
    treatment="treated",
    outcome="outcome",
    covariates=("x1",),
    outcome_type="continuous",
    negative_control_outcome="placebo",
    auxiliary_columns=("proxy",),
)


def frame(**first_row: float) -> pd.DataFrame:
    """Four units with every role filled; `first_row` overrides the first row.

    Every column is float: assigning 1e200 into an integer column makes it
    `object`, which other checks refuse first.
    """
    data = pd.DataFrame(
        {
            "unit_id": [1.0, 2.0, 3.0, 4.0],
            "treated": [0.0, 1.0, 0.0, 1.0],
            "outcome": [1.0, 2.0, 3.0, 4.0],
            "placebo": [0.5, 0.25, 0.75, 1.0],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "proxy": [0.0, 1.0, 0.0, 1.0],
        }
    )
    for column, value in first_row.items():
        data.loc[0, column] = value
    return data


def test_the_bound_is_1e150():
    assert MAGNITUDE_BOUND == 1e150


@pytest.mark.parametrize("value", [1e150, -1e150, 1e149])
def test_a_value_at_or_below_the_bound_constructs(value):
    CausalDataset(frame=frame(x1=value), schema=SCHEMA)


@pytest.mark.parametrize("column", ["x1", "outcome", "placebo", "proxy"])
@pytest.mark.parametrize("value", [1.0000001e150, -1e151, 1e300])
def test_a_value_above_the_bound_is_refused_naming_its_column(column, value):
    with pytest.raises(ValueError) as refusal:
        CausalDataset(frame=frame(**{column: value}), schema=SCHEMA)

    assert str(refusal.value) == (
        f"columns ['{column}'] hold values too large to compute with ({column}: {value!r}); "
        "every value must be at most 1e150 in magnitude. Check the column's units, or "
        "correct those values, and supply the data again."
    )


def test_every_column_over_the_bound_is_named_in_order():
    with pytest.raises(ValueError, match=r"columns \['outcome', 'x1'\]"):
        CausalDataset(frame=frame(x1=1e200, outcome=-1e200), schema=SCHEMA)


def test_a_huge_unit_is_a_label_and_is_not_refused():
    CausalDataset(frame=frame(unit_id=1e200), schema=SCHEMA)


def test_a_missing_value_is_not_compared():
    CausalDataset(frame=frame(x1=np.nan), schema=SCHEMA)


def test_an_infinity_is_still_refused_as_infinite():
    # The infinity check runs first and keeps its own message.
    with pytest.raises(ValueError, match="contain infinite values"):
        CausalDataset(frame=frame(x1=np.inf), schema=SCHEMA)


def test_a_huge_treatment_keeps_its_own_refusal():
    with pytest.raises(ValueError, match="must be binary"):
        CausalDataset(frame=frame(treated=1e200), schema=SCHEMA)
