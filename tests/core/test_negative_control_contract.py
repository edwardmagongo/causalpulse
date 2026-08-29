"""The negative-control outcome is declared on the schema, and validated there.

Declared rather than inferred, following `assignment_provenance`'s precedent:
which column is a negative control is a claim about how the data came to
exist, not something a heuristic can read off the values.
"""

import numpy as np
import pandas as pd
import pytest
from cp_core.dataset import CausalDataset, CausalSchema


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
