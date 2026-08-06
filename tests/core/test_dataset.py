import pandas as pd
import pytest

from cp_core.dataset import CausalDataset, CausalSchema


def cross_section_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "treated": [1, 1, 0, 0],
            "converted": [1, 0, 0, 1],
            "age": [21.0, 35.0, 44.0, 29.0],
        }
    )


def cross_section_schema() -> CausalSchema:
    return CausalSchema(
        unit="user_id",
        time=None,
        treatment="treated",
        outcome="converted",
        covariates=("age",),
        outcome_type="binary",
    )


def test_cross_section_schema_is_not_panel():
    assert cross_section_schema().is_panel is False


def test_dataset_counts_units_by_arm():
    dataset = CausalDataset(frame=cross_section_frame(), schema=cross_section_schema())

    assert dataset.n_units == 4
    assert dataset.n_treated_units == 2
    assert dataset.n_control_units == 2


def test_dataset_rejects_missing_column():
    schema = CausalSchema(
        unit="user_id",
        time=None,
        treatment="treated",
        outcome="nonexistent",
        covariates=("age",),
        outcome_type="binary",
    )
    with pytest.raises(ValueError, match="nonexistent"):
        CausalDataset(frame=cross_section_frame(), schema=schema)


def test_dataset_rejects_non_binary_treatment():
    frame = cross_section_frame()
    frame["treated"] = [0, 1, 2, 0]
    with pytest.raises(ValueError, match="binary"):
        CausalDataset(frame=frame, schema=cross_section_schema())


def test_dataset_rejects_duplicate_unit_period_pairs():
    frame = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "period": [0, 0, 0, 1],
            "treated": [0, 1, 0, 1],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    schema = CausalSchema(
        unit="unit",
        time="period",
        treatment="treated",
        outcome="y",
        covariates=(),
        outcome_type="continuous",
    )
    with pytest.raises(ValueError, match="duplicate"):
        CausalDataset(frame=frame, schema=schema)


def test_dataset_rejects_binary_outcome_with_non_binary_values():
    frame = cross_section_frame()
    frame["converted"] = [0.3, 0.7, 1.0, 0.0]
    with pytest.raises(ValueError, match="outcome_type='binary'"):
        CausalDataset(frame=frame, schema=cross_section_schema())


def test_panel_dataset_is_panel():
    frame = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "period": [0, 1, 0, 1],
            "treated": [0, 1, 0, 0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    schema = CausalSchema(
        unit="unit",
        time="period",
        treatment="treated",
        outcome="y",
        covariates=(),
        outcome_type="continuous",
    )
    dataset = CausalDataset(frame=frame, schema=schema)

    assert dataset.schema.is_panel is True
    assert dataset.n_units == 2
    assert dataset.n_treated_units == 1
