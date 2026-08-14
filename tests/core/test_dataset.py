import numpy as np
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


@pytest.mark.parametrize("bad_value", ["Binary", "BINARY", "count", ""])
def test_schema_rejects_an_unknown_outcome_type(bad_value: str):
    # The Literal annotation is erased at runtime: without this check a schema
    # reading "Binary" from a config file would skip the binary-value
    # validation entirely and hand estimators an unvalidated outcome column.
    with pytest.raises(ValueError, match="outcome_type"):
        CausalSchema(
            unit="user_id",
            time=None,
            treatment="treated",
            outcome="converted",
            covariates=("age",),
            outcome_type=bad_value,  # type: ignore[arg-type]
        )


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


def test_dataset_rejects_treatment_with_missing_values():
    frame = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "period": [0, 1, 0, 1],
            "treated": [np.nan, np.nan, 0, 1],
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
    with pytest.raises(ValueError, match="missing"):
        CausalDataset(frame=frame, schema=schema)


def panel_frame(n_units=4, n_periods=4, start=3, never=2):
    rows = []
    for unit in range(n_units):
        cohort = 0 if unit < never else start
        for period in range(1, n_periods + 1):
            rows.append(
                {
                    "unit_id": unit,
                    "period": period,
                    "x1": float(unit),
                    "treated": int(cohort > 0 and period >= cohort),
                    "outcome": float(unit + period),
                }
            )
    return pd.DataFrame(rows)


PANEL_SCHEMA = CausalSchema(
    unit="unit_id",
    treatment="treated",
    outcome="outcome",
    covariates=("x1",),
    outcome_type="continuous",
    time="period",
)


def test_panel_helpers_report_periods_and_balance():
    dataset = CausalDataset(frame=panel_frame(), schema=PANEL_SCHEMA)

    assert list(dataset.periods) == [1, 2, 3, 4]
    assert dataset.n_periods == 4
    assert dataset.is_balanced


def test_cohort_by_unit_is_derived_from_treatment_not_from_a_column():
    # Derived from the data, so the helper works on real datasets that carry no
    # cohort column, and cannot disagree with the treatment it is supposed to
    # describe.
    dataset = CausalDataset(frame=panel_frame(), schema=PANEL_SCHEMA)

    cohorts = dataset.cohort_by_unit()

    assert cohorts.loc[0] == 0  # never treated
    assert cohorts.loc[1] == 0
    assert cohorts.loc[2] == 3
    assert cohorts.loc[3] == 3


def test_unbalanced_panel_is_reported_not_rejected():
    # Attrition is a legitimate real-data condition. It surfaces as a property
    # an estimator checks, not as a construction error.
    frame = panel_frame().drop(index=0)

    dataset = CausalDataset(frame=frame, schema=PANEL_SCHEMA)

    assert not dataset.is_balanced


def test_treatment_switching_off_is_reported_not_rejected():
    frame = panel_frame()
    frame.loc[(frame["unit_id"] == 2) & (frame["period"] == 4), "treated"] = 0

    dataset = CausalDataset(frame=frame, schema=PANEL_SCHEMA)

    assert not dataset.treatment_is_absorbing


def test_absorbing_treatment_is_the_normal_case():
    dataset = CausalDataset(frame=panel_frame(), schema=PANEL_SCHEMA)

    assert dataset.treatment_is_absorbing


def test_panel_rejects_a_non_orderable_time_column():
    frame = panel_frame()
    frame["period"] = frame["period"].astype(str) + "-label"

    with pytest.raises(ValueError, match="orderable"):
        CausalDataset(frame=frame, schema=PANEL_SCHEMA)


def test_panel_rejects_missing_values_in_the_time_column():
    frame = panel_frame()
    frame.loc[0, "period"] = None

    with pytest.raises(ValueError, match="missing values"):
        CausalDataset(frame=frame, schema=PANEL_SCHEMA)


def test_panel_helpers_refuse_a_cross_section():
    dataset = CausalDataset(frame=cross_section_frame(), schema=cross_section_schema())

    with pytest.raises(ValueError, match="not a panel"):
        _ = dataset.periods


def test_declared_pre_period_minimums_are_enforced_not_decorative():
    from cp_causal.base import Requirements

    dataset = CausalDataset(frame=panel_frame(start=2), schema=PANEL_SCHEMA)

    unmet = Requirements(needs_panel=True, min_pre_periods=3).unmet(dataset)

    assert any("pre-treatment period" in reason for reason in unmet)
    assert Requirements(needs_panel=True, min_pre_periods=1).unmet(dataset) == ()
