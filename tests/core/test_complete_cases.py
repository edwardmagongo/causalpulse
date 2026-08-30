"""Complete-case analysis, made explicit and recorded.

Dropping incomplete rows is unbiased under MCAR and generally biased under
MAR, so it is a modelling choice rather than a cleanup step. These tests pin
that the choice is visible: what went, and which columns were responsible.
"""

import numpy as np
import pandas as pd
import pytest
from cp_core.dataset import CausalDataset, CausalSchema, complete_cases


def schema(nc: str | None = None) -> CausalSchema:
    return CausalSchema(
        unit="unit_id",
        treatment="treated",
        outcome="outcome",
        covariates=("x1", "x2"),
        outcome_type="binary",
        negative_control_outcome=nc,
    )


def frame(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "unit_id": np.arange(n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "treated": rng.integers(0, 2, size=n),
            "outcome": rng.integers(0, 2, size=n).astype(float),
        }
    )


def test_a_complete_dataset_is_returned_unchanged():
    dataset = CausalDataset(frame=frame(), schema=schema())
    reduced, record = complete_cases(dataset)
    assert record.n_rows_before == record.n_rows_after == 10
    assert record.columns_with_missing == ()
    assert len(reduced.frame) == 10


def test_rows_with_a_missing_outcome_are_dropped_and_recorded():
    f = frame()
    f.loc[[2, 5, 7], "outcome"] = np.nan
    reduced, record = complete_cases(CausalDataset(frame=f, schema=schema()))
    assert record.n_rows_before == 10
    assert record.n_rows_after == 7
    assert record.columns_with_missing == ("outcome",)
    assert record.dropped_by_column["outcome"] == 3
    assert len(reduced.frame) == 7
    assert not reduced.frame["outcome"].isna().any()


def test_rows_with_a_missing_covariate_are_dropped():
    f = frame()
    f.loc[[1, 3], "x1"] = np.nan
    _, record = complete_cases(CausalDataset(frame=f, schema=schema()))
    assert record.columns_with_missing == ("x1",)
    assert record.n_rows_after == 8


def test_the_per_column_counts_may_overlap_and_need_not_sum_to_the_total():
    # Row 4 is missing in both columns. It is counted once per column but
    # dropped once, so the counts sum to 3 while only 2 rows went. The record
    # documents this rather than implying the counts partition the drop.
    f = frame()
    f.loc[[4], "outcome"] = np.nan
    f.loc[[4, 6], "x1"] = np.nan
    _, record = complete_cases(CausalDataset(frame=f, schema=schema()))
    assert record.dropped_by_column["outcome"] == 1
    assert record.dropped_by_column["x1"] == 2
    assert record.n_rows_before - record.n_rows_after == 2


def test_a_declared_negative_control_is_considered():
    # required_columns includes it, so a row missing only the negative control
    # is still incomplete. Pinned because this column joined required_columns
    # only one phase ago.
    f = frame()
    f["outcome_nc"] = np.zeros(10)
    f.loc[[8], "outcome_nc"] = np.nan
    _, record = complete_cases(
        CausalDataset(frame=f, schema=schema(nc="outcome_nc"))
    )
    assert record.columns_with_missing == ("outcome_nc",)
    assert record.n_rows_after == 9


def test_the_returned_dataset_keeps_the_same_schema():
    f = frame()
    f.loc[[2], "outcome"] = np.nan
    reduced, _ = complete_cases(CausalDataset(frame=f, schema=schema()))
    assert reduced.schema == schema()


def test_the_returned_frame_is_reindexed():
    # A caller doing positional work on the reduced frame would otherwise get
    # gaps where the dropped rows were.
    f = frame()
    f.loc[[0, 1], "outcome"] = np.nan
    reduced, _ = complete_cases(CausalDataset(frame=f, schema=schema()))
    assert list(reduced.frame.index) == list(range(8))


def test_dropping_everything_is_refused_rather_than_returning_an_empty_dataset():
    # An estimator handed zero rows fails far from the cause.
    f = frame()
    f["outcome"] = np.nan
    with pytest.raises(ValueError, match="no complete cases"):
        complete_cases(CausalDataset(frame=f, schema=schema()))
