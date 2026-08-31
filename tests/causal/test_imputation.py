"""Filling missing outcomes, and refusing to when the fill would be a fiction."""

import numpy as np
import pandas as pd
import pytest
from cp_causal.imputation import impute_once
from cp_core.dataset import CausalDataset, CausalSchema


def _dataset(*, missing_rows=(0, 1), auxiliary=False, single_class=False):
    n = 40
    rng = np.random.default_rng(0)
    outcome = np.zeros(n) if single_class else np.tile([0.0, 1.0], n // 2)
    frame = pd.DataFrame(
        {
            "unit_id": np.arange(n),
            "treated": np.tile([0.0, 1.0], n // 2),
            "x1": rng.normal(size=n),
            "outcome": outcome,
        }
    )
    if auxiliary:
        frame["outcome_proxy"] = frame["outcome"] + rng.normal(0.0, 0.3, size=n)
    frame.loc[list(missing_rows), "outcome"] = np.nan
    schema = CausalSchema(
        unit="unit_id",
        treatment="treated",
        outcome="outcome",
        covariates=("x1",),
        outcome_type="binary",
        auxiliary_columns=("outcome_proxy",) if auxiliary else (),
    )
    return CausalDataset(frame=frame, schema=schema)


def test_it_fills_every_gap_and_leaves_observed_values_alone():
    dataset = _dataset()
    filled = impute_once(dataset, np.random.default_rng(0), draw_parameters=False)
    assert not filled.frame["outcome"].isna().any()
    observed = ~dataset.frame["outcome"].isna()
    np.testing.assert_array_equal(
        filled.frame.loc[observed, "outcome"].to_numpy(),
        dataset.frame.loc[observed, "outcome"].to_numpy(),
    )


def test_the_filled_outcome_stays_binary():
    # A probability would keep the point estimate closer but break the
    # schema's own binary contract and every downstream check that rests on
    # it, so the fill is a Bernoulli draw rather than a predicted mean.
    filled = impute_once(_dataset(), np.random.default_rng(0), draw_parameters=False)
    assert set(np.unique(filled.frame["outcome"].to_numpy())) <= {0.0, 1.0}


def test_a_dataset_with_nothing_missing_comes_back_unchanged():
    dataset = _dataset(missing_rows=())
    filled = impute_once(dataset, np.random.default_rng(0), draw_parameters=False)
    assert filled is dataset


def test_drawing_parameters_changes_the_fill_across_calls():
    # Proper multiple imputation must vary the imputation model itself, not
    # only the Bernoulli draw. Two draws that differed only in the Bernoulli
    # would understate between-imputation variance.
    dataset = _dataset(missing_rows=tuple(range(0, 30)))
    a = impute_once(dataset, np.random.default_rng(1), draw_parameters=True)
    b = impute_once(dataset, np.random.default_rng(2), draw_parameters=True)
    assert not np.array_equal(
        a.frame["outcome"].to_numpy(), b.frame["outcome"].to_numpy()
    )


def test_it_refuses_when_a_declared_auxiliary_has_gaps():
    dataset = _dataset(auxiliary=True)
    frame = dataset.frame.copy()
    frame.loc[5, "outcome_proxy"] = np.nan
    broken = CausalDataset(frame=frame, schema=dataset.schema)
    with pytest.raises(ValueError, match="outcome_proxy"):
        impute_once(broken, np.random.default_rng(0), draw_parameters=False)


def test_it_refuses_when_the_observed_outcomes_are_all_one_class():
    # A logistic model cannot be fitted on one class, and filling every gap
    # with the only value ever seen would manufacture certainty.
    with pytest.raises(ValueError, match="one class"):
        impute_once(
            _dataset(single_class=True), np.random.default_rng(0), draw_parameters=False
        )


def test_the_auxiliary_enters_the_imputation_model():
    # With a near-deterministic proxy the fill should track it. This is what
    # distinguishes an imputation that uses the auxiliary from one that
    # silently ignores it -- the failure mode that would collapse the whole
    # phase's efficiency result into a no-op.
    dataset = _dataset(missing_rows=tuple(range(0, 30)), auxiliary=True)
    frame = dataset.frame.copy()
    frame["outcome_proxy"] = np.where(np.arange(len(frame)) % 2 == 0, -5.0, 5.0)
    steered = CausalDataset(frame=frame, schema=dataset.schema)
    filled = impute_once(steered, np.random.default_rng(0), draw_parameters=False)
    imputed = filled.frame.loc[list(range(0, 30)), "outcome"].to_numpy()
    proxy = frame.loc[list(range(0, 30)), "outcome_proxy"].to_numpy()
    assert float(np.corrcoef(imputed, proxy)[0, 1]) > 0.8
