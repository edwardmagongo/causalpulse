"""Imputing outcomes that went missing, and comparing that to dropping them.

Complete-case analysis and imputation are two answers to the same question,
and neither is free: dropping incomplete rows is unbiased only when values are
missing completely at random, while imputing them trusts a model of the very
values nobody observed. This module builds both and reports them side by side
rather than picking one on the caller's behalf.

Imputation is not a repair for data missing not at random. Nothing here
recovers a value whose absence depends on the value itself; that is what the
tipping-point bounds in `sensitivity` are for.
"""

from __future__ import annotations

import numpy as np
from cp_core.dataset import CausalDataset
from sklearn.linear_model import LogisticRegression

MAX_RESAMPLE_ATTEMPTS = 20


def _imputation_design(dataset: CausalDataset) -> np.ndarray:
    """Treatment, covariates and any declared auxiliary, as one float matrix.

    The auxiliary is included here and nowhere else. That asymmetry is the
    whole point: a column the analysis model can also see carries no
    information the analysis model lacks, so imputation would gain nothing
    from it.
    """
    schema = dataset.schema
    frame = dataset.frame
    columns = [schema.treatment, *schema.covariates, *schema.auxiliary_columns]
    return np.column_stack([frame[column].to_numpy(dtype=float) for column in columns])


def impute_once(
    dataset: CausalDataset,
    rng: np.random.Generator,
    *,
    draw_parameters: bool,
) -> CausalDataset:
    """One completed copy, with every missing outcome drawn from a fitted model.

    `draw_parameters=True` bootstrap-resamples the observed rows before
    fitting, so each call sees a different imputation model. That is what
    makes a set of these draws proper multiple imputation rather than several
    reruns of one model's Bernoulli noise, and it is what puts real
    between-imputation variance into Rubin's rules.

    `draw_parameters=False` fits once at the point estimate and takes a single
    stochastic draw. That is single imputation, and it is offered so the cost
    of treating imputed values as observed can be measured rather than
    asserted. It draws a value rather than taking the most likely one, which
    would push every imputed unit to the majority outcome and bias the arm.
    """
    schema = dataset.schema
    frame = dataset.frame
    outcome = frame[schema.outcome].to_numpy(dtype=float)
    missing = np.isnan(outcome)
    if not missing.any():
        return dataset

    for column in schema.auxiliary_columns:
        if frame[column].isna().any():
            raise ValueError(
                f"auxiliary column {column!r} contains missing values. An "
                "auxiliary exists to carry information where the outcome does "
                "not, so there is no fallback for filling one."
            )

    observed = np.flatnonzero(~missing)
    if np.unique(outcome[observed]).size < 2:
        raise ValueError(
            "the observed outcomes are all one class, so no logistic model "
            "can be fitted; filling the gaps with the only value ever seen "
            "would manufacture certainty rather than impute"
        )

    design = _imputation_design(dataset)
    rows = observed
    if draw_parameters:
        # A resample can land on a single class. Redraw rather than fall back
        # to the full sample, which would silently turn this draw into the
        # single-imputation path and shrink between-imputation variance.
        for _ in range(MAX_RESAMPLE_ATTEMPTS):
            candidate = rng.choice(observed, size=observed.size, replace=True)
            if np.unique(outcome[candidate]).size >= 2:
                rows = candidate
                break
        else:
            raise ValueError(
                f"no bootstrap resample of the observed outcomes contained "
                f"both classes in {MAX_RESAMPLE_ATTEMPTS} attempts; the "
                "observed outcomes are too close to constant to impute from"
            )

    model = LogisticRegression(max_iter=1000).fit(design[rows], outcome[rows])
    # `predict_proba` orders its columns by `classes_`, which is sorted, so
    # column 1 is P(outcome = 1) given the guard above that both classes are
    # present. Asserted rather than assumed: a silent reversal here would
    # invert every imputed value.
    if not np.array_equal(model.classes_, np.array([0.0, 1.0])):
        raise ValueError(f"unexpected outcome classes {model.classes_!r}")
    probability = model.predict_proba(design[missing])[:, 1]

    filled = outcome.copy()
    filled[missing] = (rng.random(probability.size) < probability).astype(float)
    return CausalDataset(
        frame=frame.assign(**{schema.outcome: filled}), schema=schema
    )
