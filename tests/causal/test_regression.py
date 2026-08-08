import numpy as np
import pandas as pd
import pytest
from cp_causal.estimators.regression import LinRegressionAdjustment
from cp_core.dataset import CausalDataset, CausalSchema
from cp_core.values import EstimatorConfig


def make_dataset(n: int = 3000, seed: int = 0, effect: float = 2.0) -> CausalDataset:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    w = rng.binomial(1, 0.5, n)
    y = 1.0 + 3.0 * x1 - 2.0 * x2 + effect * w + rng.normal(size=n)
    frame = pd.DataFrame(
        {"unit_id": np.arange(n), "treated": w, "outcome": y, "x1": x1, "x2": x2}
    )
    return CausalDataset(
        frame=frame,
        schema=CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=("x1", "x2"),
            outcome_type="continuous",
        ),
    )


def config() -> EstimatorConfig:
    return EstimatorConfig(bootstrap_reps=200, random_seed=1)


def test_recovers_a_known_effect():
    result = LinRegressionAdjustment().fit(make_dataset(), config())

    assert result.point == pytest.approx(2.0, abs=0.1)
    assert result.method == "lin_regression"


def test_adjustment_is_more_precise_than_the_unadjusted_difference():
    # Covariates explain most of the outcome variance here, so adjusting must
    # shrink the standard error substantially. If it does not, the design
    # matrix or the centring is wrong.
    dataset = make_dataset()
    _, treatment, outcome = (
        dataset.frame[["x1", "x2"]].to_numpy(),
        dataset.frame["treated"].to_numpy(),
        dataset.frame["outcome"].to_numpy(),
    )
    unadjusted_se = np.sqrt(
        outcome[treatment == 1].var(ddof=1) / (treatment == 1).sum()
        + outcome[treatment == 0].var(ddof=1) / (treatment == 0).sum()
    )

    result = LinRegressionAdjustment().fit(dataset, config())

    assert result.se is not None
    assert result.se < 0.4 * unadjusted_se


def test_covariates_are_centred_so_the_coefficient_is_the_ate():
    # Shifting a covariate by a constant must not move the estimate. Without
    # centring the treatment coefficient is the effect at X = 0, which moves.
    dataset = make_dataset()
    shifted = dataset.frame.copy()
    shifted["x1"] = shifted["x1"] + 100.0
    shifted_dataset = CausalDataset(frame=shifted, schema=dataset.schema)

    original = LinRegressionAdjustment().fit(dataset, config())
    moved = LinRegressionAdjustment().fit(shifted_dataset, config())

    assert moved.point == pytest.approx(original.point, abs=1e-8)


def test_reports_both_analytic_and_bootstrap_intervals():
    result = LinRegressionAdjustment().fit(make_dataset(), config())

    assert result.se is not None
    assert result.ci_basis == "analytic"
    assert result.bootstrap_ci is not None


def test_requires_covariates():
    assert LinRegressionAdjustment().requirements.needs_covariates is True


def test_linearity_is_flagged_as_an_assumption():
    result = LinRegressionAdjustment().fit(make_dataset(), config())
    names = {a.name for a in result.assumptions}

    assert "outcome_model_specification" in names
    assert "randomised_assignment" in names


def test_null_effect_is_not_reported_as_significant():
    result = LinRegressionAdjustment().fit(make_dataset(effect=0.0), config())

    assert result.is_significant is False
