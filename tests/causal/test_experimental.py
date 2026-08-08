import numpy as np
import pandas as pd
import pytest
from cp_causal.estimators.experimental import DifferenceInMeans
from cp_core.dataset import CausalDataset, CausalSchema
from cp_core.values import EstimatorConfig


def make_dataset(effect: float = 0.2, n: int = 2000, seed: int = 0) -> CausalDataset:
    rng = np.random.default_rng(seed)
    w = rng.binomial(1, 0.5, n)
    p = np.where(w == 1, 0.5 + effect, 0.5)
    y = rng.binomial(1, p)
    frame = pd.DataFrame(
        {"unit_id": np.arange(n), "treated": w, "outcome": y, "x1": rng.normal(size=n)}
    )
    return CausalDataset(
        frame=frame,
        schema=CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=("x1",),
            outcome_type="binary",
        ),
    )


def config(**kw: object) -> EstimatorConfig:
    defaults: dict[str, object] = {"bootstrap_reps": 200, "random_seed": 1}
    defaults.update(kw)
    return EstimatorConfig(**defaults)  # type: ignore[arg-type]


def test_recovers_a_known_effect():
    result = DifferenceInMeans().fit(make_dataset(effect=0.2), config())

    assert result.point == pytest.approx(0.2, abs=0.05)
    assert result.estimand == "ATE"
    assert result.method == "difference_in_means"


def test_reports_both_analytic_and_bootstrap_intervals():
    # The dual-uncertainty rule: an estimator reporting only one is incomplete.
    result = DifferenceInMeans().fit(make_dataset(), config())

    assert result.se is not None
    assert result.ci_basis == "analytic"
    assert result.bootstrap_ci is not None
    assert result.bootstrap_ci.lower < result.point < result.bootstrap_ci.upper


def test_binary_outcome_uses_the_newcombe_interval_and_pp_scale():
    result = DifferenceInMeans().fit(make_dataset(), config())

    assert result.scale == "percentage_point"
    assert result.diagnostics["interval_method"].interpretation.startswith("Newcombe")


def test_continuous_outcome_uses_welch_and_absolute_scale():
    rng = np.random.default_rng(3)
    n = 2000
    w = rng.binomial(1, 0.5, n)
    y = 10.0 + 2.0 * w + rng.normal(size=n)
    frame = pd.DataFrame({"unit_id": np.arange(n), "treated": w, "outcome": y})
    dataset = CausalDataset(
        frame=frame,
        schema=CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=(),
            outcome_type="continuous",
        ),
    )

    result = DifferenceInMeans().fit(dataset, config())

    assert result.point == pytest.approx(2.0, abs=0.15)
    assert result.scale == "absolute"


def test_counterfactual_reports_the_control_mean():
    dataset = make_dataset(effect=0.2)
    result = DifferenceInMeans().fit(dataset, config())
    frame = dataset.frame
    control_mean = frame[frame["treated"] == 0]["outcome"].mean()

    assert result.counterfactual is not None
    assert result.counterfactual.predicted_absent_treatment == pytest.approx(
        control_mean
    )


def test_randomisation_is_recorded_as_not_testable():
    # Nothing in the data proves assignment was random. Say so.
    result = DifferenceInMeans().fit(make_dataset(), config())
    names = {a.name: a for a in result.assumptions}

    assert names["randomised_assignment"].status == "NOT_TESTABLE"


def test_null_effect_is_not_reported_as_significant():
    result = DifferenceInMeans().fit(make_dataset(effect=0.0, n=4000), config())

    assert result.is_significant is False


def test_fit_is_deterministic_for_a_fixed_seed():
    dataset = make_dataset()
    first = DifferenceInMeans().fit(dataset, config())
    second = DifferenceInMeans().fit(dataset, config())

    assert first.bootstrap_ci == second.bootstrap_ci


def test_estimator_declares_no_covariate_requirement():
    assert DifferenceInMeans().requirements.needs_covariates is False
