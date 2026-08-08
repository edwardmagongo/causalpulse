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


def test_bootstrap_ci_is_finite_with_only_two_treated_units():
    # Regression test: with 2 treated units, a resample missing both occurs
    # with probability ~= e^-2 (~13.5%) per replicate, so across the default
    # 200 replicates at least one degenerate draw is a near-certainty. Before
    # the fix this produced Interval(lower=nan, upper=nan) silently.
    rng = np.random.default_rng(42)
    n_treated, n_control = 2, 38
    n = n_treated + n_control
    w = np.array([1] * n_treated + [0] * n_control)
    y = rng.normal(size=n)
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

    assert result.bootstrap_ci is not None
    assert np.isfinite(result.bootstrap_ci.lower)
    assert np.isfinite(result.bootstrap_ci.upper)


def test_bootstrap_interval_reflects_a_large_unambiguous_effect():
    # Guards against a shuffled or mis-cast resample index: with a huge,
    # unambiguous effect and low noise, a bootstrap that drew the wrong rows
    # would produce an interval that either misses the true effect or fails
    # to exclude zero.
    rng = np.random.default_rng(13)
    n = 2000
    w = rng.binomial(1, 0.5, n)
    y = np.where(w == 1, 10.0, 0.0) + rng.normal(scale=0.5, size=n)
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

    assert result.bootstrap_ci is not None
    assert result.bootstrap_ci.contains(10.0)
    assert not result.bootstrap_ci.contains(0.0)
    assert result.bootstrap_ci.width < 1.0


def test_different_bootstrap_seeds_produce_different_intervals():
    # Pins that the seed is genuinely wired through bootstrap_effect rather
    # than silently ignored.
    dataset = make_dataset()
    first = DifferenceInMeans().fit(dataset, config(random_seed=1))
    second = DifferenceInMeans().fit(dataset, config(random_seed=2))

    assert first.bootstrap_ci != second.bootstrap_ci


def test_bootstrap_degenerate_replicates_diagnostic_is_present_and_non_negative():
    result = DifferenceInMeans().fit(make_dataset(), config())

    diagnostic = result.diagnostics["bootstrap_degenerate_replicates"]
    assert diagnostic.value >= 0.0


def test_bootstrap_degenerate_replicates_is_nonzero_with_only_two_treated_units():
    # Mirrors test_bootstrap_ci_is_finite_with_only_two_treated_units: with 2
    # treated units, each replicate degenerates (misses one arm entirely)
    # with probability ~= e^-2 (~13.5%), so across 200 replicates a nonzero
    # count is effectively certain.
    rng = np.random.default_rng(42)
    n_treated, n_control = 2, 38
    n = n_treated + n_control
    w = np.array([1] * n_treated + [0] * n_control)
    y = rng.normal(size=n)
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

    diagnostic = result.diagnostics["bootstrap_degenerate_replicates"]
    assert diagnostic.value > 0.0


def test_counterfactual_band_widens_as_ci_level_increases():
    dataset = make_dataset()
    result_95 = DifferenceInMeans().fit(dataset, config(ci_level=0.95))
    result_99 = DifferenceInMeans().fit(dataset, config(ci_level=0.99))
    assert result_95.counterfactual is not None
    assert result_99.counterfactual is not None

    width_95 = result_95.counterfactual.upper - result_95.counterfactual.lower
    width_99 = result_99.counterfactual.upper - result_99.counterfactual.lower

    assert width_99 > width_95
