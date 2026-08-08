import numpy as np
import pytest
from cp_causal.estimators.doubly_robust import AugmentedIPW
from cp_causal.estimators.weighting import InverseProbabilityWeighting
from cp_core.dataset import CausalDataset, CausalSchema
from cp_core.values import EstimatorConfig
from cp_simulation.scenarios import build_scenario


def config(**kw: object) -> EstimatorConfig:
    defaults: dict[str, object] = {"bootstrap_reps": 50, "random_seed": 1}
    defaults.update(kw)
    return EstimatorConfig(**defaults)  # type: ignore[arg-type]


def cripple_propensity_covariates(dataset: CausalDataset) -> CausalDataset:
    """Hide the x2 confounder from whichever model reads schema.covariates."""
    return CausalDataset(
        frame=dataset.frame,
        schema=CausalSchema(
            unit=dataset.schema.unit,
            treatment=dataset.schema.treatment,
            outcome=dataset.schema.outcome,
            covariates=("x1",),
            outcome_type=dataset.schema.outcome_type,
        ),
    )


def mean_estimate(
    estimator,
    make_dataset,
    seeds: int = 12,
    estimator_config: EstimatorConfig | None = None,
) -> float:
    used_config = config() if estimator_config is None else estimator_config
    return float(
        np.mean(
            [estimator.fit(make_dataset(s), used_config).point for s in range(seeds)]
        )
    )


def full(seed: int) -> CausalDataset:
    return build_scenario("D").generate(
        n_units=4000, rng=np.random.default_rng(seed)
    ).dataset


def crippled(seed: int) -> CausalDataset:
    return cripple_propensity_covariates(full(seed))


def test_recovers_the_effect_under_confounding():
    assert mean_estimate(AugmentedIPW(), full) == pytest.approx(0.05, abs=0.01)


def test_survives_a_misspecified_propensity_model_where_ipw_does_not():
    # The double-robustness claim, stated as a comparison rather than an
    # absolute: with the propensity model crippled the same way for both
    # estimators, AIPW must be markedly closer to truth than IPW.
    #
    # IPW has only one nuisance model, so narrowing `schema.covariates` (via
    # `cripple_propensity_covariates`) is the honest way to misspecify it.
    # AIPW has two; narrowing the schema would cripple both at once, so its
    # propensity model is crippled explicitly via `EstimatorConfig.options`
    # instead, leaving its outcome model on the full declared covariate set.
    ipw_bias = abs(mean_estimate(InverseProbabilityWeighting(), crippled) - 0.05)
    aipw_bias = abs(
        mean_estimate(
            AugmentedIPW(),
            full,
            estimator_config=config(options={"propensity_covariates": ("x1",)}),
        )
        - 0.05
    )

    assert ipw_bias > 0.008
    assert aipw_bias < 0.5 * ipw_bias


def test_uses_exactly_schema_covariates_by_default():
    # Regression guard for the semantics `_outcome_design` used to override:
    # with no options supplied, AIPW's nuisance models read exactly
    # `schema.covariates`, so an extra undeclared column in the frame must
    # not move the estimate at all.
    dataset = full(0)
    baseline = AugmentedIPW().fit(dataset, config()).point

    extended_frame = dataset.frame.copy()
    extended_frame["noise"] = np.arange(len(extended_frame), dtype=float)
    extended_dataset = CausalDataset(frame=extended_frame, schema=dataset.schema)
    extended = AugmentedIPW().fit(extended_dataset, config()).point

    assert extended == pytest.approx(baseline, abs=1e-12)


def test_cross_fitting_is_used_by_default():
    assert AugmentedIPW().cross_fit_folds >= 2


def test_fit_is_deterministic_for_a_fixed_seed():
    dataset = full(0)
    first = AugmentedIPW().fit(dataset, config())
    second = AugmentedIPW().fit(dataset, config())

    assert first.point == pytest.approx(second.point, abs=1e-12)


def test_reports_influence_function_standard_error_and_bootstrap():
    result = AugmentedIPW().fit(full(0), config())

    assert result.se is not None and result.se > 0
    assert result.ci_basis == "analytic"
    assert result.bootstrap_ci is not None


def test_reports_overlap_assumptions():
    result = AugmentedIPW().fit(full(0), config())
    names = {a.name for a in result.assumptions}

    assert {"positivity", "covariate_balance", "unconfoundedness"} <= names


def test_recovers_a_heterogeneous_scenario_average_effect():
    estimates = [
        AugmentedIPW()
        .fit(
            build_scenario("F").generate(
                n_units=4000, rng=np.random.default_rng(s)
            ).dataset,
            config(),
        )
        .point
        for s in range(10)
    ]

    assert float(np.mean(estimates)) == pytest.approx(0.038, abs=0.012)


def test_requires_covariates():
    assert AugmentedIPW().requirements.needs_covariates is True
