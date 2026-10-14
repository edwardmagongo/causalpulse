from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cp_causal.estimators.doubly_robust import AugmentedIPW
from cp_causal.estimators.weighting import InverseProbabilityWeighting
from cp_causal.propensity import POSITIVITY_THRESHOLD
from cp_core.dataset import CausalDataset, CausalSchema, schema_from_dict
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


def test_rejects_a_bare_string_for_a_covariates_option():
    # A bare string is iterable, so `tuple("x1")` silently becomes
    # `("x", "1")` instead of raising — that must be rejected explicitly
    # rather than left to surface as a confusing "column not present" error.
    with pytest.raises(ValueError, match="propensity_covariates"):
        AugmentedIPW().fit(
            full(0), config(options={"propensity_covariates": "x1"})
        )


def _with_declared_extras(dataset: CausalDataset) -> CausalDataset:
    """The same frame, with a proxy declared auxiliary and a negative control.

    Both columns exist in the frame, so the "not present" guard cannot catch
    them; only a role check can.
    """
    frame = dataset.frame.copy()
    rng = np.random.default_rng(0)
    frame["outcome_proxy"] = frame["outcome"] + rng.normal(0.0, 0.25, size=len(frame))
    frame["nco"] = rng.normal(size=len(frame))
    return CausalDataset(
        frame=frame,
        schema=CausalSchema(
            unit=dataset.schema.unit,
            treatment=dataset.schema.treatment,
            outcome=dataset.schema.outcome,
            covariates=dataset.schema.covariates,
            outcome_type=dataset.schema.outcome_type,
            auxiliary_columns=("outcome_proxy",),
            negative_control_outcome="nco",
        ),
    )


@pytest.mark.parametrize("option", ["propensity_covariates", "outcome_covariates"])
def test_no_option_can_hand_an_estimator_a_declared_auxiliary(option: str):
    # Keeping the auxiliary out of the
    # adjustment set is "a correctness requirement, not a convenience --
    # enforced rather than documented". The schema's own validation only
    # covers `covariates`; an auxiliary is by construction a descendant of
    # the outcome, so conditioning on it through an option is the exact bias
    # the declaration exists to prevent, and it must be refused by name.
    dataset = _with_declared_extras(full(0))
    with pytest.raises(ValueError, match="outcome_proxy"):
        AugmentedIPW().fit(
            dataset, config(options={option: ("x1", "x2", "outcome_proxy")})
        )


@pytest.mark.parametrize("option", ["propensity_covariates", "outcome_covariates"])
def test_no_option_can_hand_an_estimator_the_negative_control(option: str):
    # The same hole, one column over. A negative control outcome is chosen
    # precisely because the treatment cannot affect it; adjusting for it
    # would destroy the falsification test that reads it.
    dataset = _with_declared_extras(full(0))
    with pytest.raises(ValueError, match="nco"):
        AugmentedIPW().fit(dataset, config(options={option: ("x1", "x2", "nco")}))


def poor_overlap() -> CausalDataset:
    """Scenario G at a fixed seed: overlap is bad by construction, so the
    trimming threshold has something to bite on."""
    return build_scenario("G").generate(n_units=1500, rng=np.random.default_rng(0)).dataset


def test_aipw_reads_a_trim_threshold_from_options():
    dataset = poor_overlap()
    loose = AugmentedIPW().fit(dataset, config(bootstrap_reps=25, options={"trim": 0.001}))
    tight = AugmentedIPW().fit(dataset, config(bootstrap_reps=25, options={"trim": 0.10}))
    # Measured on this exact frame, through AIPW's own cross-fitted
    # propensity model: 0 units clipped at trim=0.001 against 134 at
    # trim=0.10 (41 at 0.05). These counts differ slightly from the IPW
    # counterpart's 0/39/137 because AIPW reads out-of-fold scores from a
    # 5-fold cross-fit rather than a single in-sample fit. A tighter clip
    # must bind on strictly more units; equality would mean the option is
    # being ignored.
    assert tight.diagnostics["n_trimmed"].value > loose.diagnostics["n_trimmed"].value
    # AIPW divides by `e` and `1 - e` directly, so the clip must also reach
    # the estimate. These two configs differ in nothing but `trim` and share
    # a seed, so the cross-fitting folds are identical; before the option was
    # wired through they returned bit-identical points (measured: both
    # 0.043878450040).
    assert loose.point != tight.point


def test_aipw_defaults_to_the_positivity_threshold():
    dataset = poor_overlap()
    default = AugmentedIPW().fit(dataset, config(bootstrap_reps=25))
    explicit = AugmentedIPW().fit(
        dataset, config(bootstrap_reps=25, options={"trim": POSITIVITY_THRESHOLD})
    )
    assert default.point == pytest.approx(explicit.point)


@pytest.mark.parametrize("bad", [-0.1, 0.5, 0.9, "0.01"])
def test_aipw_refuses_an_unusable_trim(bad):
    dataset = build_scenario("A").generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset
    with pytest.raises((ValueError, TypeError), match="trim"):
        AugmentedIPW().fit(dataset, config(bootstrap_reps=25, options={"trim": bad}))


REAL_DATA = Path(__file__).parents[1] / "fixtures" / "real_data"
LALONDE_SCHEMA = {
    "unit": "unit_id",
    "treatment": "treat",
    "outcome": "re78",
    "covariates": ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"],
    "outcome_type": "continuous",
}


@pytest.mark.slow
def test_aipw_on_lalonde_does_not_depend_on_the_units_of_earnings():
    # Measured before the fix: -3,682.62 [-4,325.22, -3,040.02] as given and
    # -5,961.60 [-6,586.43, -5,336.76] with re75 multiplied by a million.
    frame = pd.read_csv(REAL_DATA / "lalonde.csv")
    schema = schema_from_dict(LALONDE_SCHEMA)
    given = AugmentedIPW().fit(CausalDataset(frame=frame, schema=schema), EstimatorConfig())
    rescaled = AugmentedIPW().fit(
        CausalDataset(frame=frame.assign(re75=frame["re75"] * 1e6), schema=schema),
        EstimatorConfig(),
    )

    assert rescaled.point == pytest.approx(given.point, rel=1e-6)
    assert rescaled.ci.lower == pytest.approx(given.ci.lower, rel=1e-6)
    assert rescaled.ci.upper == pytest.approx(given.ci.upper, rel=1e-6)
