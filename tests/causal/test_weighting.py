import numpy as np
import pytest
from cp_causal.estimators.weighting import InverseProbabilityWeighting
from cp_causal.propensity import POSITIVITY_THRESHOLD
from cp_core.dataset import CausalDataset
from cp_core.values import EstimatorConfig
from cp_simulation.oracle import naive_difference_in_means
from cp_simulation.scenarios import build_scenario


def config(**kw: object) -> EstimatorConfig:
    defaults: dict[str, object] = {"bootstrap_reps": 100, "random_seed": 1}
    defaults.update(kw)
    return EstimatorConfig(**defaults)  # type: ignore[arg-type]


def test_removes_confounding_bias_the_naive_estimator_retains():
    # The whole point of the method. Verified independently: over 30 seeds the
    # naive bias is about +0.131 and IPW's about -0.002.
    estimates = []
    naive = []
    for seed in range(15):
        simulated = build_scenario("D").generate(
            n_units=4000, rng=np.random.default_rng(seed)
        )
        estimates.append(
            InverseProbabilityWeighting().fit(simulated.dataset, config()).point
        )
        naive.append(naive_difference_in_means(simulated.dataset))

    assert abs(float(np.mean(estimates)) - 0.05) < 0.01
    assert float(np.mean(naive)) - 0.05 > 0.10


def test_recovers_the_effect_under_randomisation_too():
    # A single seed at n=4000 has an estimator SD of about 0.0156, so a
    # one-seed assertion at any tight tolerance is a coin flip on the seed,
    # not a test of the estimator: seed 0 alone draws 0.0121, which is 2.4 SD
    # below the true ATE and fails a +/-0.03 check by construction (~5.5% of
    # seeds do). Average over seeds instead, as the other tests in this file
    # already do, so the assertion targets the estimator's bias, not luck.
    estimates = []
    for seed in range(10):
        simulated = build_scenario("A").generate(
            n_units=4000, rng=np.random.default_rng(seed)
        )
        estimates.append(
            InverseProbabilityWeighting().fit(simulated.dataset, config()).point
        )

    assert float(np.mean(estimates)) == pytest.approx(0.05, abs=0.02)


def test_handles_selection_on_the_baseline_outcome():
    estimates = [
        InverseProbabilityWeighting()
        .fit(
            build_scenario("G").generate(n_units=4000, rng=np.random.default_rng(s)).dataset,
            config(),
        )
        .point
        for s in range(10)
    ]

    assert abs(float(np.mean(estimates)) - 0.05) < 0.015


def test_reports_overlap_and_balance_assumptions():
    simulated = build_scenario("D").generate(n_units=2000, rng=np.random.default_rng(0))

    result = InverseProbabilityWeighting().fit(simulated.dataset, config())
    names = {a.name for a in result.assumptions}

    assert {"positivity", "covariate_balance", "unconfoundedness"} <= names


def test_reports_effective_sample_size_below_the_row_count():
    simulated = build_scenario("D").generate(n_units=2000, rng=np.random.default_rng(0))

    result = InverseProbabilityWeighting().fit(simulated.dataset, config())
    ess = result.diagnostics["effective_sample_size"].value

    assert 0 < float(ess) <= 2000


def test_severe_non_overlap_downgrades_the_assumptions_without_refusing():
    # The estimator still returns a number; it marks the assumption instead.
    # Refusing to answer is not the contract — reporting honestly is.
    simulated = build_scenario("D", confounding_strength=8.0).generate(
        n_units=3000, rng=np.random.default_rng(2)
    )

    result = InverseProbabilityWeighting().fit(simulated.dataset, config())

    assert result.point is not None
    assert result.worst_assumption_status in {"WARN", "FAIL"}


def test_reports_both_analytic_and_bootstrap_intervals():
    simulated = build_scenario("D").generate(n_units=2000, rng=np.random.default_rng(0))

    result = InverseProbabilityWeighting().fit(simulated.dataset, config())

    assert result.se is not None
    assert result.bootstrap_ci is not None


def test_analytic_standard_error_agrees_with_the_bootstrap():
    # This pins the influence-function normaliser. Using the mean weight within
    # an arm instead of E[T*w] over the whole sample halves the standard error
    # — plausible-looking, and it would silently destroy interval coverage.
    # The bootstrap does not share that mistake, so the two disagreeing by
    # roughly a factor of two is the signal.
    simulated = build_scenario("D").generate(n_units=4000, rng=np.random.default_rng(0))

    result = InverseProbabilityWeighting().fit(
        simulated.dataset, config(bootstrap_reps=400, random_seed=3)
    )

    assert result.se is not None and result.bootstrap_ci is not None
    bootstrap_se = result.bootstrap_ci.width / (2 * 1.959963984540054)
    assert 0.75 < result.se / bootstrap_se < 1.35


def test_requires_covariates():
    assert InverseProbabilityWeighting().requirements.needs_covariates is True


def poor_overlap() -> CausalDataset:
    """Scenario G at a fixed seed: overlap is bad by construction, so the
    trimming threshold has something to bite on."""
    return build_scenario("G").generate(n_units=1500, rng=np.random.default_rng(0)).dataset


def test_ipw_reads_a_trim_threshold_from_options():
    dataset = poor_overlap()
    loose = InverseProbabilityWeighting().fit(
        dataset, config(bootstrap_reps=25, options={"trim": 0.001})
    )
    tight = InverseProbabilityWeighting().fit(
        dataset, config(bootstrap_reps=25, options={"trim": 0.10})
    )
    # Measured on this exact frame: 0 units clipped at trim=0.001 against 137
    # at trim=0.10, with the propensity scores spanning 0.0042 to 0.983. A
    # tighter clip must bind on strictly more units; equality would mean the
    # option is being ignored, which is exactly what both estimators did
    # before it was wired through.
    assert tight.diagnostics["n_trimmed"].value > loose.diagnostics["n_trimmed"].value


def test_ipw_defaults_to_the_positivity_threshold():
    dataset = poor_overlap()
    default = InverseProbabilityWeighting().fit(dataset, config(bootstrap_reps=25))
    explicit = InverseProbabilityWeighting().fit(
        dataset, config(bootstrap_reps=25, options={"trim": POSITIVITY_THRESHOLD})
    )
    assert default.point == pytest.approx(explicit.point)


@pytest.mark.parametrize("bad", [-0.1, 0.5, 0.9, "0.01"])
def test_ipw_refuses_an_unusable_trim(bad):
    dataset = build_scenario("A").generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset
    with pytest.raises((ValueError, TypeError), match="trim"):
        InverseProbabilityWeighting().fit(
            dataset, config(bootstrap_reps=25, options={"trim": bad})
        )
