import numpy as np
import pytest
from cp_causal.propensity import POSITIVITY_THRESHOLD, standardised_mean_differences
from cp_causal.registry import get_estimator
from cp_core.values import EstimatorConfig
from cp_simulation.scenarios import build_scenario

CONFIG = EstimatorConfig(random_seed=0, bootstrap_reps=50)


def _fit(name, n_units=600):
    dataset = build_scenario("D").generate(
        n_units=n_units, rng=np.random.default_rng(0)
    ).dataset
    return dataset, get_estimator(name)().fit(dataset, CONFIG)


@pytest.mark.parametrize("name", ["ipw", "aipw"])
def test_it_exposes_the_scores_it_actually_weighted_with(name):
    dataset, estimate = _fit(name)

    scores = estimate.diagnostics["per_unit_propensity"].value

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(dataset.frame),)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("name", ["ipw", "aipw"])
def test_the_exposed_scores_are_the_trimmed_ones(name):
    # The figure must show the scores the estimate was built from. Untrimmed
    # scores would display an overlap problem the estimate did not face.
    _, estimate = _fit(name)

    scores = estimate.diagnostics["per_unit_propensity"].value

    assert scores.min() >= POSITIVITY_THRESHOLD
    assert scores.max() <= 1.0 - POSITIVITY_THRESHOLD


@pytest.mark.parametrize("name", ["ipw", "aipw"])
def test_it_exposes_one_standardised_mean_difference_per_covariate(name):
    dataset, estimate = _fit(name)
    n_covariates = len(dataset.schema.covariates)

    before = estimate.diagnostics["smd_before"].value
    after = estimate.diagnostics["smd_after"].value

    assert before.shape == (n_covariates,)
    assert after.shape == (n_covariates,)


@pytest.mark.parametrize("name", ["ipw", "aipw"])
def test_smd_before_is_the_unweighted_comparison(name):
    # Recomputed independently rather than compared against the estimator's
    # own value, which would only prove it is self-consistent.
    dataset, estimate = _fit(name)
    frame, schema = dataset.frame, dataset.schema
    covariates = frame[list(schema.covariates)].to_numpy(dtype=float)
    treatment = frame[schema.treatment].to_numpy(dtype=int)

    expected = standardised_mean_differences(covariates, treatment)

    assert np.allclose(estimate.diagnostics["smd_before"].value, expected)


@pytest.mark.parametrize("name", ["ipw", "aipw"])
def test_weighting_reduces_imbalance_on_a_confounded_scenario(name):
    # Scenario D is confounded by construction, so the weighted comparison
    # should be better balanced than the unweighted one. If this ever fails,
    # the two arrays have been swapped.
    _, estimate = _fit(name)

    before = np.abs(estimate.diagnostics["smd_before"].value).max()
    after = np.abs(estimate.diagnostics["smd_after"].value).max()

    assert after < before


@pytest.mark.parametrize("name", ["ipw", "aipw"])
def test_every_new_diagnostic_carries_an_interpretation(name):
    # A diagnostic with no interpretation is a number a reader cannot use.
    _, estimate = _fit(name)

    for key in ("per_unit_propensity", "smd_before", "smd_after"):
        assert estimate.diagnostics[key].interpretation.strip()


def test_the_existing_diagnostics_are_unchanged():
    # This task is additive. If a key disappeared, something else broke.
    _, estimate = _fit("ipw")

    assert {"effective_sample_size", "max_weight", "n_trimmed"} <= set(estimate.diagnostics)
