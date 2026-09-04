import numpy as np
from cp_causal.manifest import run_analysis
from cp_core.manifest import ScenarioReference
from cp_core.values import EstimatorConfig
from cp_simulation.scenarios import build_scenario


def test_run_analysis_fits_every_requested_estimator_and_builds_a_matching_manifest():
    dataset = build_scenario("D").generate(n_units=500, rng=np.random.default_rng(3)).dataset
    reference = ScenarioReference(scenario="D", seed=3, params={"n_units": 500})
    estimators = (
        ("difference_in_means", EstimatorConfig(random_seed=1)),
        ("lin_regression", EstimatorConfig(random_seed=1)),
    )

    manifest, fits = run_analysis(dataset, reference, estimators)

    assert manifest.dataset == reference
    assert manifest.schema == dataset.schema
    assert manifest.estimators == estimators
    assert [fit.method for fit in fits] == ["difference_in_means", "lin_regression"]
    assert all(
        fit.fit_metadata.estimator == name
        for fit, (name, _) in zip(fits, estimators, strict=True)
    )
