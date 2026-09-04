import numpy as np
from cp_causal.manifest import dataset_resolver, replay, run_analysis
from cp_core.dataset import CausalDataset, CausalSchema
from cp_core.manifest import DatasetReference, FixtureReference, ScenarioReference
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


def test_replay_refits_through_an_injected_resolver():
    dataset = build_scenario("D").generate(n_units=500, rng=np.random.default_rng(3)).dataset
    reference = ScenarioReference(scenario="D", seed=3, params={"n_units": 500})
    estimators = (("difference_in_means", EstimatorConfig(random_seed=1)),)
    manifest, original = run_analysis(dataset, reference, estimators)

    def fake_resolve(ref: DatasetReference, schema: CausalSchema) -> CausalDataset:
        assert ref == reference
        assert schema == dataset.schema
        return dataset

    replayed = replay(manifest, fake_resolve)

    assert len(replayed) == len(original)
    assert replayed[0].point == original[0].point
    assert replayed[0].ci == original[0].ci
    assert replayed[0].method == original[0].method


def test_dataset_resolver_dispatches_by_reference_type():
    calls = []
    stub = build_scenario("A").generate(n_units=100, rng=np.random.default_rng(0)).dataset

    def scenario_resolver(ref: ScenarioReference, schema: CausalSchema) -> CausalDataset:
        calls.append("scenario")
        return stub

    def fixture_resolver(ref: FixtureReference, schema: CausalSchema) -> CausalDataset:
        calls.append("fixture")
        return stub

    resolve = dataset_resolver(scenario_resolver, fixture_resolver)

    resolve(ScenarioReference(scenario="A", seed=0, params={"n_units": 100}), stub.schema)
    resolve(FixtureReference(path="x", content_sha256="y"), stub.schema)

    assert calls == ["scenario", "fixture"]
