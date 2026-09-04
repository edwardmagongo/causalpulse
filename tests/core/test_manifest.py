import pytest
from cp_core.dataset import CausalSchema
from cp_core.manifest import (
    AnalysisManifest,
    FixtureReference,
    ScenarioReference,
)
from cp_core.values import EstimatorConfig


def _schema() -> CausalSchema:
    return CausalSchema(
        unit="unit_id",
        treatment="treat",
        outcome="y",
        covariates=("x1", "x2"),
        outcome_type="continuous",
    )


def test_manifest_rejects_an_empty_estimator_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        AnalysisManifest.capture(
            dataset=ScenarioReference(scenario="A", seed=0, params={"n_units": 100}),
            schema=_schema(),
            estimators=(),
        )


def test_manifest_capture_records_versions_and_commit() -> None:
    manifest = AnalysisManifest.capture(
        dataset=ScenarioReference(scenario="A", seed=0, params={"n_units": 100}),
        schema=_schema(),
        estimators=(("difference_in_means", EstimatorConfig()),),
    )
    assert manifest.schema == _schema()
    assert manifest.estimators == (("difference_in_means", EstimatorConfig()),)
    assert isinstance(manifest.package_versions, dict) and manifest.package_versions
    # git_commit may be None outside a git checkout, but must not raise.


def test_manifest_json_round_trip_with_a_scenario_reference() -> None:
    original = AnalysisManifest.capture(
        dataset=ScenarioReference(
            scenario="D", seed=7, params={"n_units": 500, "confounding_strength": 1.5}
        ),
        schema=_schema(),
        estimators=(
            ("difference_in_means", EstimatorConfig()),
            ("aipw", EstimatorConfig(bootstrap_reps=200, random_seed=3)),
        ),
    )
    restored = AnalysisManifest.from_json(original.to_json())
    assert restored.dataset == original.dataset
    assert restored.schema == original.schema
    assert restored.estimators == original.estimators
    assert restored.package_versions == original.package_versions
    assert restored.git_commit == original.git_commit
    assert restored.created_at == original.created_at


def test_manifest_json_round_trip_with_a_fixture_reference() -> None:
    original = AnalysisManifest.capture(
        dataset=FixtureReference(
            path="tests/fixtures/real_data/lalonde.csv", content_sha256="a" * 64
        ),
        schema=_schema(),
        estimators=(("ipw", EstimatorConfig()),),
    )
    restored = AnalysisManifest.from_json(original.to_json())
    assert restored.dataset == original.dataset


def test_from_json_refuses_an_unknown_schema_version() -> None:
    manifest = AnalysisManifest.capture(
        dataset=ScenarioReference(scenario="A", seed=0, params={"n_units": 50}),
        schema=_schema(),
        estimators=(("difference_in_means", EstimatorConfig()),),
    )
    tampered = manifest.to_json().replace('"schema_version": 1', '"schema_version": 99')
    with pytest.raises(ValueError, match="schema_version"):
        AnalysisManifest.from_json(tampered)
