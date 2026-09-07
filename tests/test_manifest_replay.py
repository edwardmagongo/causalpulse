"""End-to-end proof that to_json -> from_json -> replay reproduces identical
numbers, against both kinds of DatasetReference.

Lives outside every package's own test directory because it is the one place
that imports both cp_simulation and cp_causal and combines their resolvers into
one `dataset_resolver`. Neither cp_core nor cp_causal may do that themselves --
`.importlinter`'s `estimators-cannot-see-truth` contract names both -- so the
composition has nowhere else to be proved. `tests/test_determinism.py` is the
existing precedent for a whole-system test living directly under `tests/`.
"""

import numpy as np
import pandas as pd
import pytest
from cp_causal.manifest import dataset_resolver, replay, run_analysis
from cp_core.dataset import CausalSchema
from cp_core.manifest import (
    AnalysisManifest,
    FixtureReference,
    ScenarioReference,
    resolve_fixture_reference,
)
from cp_core.values import EstimatorConfig
from cp_simulation.scenarios import build_scenario, resolve_scenario_reference

RESOLVE = dataset_resolver(resolve_scenario_reference)

TINY_SCHEMA = CausalSchema(
    unit="unit_id", treatment="treat", outcome="y", covariates=(), outcome_type="continuous"
)


def _assert_diagnostics_match(replayed, original):
    # Diagnostic.value is typed `float | np.ndarray` (per_unit_propensity,
    # smd_before, smd_after are arrays). Diagnostic keeps the dataclass
    # default eq=True since most diagnostics are scalar, but `==` on a pair
    # holding arrays returns an array rather than a bool, so a dict-level
    # `==` over `diagnostics` raises "truth value of an array is ambiguous"
    # instead of comparing. Compared field-by-field here instead.
    assert replayed.keys() == original.keys()
    for key, replayed_diag in replayed.items():
        original_diag = original[key]
        assert replayed_diag.name == original_diag.name, key
        assert replayed_diag.unit == original_diag.unit, key
        assert replayed_diag.interpretation == original_diag.interpretation, key
        assert np.array_equal(replayed_diag.value, original_diag.value), key


def _assert_estimates_match_except_provenance(replayed, original):
    # EffectEstimate and FitMetadata both declare eq=False (an unhashable
    # package_versions mapping forced the choice), so `==` on either is
    # identity, not structural -- every field is compared explicitly.
    for name in (
        "estimand",
        "method",
        "point",
        "se",
        "ci",
        "ci_basis",
        "bootstrap_ci",
        "p_value",
        "n_treated",
        "n_control",
        "scale",
        "assumptions",
        "counterfactual",
        "subgroup_findings",
    ):
        assert getattr(replayed, name) == getattr(original, name), name
    _assert_diagnostics_match(replayed.diagnostics, original.diagnostics)
    rmeta, ometa = replayed.fit_metadata, original.fit_metadata
    assert rmeta.estimator == ometa.estimator
    assert rmeta.package_versions == ometa.package_versions
    assert rmeta.git_commit == ometa.git_commit
    assert rmeta.config_hash == ometa.config_hash
    assert rmeta.platform == ometa.platform
    assert rmeta.random_seed == ometa.random_seed
    # `timestamp` and `runtime_seconds` are deliberately excluded: FitMetadata's
    # own docstring states they vary between two runs of the same fit.


def _write_tiny_csv(path):
    pd.DataFrame(
        {
            "unit_id": range(200),
            "treat": [i % 2 for i in range(200)],
            "y": np.random.default_rng(0).normal(size=200),
        }
    ).to_csv(path, index=False)
    return path


def test_replay_reproduces_a_scenario_based_manifest_bit_for_bit():
    dataset = build_scenario("D").generate(n_units=500, rng=np.random.default_rng(9)).dataset
    reference = ScenarioReference(scenario="D", seed=9, params={"n_units": 500})
    estimators = (
        ("difference_in_means", EstimatorConfig(random_seed=1)),
        ("ipw", EstimatorConfig(random_seed=1)),
    )
    manifest, original = run_analysis(dataset, reference, estimators)

    restored = AnalysisManifest.from_json(manifest.to_json())
    replayed = replay(restored, RESOLVE)

    assert len(replayed) == len(original)
    for r, o in zip(replayed, original, strict=True):
        _assert_estimates_match_except_provenance(r, o)


def test_replay_reproduces_a_fixture_based_manifest_bit_for_bit(tmp_path):
    path = _write_tiny_csv(tmp_path / "tiny.csv")
    reference = FixtureReference.from_path(str(path))
    dataset = resolve_fixture_reference(reference, TINY_SCHEMA)
    manifest, original = run_analysis(
        dataset, reference, (("difference_in_means", EstimatorConfig()),)
    )

    restored = AnalysisManifest.from_json(manifest.to_json())
    replayed = replay(restored, RESOLVE)

    _assert_estimates_match_except_provenance(replayed[0], original[0])


def test_replay_refuses_when_a_fixture_changed_since_the_manifest_was_built(tmp_path):
    path = _write_tiny_csv(tmp_path / "tiny.csv")
    reference = FixtureReference.from_path(str(path))
    dataset = resolve_fixture_reference(reference, TINY_SCHEMA)
    manifest, _ = run_analysis(
        dataset, reference, (("difference_in_means", EstimatorConfig()),)
    )

    path.write_text(path.read_text() + "200,1,0.5\n")

    with pytest.raises(ValueError, match="does not match its recorded content"):
        replay(manifest, RESOLVE)
