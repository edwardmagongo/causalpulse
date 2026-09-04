"""Fit several estimators on one dataset, and replay the result later.

`run_analysis` is the orchestration Phase 15's `analyze` CLI command will
eventually wrap; building it now is what makes `AnalysisManifest.to_json` and
`replay` (§9 of the parent spec) testable at all, rather than early CLI work.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from cp_core.dataset import CausalDataset, CausalSchema
from cp_core.manifest import (
    AnalysisManifest,
    DatasetReference,
    FixtureReference,
    Resolver,
    ScenarioReference,
    resolve_fixture_reference,
)
from cp_core.results import EffectEstimate
from cp_core.values import EstimatorConfig

from cp_causal.registry import get_estimator


def run_analysis(
    dataset: CausalDataset,
    reference: DatasetReference,
    estimators: Sequence[tuple[str, EstimatorConfig]],
) -> tuple[AnalysisManifest, tuple[EffectEstimate, ...]]:
    """Fit every named estimator against `dataset`, and record how to redo it.

    `dataset` and `reference` are both required rather than one derived from
    the other: the caller already has a `CausalDataset` in hand (from a
    scenario they just generated, or a fixture they just loaded), and
    `run_analysis` does not re-resolve `reference` to check it matches -- that
    would discard the caller's dataset for no benefit. The check that a
    reference actually reproduces the dataset it names happens at `replay`
    time instead, which resolves `reference` independently.
    """
    fits = tuple(get_estimator(name)().fit(dataset, config) for name, config in estimators)
    manifest = AnalysisManifest.capture(
        dataset=reference, schema=dataset.schema, estimators=estimators
    )
    return manifest, fits


def replay(manifest: AnalysisManifest, resolve: Resolver) -> tuple[EffectEstimate, ...]:
    """Reconstruct `manifest`'s dataset and refit every estimator it named."""
    dataset = resolve(manifest.dataset, manifest.schema)
    return tuple(
        get_estimator(name)().fit(dataset, config) for name, config in manifest.estimators
    )


def dataset_resolver(
    resolve_scenario: Callable[[ScenarioReference, CausalSchema], CausalDataset],
    resolve_fixture: Callable[
        [FixtureReference, CausalSchema], CausalDataset
    ] = resolve_fixture_reference,
) -> Resolver:
    """Combine a scenario resolver and a fixture resolver into one `Resolver`.

    Lives here, rather than as a bare dispatch inside `replay`, because this
    package cannot import the scenario resolver directly: it sits in the
    simulation package, which holds ground truth, and `.importlinter`'s
    `estimators-cannot-see-truth` contract names `cp_causal` alongside
    `cp_core` and `cp_statistics`. (`tests/test_boundaries.py` enforces the
    same rule by grepping this source tree for that package's name, so it is
    not written out here.) The caller supplies the resolver instead: a test,
    or eventually Phase 15's CLI, both of which are free to import both
    packages since neither is a member of the contract's `source_modules`.
    """

    def resolve(reference: DatasetReference, schema: CausalSchema) -> CausalDataset:
        if isinstance(reference, ScenarioReference):
            return resolve_scenario(reference, schema)
        return resolve_fixture(reference, schema)

    return resolve
