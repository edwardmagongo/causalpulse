"""A multi-method analysis run, named well enough to be replayed.

`AnalysisManifest` records what would need to be re-run to reproduce a set of
`EffectEstimate`s: the dataset (by reference, not by embedding it -- see
`DatasetReference`), the schema, and the estimators fitted against it. It holds
no algorithm of its own beyond serialisation and the fixture-reference
resolver, which is pure I/O -- the same reasoning that put `complete_cases` in
`cp_core.dataset` at Phase 11: a row-filter or a file-read is not an estimation
algorithm.

Resolving a `ScenarioReference` needs `cp_simulation.build_scenario`, and this
package is contractually forbidden from importing `cp_simulation` (see
`.importlinter`'s `estimators-cannot-see-truth` contract, which names `cp_core`
alongside `cp_statistics` and `cp_causal`). So this module defines only the
*shape* a scenario resolver must have (`Resolver`) and resolves
`FixtureReference` alone; `cp_simulation.resolve_scenario_reference` and
`cp_causal.manifest.dataset_resolver` complete the picture from the two
packages actually allowed to.

Serialisation is JSON, which constrains what a manifest can carry:
`CausalSchema.treatment_start` and `EstimatorConfig.options` are both typed to
hold arbitrary objects, and `to_json` raises `TypeError` from `json.dumps` if
either holds something JSON cannot represent. That is a deliberate limit rather
than an oversight -- a manifest that silently dropped or stringified a
configuration value would replay a different analysis than the one it names.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cp_core.dataset import CausalDataset, CausalSchema
from cp_core.values import EstimatorConfig

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScenarioReference:
    """Names a simulated dataset by how to regenerate it, not by its content.

    `params` supports exactly `"n_units"` (required) and
    `"confounding_strength"` (optional, defaulting to `build_scenario`'s own
    default of 1.0) -- the two arguments every `build_scenario(...).generate(...)`
    call in this project's test suite varies. A scenario built with a
    `negative_control`, `misclassification`, `missingness`, or `outcome_proxy`
    override cannot be reconstructed from a `ScenarioReference` alone;
    `resolve_scenario_reference` refuses rather than silently reconstructing a
    different dataset.
    """

    scenario: str
    seed: int
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FixtureReference:
    """Names a vendored dataset by path and content hash.

    Replay reads the file at `path` and refuses if its current content does not
    hash to `content_sha256` -- a fixture edited after a manifest was built must
    not be silently replayed against.
    """

    path: str
    content_sha256: str

    @classmethod
    def from_path(cls, path: str) -> FixtureReference:
        content = Path(path).read_bytes()
        return cls(path=path, content_sha256=hashlib.sha256(content).hexdigest())


DatasetReference = ScenarioReference | FixtureReference
Resolver = Callable[[DatasetReference, CausalSchema], CausalDataset]


def resolve_fixture_reference(reference: FixtureReference, schema: CausalSchema) -> CausalDataset:
    """Load a vendored CSV, refusing if it no longer matches its recorded hash."""
    path = Path(reference.path)
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != reference.content_sha256:
        raise ValueError(
            f"fixture at {reference.path!r} does not match its recorded content: "
            f"expected sha256 {reference.content_sha256}, got {actual}. "
            "The file changed after this reference was built."
        )
    frame = pd.read_csv(path)
    return CausalDataset(frame=frame, schema=schema)


# The `kind` tag is added and stripped only at the JSON boundary rather than
# living as a field on either reference class: a `kind` field would be a value
# every constructor call had to pass that is, in practice, fixed by the class.
def _reference_to_dict(reference: DatasetReference) -> dict[str, object]:
    if isinstance(reference, ScenarioReference):
        return {"kind": "scenario", **asdict(reference)}
    return {"kind": "fixture", **asdict(reference)}


def _reference_from_dict(data: Any) -> DatasetReference:
    kind = data["kind"]
    if kind == "scenario":
        return ScenarioReference(
            scenario=str(data["scenario"]),
            seed=int(data["seed"]),
            params=dict(data["params"]),
        )
    if kind == "fixture":
        return FixtureReference(path=str(data["path"]), content_sha256=str(data["content_sha256"]))
    raise ValueError(f"unknown DatasetReference kind {kind!r}")


def _config_from_dict(data: Any) -> EstimatorConfig:
    return EstimatorConfig(
        bootstrap_reps=int(data["bootstrap_reps"]),
        bootstrap_kind=data["bootstrap_kind"],
        ci_level=float(data["ci_level"]),
        random_seed=data["random_seed"],
        options=dict(data["options"]),
    )


# eq=False mirrors `FitMetadata`, and for the same reason its docstring gives:
# `frozen=True` with the default `eq=True` generates a `__hash__` over every
# field, and `package_versions` is an unhashable mapping. Comparisons in tests
# are therefore field-by-field, deliberately.
@dataclass(frozen=True, eq=False)
class AnalysisManifest:
    """What produced a set of `EffectEstimate`s, and how to reproduce it."""

    dataset: DatasetReference
    schema: CausalSchema
    estimators: tuple[tuple[str, EstimatorConfig], ...]
    package_versions: Mapping[str, str]
    git_commit: str | None
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.estimators:
            raise ValueError("AnalysisManifest needs at least one (estimator_name, config) pair")

    @classmethod
    def capture(
        cls,
        dataset: DatasetReference,
        schema: CausalSchema,
        estimators: Sequence[tuple[str, EstimatorConfig]],
    ) -> AnalysisManifest:
        # Reuses `FitMetadata`'s own version/commit capture rather than
        # duplicating it. Imported inside the method, not at module scope,
        # because `cp_core.results` imports nothing from here but this module
        # would otherwise sit above it in import order for no reason.
        from cp_core.results import _git_commit, _package_versions

        return cls(
            dataset=dataset,
            schema=schema,
            estimators=tuple(estimators),
            package_versions=_package_versions(),
            git_commit=_git_commit(),
            created_at=datetime.now(UTC),
        )

    def to_json(self) -> str:
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset": _reference_to_dict(self.dataset),
            "schema": asdict(self.schema),
            "estimators": [
                {"name": name, "config": asdict(config)} for name, config in self.estimators
            ],
            "package_versions": dict(self.package_versions),
            "git_commit": self.git_commit,
            "created_at": self.created_at.isoformat(),
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> AnalysisManifest:
        payload = json.loads(raw)
        if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"AnalysisManifest schema_version {payload['schema_version']!r} is not "
                f"the {MANIFEST_SCHEMA_VERSION} this code reads"
            )
        # JSON has no tuple type, so every tuple-typed field comes back as a
        # list and must be converted or the round-tripped schema compares
        # unequal to the original. `covariates` and `auxiliary_columns` are
        # `CausalSchema`'s only two such fields as of Phase 13; a third would
        # need the same treatment here.
        schema_data = dict(payload["schema"])
        schema_data["covariates"] = tuple(schema_data["covariates"])
        schema_data["auxiliary_columns"] = tuple(schema_data.get("auxiliary_columns", ()))
        schema = CausalSchema(**schema_data)
        estimators = tuple(
            (str(entry["name"]), _config_from_dict(entry["config"]))
            for entry in payload["estimators"]
        )
        return cls(
            dataset=_reference_from_dict(payload["dataset"]),
            schema=schema,
            estimators=estimators,
            package_versions=dict(payload["package_versions"]),
            git_commit=payload["git_commit"],
            created_at=datetime.fromisoformat(payload["created_at"]),
        )
