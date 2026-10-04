"""The fixed set of analyses the study interprets, and how it is built.

Every case is a real fit: a simulated scenario or a real dataset, routed and
fitted by `cp_report.pipeline.analyse` exactly as `causalpulse analyze
--report` would, then flattened by `cp_llm.extract.fact_table`. An awkward
case also declares the property it is in the corpus for, and `build` refuses
to write it unless the fit shows that property: a declared property that the
data do not have would make the case describe something that is not there.

`build` writes each case's `facts.json` and a manifest of their hashes.
Everything after it reads the written corpus, never a fit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from cp_core.dataset import CausalDataset, CausalSchema, complete_cases
from cp_core.manifest import FixtureReference
from cp_llm.extract import fact_table
from cp_llm.facts import FACTS_FILENAME, FactTable
from cp_report.pipeline import analyse
from cp_simulation.donor_scenarios import DONOR_SCENARIOS, build_donor_scenario
from cp_simulation.panel_scenarios import PANEL_PROTOCOLS, build_panel_scenario
from cp_simulation.params import ScenarioParams
from cp_simulation.scenarios import build_scenario

Design = Literal["cross_section", "panel", "donor_pool"]
Group = Literal["scenario", "real_data", "awkward"]

MANIFEST_FILENAME = "corpus.json"
CORPUS_SCHEMA_VERSION = 1

# P1 to P6 at the panel arm's larger protocol: P3, P4 and P5 adopt too late
# for fewer than 7 periods. P7 has a protocol of its own (`PANEL_PROTOCOLS`).
PANEL_N_UNITS = 150
PANEL_N_PERIODS = 10
CROSS_SECTION_N_UNITS = 400
SEED = 0

LALONDE = "tests/fixtures/real_data/lalonde.csv"
PROP99 = "tests/fixtures/real_data/prop99.csv"


class CorpusPropertyError(ValueError):
    """An awkward case's fit does not show the property it was declared for."""


@dataclass(frozen=True)
class Requirement:
    """A property an awkward case exists for, checked on its fact table."""

    description: str
    holds: Callable[[FactTable], bool]


def _statuses(facts: FactTable) -> list[str]:
    return [str(f.value) for f in facts if f.id.startswith("check.") and f.id.endswith(".status")]


def _agreements(facts: FactTable) -> list[str]:
    return [str(f.value) for f in facts if f.id.startswith("cmp.") and f.id.endswith(".agreement")]


def _count(facts: FactTable, fact_id: str) -> int:
    fact = facts.get(fact_id)
    return int(fact.value) if fact is not None else 0


FAIL_STATUS = Requirement("at least one check with status FAIL", lambda f: "FAIL" in _statuses(f))
DIVERGENT = Requirement(
    "a comparison whose verdict is divergent", lambda f: "divergent" in _agreements(f)
)
SINGLE_ESTIMATE = Requirement(
    "exactly one estimate", lambda f: _count(f, "analysis.n_estimates") == 1
)


Maker = Callable[[Path], CausalDataset]


@dataclass(frozen=True)
class Case:
    case_id: str
    design: Design
    group: Group
    reason: str
    make: Maker
    estimators: tuple[str, ...] | None = None
    requirements: tuple[Requirement, ...] = ()


def _cross_section(
    scenario_id: str, strength: float = 1.0, n_units: int = CROSS_SECTION_N_UNITS
) -> Maker:
    def make(_root: Path) -> CausalDataset:
        params = ScenarioParams(confounding_strength=strength)
        rng = np.random.default_rng(SEED)
        return build_scenario(scenario_id, params).generate(n_units, rng).dataset

    return make


def _complete_case_h(_root: Path) -> CausalDataset:
    dataset = build_scenario("H", ScenarioParams()).generate(
        CROSS_SECTION_N_UNITS, np.random.default_rng(SEED)
    ).dataset
    complete, _record = complete_cases(dataset)
    return complete


def _panel(scenario_id: str) -> Maker:
    n_units, n_periods = PANEL_PROTOCOLS.get(scenario_id, (PANEL_N_UNITS, PANEL_N_PERIODS))

    def make(_root: Path) -> CausalDataset:
        return build_panel_scenario(scenario_id).generate(
            n_units=n_units, n_periods=n_periods, rng=np.random.default_rng(SEED)
        ).dataset

    return make


def _donor(scenario_id: str) -> Maker:
    spec = DONOR_SCENARIOS[scenario_id]

    def make(_root: Path) -> CausalDataset:
        return build_donor_scenario(scenario_id).generate(
            n_units=spec.n_units, n_periods=spec.n_periods, rng=np.random.default_rng(SEED)
        ).dataset

    return make


def _lalonde(root: Path) -> CausalDataset:
    schema = CausalSchema(
        unit="unit_id",
        treatment="treat",
        outcome="re78",
        covariates=("age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"),
        outcome_type="continuous",
    )
    return CausalDataset(frame=pd.read_csv(root / LALONDE), schema=schema)


def _prop99(root: Path) -> CausalDataset:
    schema = CausalSchema(
        unit="state",
        treatment="treatment",
        outcome="cigsale",
        covariates=(),
        outcome_type="continuous",
        time="year",
    )
    return CausalDataset(frame=pd.read_csv(root / PROP99), schema=schema)


def default_cases() -> tuple[Case, ...]:
    """The corpus of the design (§5.1), in the order it is reported.

    No fit-failure case: no router default on any scenario, at any size tried,
    or on either real dataset, failed to fit on unmodified data (measured
    2026-10-04), and a failure induced by patching an estimator would describe
    no real analysis.
    """
    cases: list[Case] = []
    for scenario_id in ("A", "B", "C", "D", "F", "G"):
        cases.append(
            Case(
                f"cross-{scenario_id}",
                "cross_section",
                "scenario",
                f"cross-sectional scenario {scenario_id}",
                _cross_section(scenario_id),
            )
        )
    cases.append(
        Case(
            "cross-H-complete",
            "cross_section",
            "scenario",
            "scenario H reduced to complete cases; as generated it is refused",
            _complete_case_h,
        )
    )
    for scenario_id in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        cases.append(
            Case(
                f"panel-{scenario_id}",
                "panel",
                "scenario",
                f"panel scenario {scenario_id}",
                _panel(scenario_id),
            )
        )
    for scenario_id in ("S0", "S1", "S2", "S3", "S4", "S5"):
        cases.append(
            Case(
                f"donor-{scenario_id}",
                "donor_pool",
                "scenario",
                f"donor-pool scenario {scenario_id}",
                _donor(scenario_id),
            )
        )
    cases.append(
        Case("real-lalonde", "cross_section", "real_data", "LaLonde (NSW and PSID)", _lalonde)
    )
    cases.append(
        Case("real-prop99", "donor_pool", "real_data", "California Proposition 99", _prop99)
    )
    cases.append(
        Case(
            "awkward-fail-divergent",
            "cross_section",
            "awkward",
            "strong confounding: a failed check and methods that diverge",
            _cross_section("D", strength=3.0, n_units=2000),
            requirements=(FAIL_STATUS, DIVERGENT),
        )
    )
    cases.append(
        Case(
            "awkward-divergent",
            "cross_section",
            "awkward",
            "selection on untreated outcomes: methods that diverge",
            _cross_section("G", strength=3.0),
            requirements=(DIVERGENT,),
        )
    )
    cases.append(
        Case(
            "awkward-single",
            "cross_section",
            "awkward",
            "one method, so no comparison",
            _cross_section("D"),
            estimators=("difference_in_means",),
            requirements=(SINGLE_ESTIMATE,),
        )
    )
    return tuple(cases)


@dataclass(frozen=True)
class CorpusEntry:
    """A built case: its declaration as recorded, and its fact table."""

    case_id: str
    design: Design
    group: Group
    reason: str
    estimators: tuple[str, ...] | None
    requirements: tuple[str, ...]
    facts_sha256: str
    facts: FactTable


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_case(case: Case, root: Path) -> FactTable:
    """Fit `case` and return its fact table, or raise if a requirement fails."""
    dataset = case.make(root)
    reference = FixtureReference(path=f"corpus:{case.case_id}", content_sha256="0" * 64)
    facts = fact_table(analyse(dataset, reference, case.estimators, random_seed=SEED))
    missing = [r.description for r in case.requirements if not r.holds(facts)]
    if missing:
        raise CorpusPropertyError(
            f"{case.case_id}: the fit does not show {', '.join(missing)}; "
            "the case would describe a property its data do not have"
        )
    return facts


def build(
    cases: Sequence[Case],
    out: Path,
    root: Path,
    progress: Callable[[str], None] = lambda _message: None,
) -> tuple[CorpusEntry, ...]:
    """Fit every case, write its fact table under `out`, then the manifest.

    The manifest is written last, and only if every case was written, so a
    corpus directory with a manifest is a complete one.
    """
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("two cases share an id")
    entries: list[CorpusEntry] = []
    for case in cases:
        progress(f"fitting {case.case_id}")
        facts = build_case(case, root)
        raw = facts.to_json().encode("utf-8")
        case_dir = out / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / FACTS_FILENAME).write_bytes(raw)
        entries.append(
            CorpusEntry(
                case_id=case.case_id,
                design=case.design,
                group=case.group,
                reason=case.reason,
                estimators=case.estimators,
                requirements=tuple(r.description for r in case.requirements),
                facts_sha256=_sha256(raw),
                facts=facts,
            )
        )
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "cases": [
            {
                "case_id": e.case_id,
                "design": e.design,
                "group": e.group,
                "reason": e.reason,
                "estimators": list(e.estimators) if e.estimators is not None else None,
                "requirements": list(e.requirements),
                "facts_sha256": e.facts_sha256,
            }
            for e in entries
        ],
    }
    (out / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return tuple(entries)


def load(corpus: Path) -> tuple[CorpusEntry, ...]:
    """Read a built corpus, refusing any fact table whose bytes changed."""
    manifest = json.loads((corpus / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError(f"corpus schema_version is not {CORPUS_SCHEMA_VERSION}")
    entries = []
    for record in manifest["cases"]:
        raw = (corpus / record["case_id"] / FACTS_FILENAME).read_bytes()
        if _sha256(raw) != record["facts_sha256"]:
            raise ValueError(f"{record['case_id']}: {FACTS_FILENAME} does not match the manifest")
        estimators = record["estimators"]
        entries.append(
            CorpusEntry(
                case_id=record["case_id"],
                design=record["design"],
                group=record["group"],
                reason=record["reason"],
                estimators=tuple(estimators) if estimators is not None else None,
                requirements=tuple(record["requirements"]),
                facts_sha256=record["facts_sha256"],
                facts=FactTable.from_json(raw.decode("utf-8")),
            )
        )
    return tuple(entries)
