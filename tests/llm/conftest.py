"""Real fits for `tests/llm/`.

`outcome` is scenario D at n=400 fitted by `difference_in_means` and `ipw`:
two methods, so there is a comparison; and `ipw`'s positivity check warns on
this seed, so the completeness rule has a check it must demand. A hand-built
`AnalysisOutcome` would only show the fact table agrees with the fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
from cp_core.dataset import CausalDataset
from cp_core.manifest import FixtureReference
from cp_llm.extract import fact_table
from cp_llm.facts import FactTable
from cp_report.pipeline import AnalysisOutcome, analyse
from cp_simulation.params import ScenarioParams
from cp_simulation.scenarios import build_scenario

REFERENCE = FixtureReference(path="d.csv", content_sha256="0" * 64)


@pytest.fixture(scope="session")
def cross_section() -> CausalDataset:
    return build_scenario("D", ScenarioParams()).generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset


@pytest.fixture(scope="session")
def outcome(cross_section: CausalDataset) -> AnalysisOutcome:
    return analyse(cross_section, REFERENCE, ["difference_in_means", "ipw"])


@pytest.fixture(scope="session")
def facts(outcome: AnalysisOutcome) -> FactTable:
    return fact_table(outcome)
