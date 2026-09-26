"""Hand-built `BenchmarkEvidence` and a tiny real grid, shared by the benchmark report tests.

Plain functions, not fixtures, for the same reason `report.builders` gives:
call sites pass their own overrides inline.
"""

from __future__ import annotations

import pandas as pd
from cp_evaluation.expectations import ExpectationResult, Measurement
from cp_evaluation.grid import CrossSectionAxes, GridConfig, PanelAxes
from cp_evaluation.store import RunStatus
from cp_report.benchmark import BenchmarkEvidence
from cp_report.evidence import ReportFigure


def result(
    id="E1",
    statement="difference_in_means's bias on scenario A is within its tolerance band.",
    verdict="met",
    note="",
) -> ExpectationResult:
    return ExpectationResult(
        id=id,
        statement=statement,
        rule="measured within +/- 4 mcse of the recorded band",
        verdict=verdict,
        measurements=(
            Measurement(label="difference_in_means on A: bias", value=0.01, mcse=0.005, n_used=4),
        ),
        note=note,
    )


def draw(ax):
    ax.plot([0, 1], [0, 1])
    return ax


def tables() -> dict[str, pd.DataFrame]:
    return {
        "accuracy__scenario": pd.DataFrame({"method": ["difference_in_means"], "bias": [0.01]}),
        "coverage__scenario": pd.DataFrame(
            {"method": ["difference_in_means"], "coverage_analytic": [0.94]}
        ),
        # Present in the full set but not in REPORT_TABLES: proves curation filters.
        "runtime__scenario": pd.DataFrame(
            {"method": ["difference_in_means"], "runtime_seconds": [0.2]}
        ),
    }


def make_benchmark_evidence() -> BenchmarkEvidence:
    """The smallest instance that exercises every field the two entry points read."""
    return BenchmarkEvidence(
        expectations=(
            result("E1", verdict="met"),
            result(
                "E5",
                statement=(
                    "hierarchical_did's credible-interval coverage is not below 0.90, pooled."
                ),
                verdict="failed",
                note=(
                    "hierarchical_did's coverage is low on two panel scenarios; "
                    "see the design doc."
                ),
            ),
            result(
                "E7",
                statement="No estimator is more than 10x slower than the fastest.",
                verdict="inconclusive",
            ),
        ),
        tables=tables(),
        status=RunStatus(expected=6, done=6, orphans=0, by_scenario={"A": (4, 4), "P1": (2, 2)}),
        replications=2,
        scenarios_run=("A", "P1"),
        estimators_run=("difference_in_means", "lin_regression"),
        # Ends with a period on purpose. The real `NOT_COVERED` reasons do not, and
        # `causal_language_lint` only splits sentences on `.!?`, so an unpunctuated
        # fragment merges with the next one for negation and hedge purposes (design
        # doc §7.1). This fixture avoids that so the lint-refusal tests exercise this
        # module's own plumbing, not that separate defect.
        not_covered=(("donor_pool", "Not yet implemented."),),
        figures=(ReportFigure(caption="bias by method and scenario.", draw=draw),),
        limitations=("hierarchical_did is excluded from scenario P6.",),
        narratives={},
        exploratory_note="Analytic and bootstrap coverage are compared; no direction is claimed.",
    )


def tiny_grid_with_scenario_d() -> GridConfig:
    """A grid small enough to fit in seconds that still has scenario D.

    `REPORT_PROFILE_SELECTIONS` draws an axis profile on scenario D's
    `confounding_strength` (an axis only D, G and H declare), which needs two
    values: a baseline and one perturbed point.
    """
    return GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("D",), n_units=(200,), confounding_strength=(1.0, 1.5)
        ),
        panel=PanelAxes(scenarios=("P1",), n_units=(100,), n_periods=(6,)),
        replications=2,
    )
