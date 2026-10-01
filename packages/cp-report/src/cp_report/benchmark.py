"""The benchmark study's research report: data, sections and renderers.

Sub-project 2, part 2 (`docs/specs/2026-09-23-benchmark-study-report-design.md`).
Mirrors `cp_report.evidence`/`sections`/`html`/`markdown` for a different document
shape: aggregate performance across many estimators and scenarios rather than one
analysis's assumption checks. `BenchmarkEvidence` plays `Evidence`'s role -- the one
argument each entry point takes -- and is built by `benchmark_evidence_from_run`
from a finished study's stored results, the only function here that imports
`cp_evaluation`.

Six sections, each a frozen dataclass with a `prose_fragments()` method, exactly as
`cp_report.sections` does: `OverviewData`, `HeadlineFindingsData`,
`MethodComparisonsData`, `LimitationsData` (this module's own, distinct from
`cp_report.sections.LimitationsData`), `CateSectionData` (omitted when there is no
scored CATE row) and `P7SectionData` (omitted when no estimator fitted on scenario P7). No new
figure-drawing code: every `ReportFigure` wraps an existing `cp_figures.benchmark` draw
function via `partial`.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cp_evaluation.cate import CATE_METRIC_COLUMNS
from cp_evaluation.expectations import ExpectationResult, Verdict, evaluate_expectations
from cp_evaluation.metrics import summarise
from cp_evaluation.store import RunStatus, RunStore
from cp_evaluation.study import (
    baseline_slice,
    format_value,
    headline_tables,
    markdown_table,
    not_covered_for,
)
from cp_figures.benchmark import axis_profile_data, draw_axis_profile, draw_heatmap, heatmap_data
from cp_simulation.cate_scenarios import CATE_SCENARIOS
from cp_simulation.panel_scenarios import PANEL_PROTOCOLS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cp_report.evidence import ReportFigure
from cp_report.figures import figure_to_file, figure_to_inline_svg
from cp_report.language import require_clean_prose

# One heatmap: each estimator's primary-interval coverage over the baseline-slice pooled
# summary, in a synthetic `coverage_primary` column (see `_primary_values`). A bias heatmap
# was tried and dropped: two scenarios built to break the DiD family (P4, P5) have biases
# hundreds of times the rest, so the colour scale left every other cell one colour, and the
# bias table already gives the numbers. The choice is recorded in the design doc's
# "As built" section.
REPORT_HEATMAP_METRICS: tuple[str, ...] = ("coverage_primary",)
REPORT_PROFILE_SELECTIONS: tuple[tuple[str, str, str], ...] = (
    ("D", "confounding_strength", "bias"),
)

# The two tables the document embeds, each a compact method-by-scenario view built from a
# `headline_tables` source table (`BenchmarkEvidence.tables` keeps the full set as a record;
# a 130-row, 18-column source table is not something a reader can read). Bias comes from
# `accuracy__scenario`, coverage from `coverage__scenario`.
BIAS_TABLE = "Bias by method and scenario"
COVERAGE_TABLE = "Coverage of each estimator's primary interval, by scenario"

# `coverage_metrics` reports coverage per `ci_basis`
# (2026-09-23-coverage-basis-metrics-design.md). Each estimator's primary interval has
# exactly one basis, so the coverage table collapses the basis columns into one value per
# cell and an `interval` column naming the basis. The note is code, not narrative, so no
# author can forget what that column means; re-read it against `coverage_metrics` if the
# basis columns change.
COVERAGE_COLUMN_NOTE = (
    "Each coverage value is how often the interval an estimator reports as its primary one "
    "contains the true effect, and the interval column names that interval's kind: analytic "
    "for standard-error intervals, posterior for the credible interval hierarchical_did "
    "reports, bootstrap_se for the Wald interval callaway_santanna builds from a "
    "cluster-bootstrap standard error, and permutation for the interval synthetic_control "
    "builds from its placebo distribution."
)


@dataclass(frozen=True)
class OverviewData:
    """The grid's shape and what its replication count buys in precision.

    `coverage_mcse` is the binomial Monte Carlo standard error a coverage figure
    carries at this replication count, `sqrt(.95*.05/replications)` -- the arithmetic
    `cp_evaluation.study.coverage_of_study_markdown` already does.
    """

    n_estimators: int
    n_scenarios: int
    n_cells: int
    replications: int
    coverage_mcse: float

    def prose_fragments(self) -> tuple[str, ...]:
        """Nothing here is free text: every field is a count or a computed number."""
        return ()


def overview_data(
    n_estimators: int, n_scenarios: int, n_cells: int, replications: int
) -> OverviewData:
    return OverviewData(
        n_estimators=n_estimators,
        n_scenarios=n_scenarios,
        n_cells=n_cells,
        replications=replications,
        coverage_mcse=float(np.sqrt(0.95 * 0.05 / replications)),
    )


@dataclass(frozen=True)
class ExpectationRow:
    """One pre-registered expectation as a report reader sees it.

    `narrative` is the caller's reader-facing paragraph when one was supplied, else
    `result.note`, else `result.statement` -- `ExpectationResult.note` exists for
    exactly this, a reworded explanation standing in for the raw statement.
    """

    id: str
    statement: str
    verdict: Verdict
    narrative: str


@dataclass(frozen=True)
class HeadlineFindingsData:
    """One row per pre-registered expectation, plus the tally a reader might check."""

    rows: tuple[ExpectationRow, ...]
    n_met: int
    n_failed: int
    n_inconclusive: int

    def prose_fragments(self) -> tuple[str, ...]:
        """Included: each row's `statement` and, when it differs, its `narrative` --
        both free text, and the narrative is exactly the caller-supplied override the
        lint exists for. Excluded: `verdict`, from a fixed vocabulary."""
        fragments: list[str] = []
        for row in self.rows:
            fragments.append(row.statement)
            if row.narrative != row.statement:
                fragments.append(row.narrative)
        return tuple(fragments)


def headline_findings_data(
    results: Sequence[ExpectationResult], narratives: Mapping[str, str]
) -> HeadlineFindingsData:
    rows = tuple(
        ExpectationRow(
            id=result.id,
            statement=result.statement,
            verdict=result.verdict,
            narrative=narratives.get(result.id, result.note or result.statement),
        )
        for result in results
    )
    return HeadlineFindingsData(
        rows=rows,
        n_met=sum(1 for r in results if r.verdict == "met"),
        n_failed=sum(1 for r in results if r.verdict == "failed"),
        n_inconclusive=sum(1 for r in results if r.verdict == "inconclusive"),
    )


@dataclass(frozen=True, eq=False)
class MethodComparisonsData:
    """The curated tables, the coverage-column note, and the exploratory note.

    `eq=False`: holds DataFrames, whose comparison has an ambiguous truth value.
    `coverage_note` is `COVERAGE_COLUMN_NOTE` when a coverage table is embedded, else
    empty.
    """

    tables: dict[str, pd.DataFrame]
    exploratory_note: str
    coverage_note: str

    def prose_fragments(self) -> tuple[str, ...]:
        """Included: both notes. Excluded: `tables`, numeric data."""
        return (self.exploratory_note, *((self.coverage_note,) if self.coverage_note else ()))


# The `coverage_metrics` columns that hold a primary interval's coverage, by the `ci_basis`
# each reads. An estimator's primary interval has exactly one basis, so at most one of these
# is populated for any of its rows.
_PRIMARY_COVERAGE: tuple[tuple[str, str], ...] = (
    ("analytic", "coverage_analytic"),
    ("posterior", "coverage_posterior"),
    ("bootstrap_se", "coverage_bootstrap_se"),
    ("permutation", "coverage_permutation"),
)


def _primary_values(frame: pd.DataFrame) -> pd.Series:
    """Each row's primary-interval coverage: the first populated `_PRIMARY_COVERAGE` column.

    Empty when the frame carries none of those columns.
    """
    columns = [column for _, column in _PRIMARY_COVERAGE if column in frame.columns]
    if not columns:
        return pd.Series(np.nan, index=frame.index)
    return frame[columns].bfill(axis=1).iloc[:, 0]


def _bias_table(accuracy: pd.DataFrame) -> pd.DataFrame:
    """Bias, one row per method and one column per scenario; blank where not fitted."""
    wide = accuracy.pivot(index="method", columns="scenario", values="bias")
    wide.columns.name = None
    return wide.reset_index()


def _coverage_table(coverage: pd.DataFrame) -> pd.DataFrame:
    """Each estimator's primary-interval coverage per scenario, with the interval's kind.

    Whichever `_PRIMARY_COVERAGE` column an estimator populates supplies its values; the
    `interval` column names that column's basis, and is empty for an estimator with no
    fitted row anywhere.
    """
    present = [(kind, column) for kind, column in _PRIMARY_COVERAGE if column in coverage.columns]
    values = _primary_values(coverage)
    kinds = {
        str(method): next((kind for kind, column in present if rows[column].notna().any()), "")
        for method, rows in coverage.groupby("method")
    }
    wide = coverage.assign(_value=values).pivot(index="method", columns="scenario", values="_value")
    wide.columns.name = None
    wide.insert(0, "interval", [kinds[str(method)] for method in wide.index])
    return wide.reset_index()


def method_comparisons_data(
    tables: dict[str, pd.DataFrame], exploratory_note: str
) -> MethodComparisonsData:
    """Build the curated bias and coverage tables from the full `headline_tables()` set.

    A source table missing from `tables` is skipped rather than raising: the document
    renders what it has. `coverage_note` accompanies the coverage table and only it.
    """
    curated: dict[str, pd.DataFrame] = {}
    if "accuracy__scenario" in tables:
        curated[BIAS_TABLE] = _bias_table(tables["accuracy__scenario"])
    if "coverage__scenario" in tables:
        curated[COVERAGE_TABLE] = _coverage_table(tables["coverage__scenario"])
    return MethodComparisonsData(
        tables=curated,
        exploratory_note=exploratory_note,
        coverage_note=COVERAGE_COLUMN_NOTE if COVERAGE_TABLE in curated else "",
    )


@dataclass(frozen=True)
class LimitationsData:
    """What the study did not cover, plus whatever else the caller states.

    `not_covered` is `cp_evaluation.study.NOT_COVERED`, the structural source this
    section leads with; `statements` is the caller's tuple, carried through unmodified.
    """

    not_covered: tuple[tuple[str, str], ...]
    statements: tuple[str, ...]

    def prose_fragments(self) -> tuple[str, ...]:
        """Included: every `not_covered` reason and every caller statement -- the
        likeliest place for an unhedged claim, since a report's author wrote it.
        Excluded: the `not_covered` names, from a fixed set of identifiers."""
        return tuple(reason for _, reason in self.not_covered) + self.statements


def limitations_data(
    not_covered: tuple[tuple[str, str], ...], statements: tuple[str, ...]
) -> LimitationsData:
    return LimitationsData(not_covered=not_covered, statements=statements)


# Fixed prose, linted with the rest of the document. It states what each figure
# measures and how to read a calibration slope; it claims nothing about why an
# estimator scores as it does (design §6.3).
CATE_NOTE = (
    "Each number in this table compares the per-unit effect an estimator reports with the "
    "true per-unit effect of the same simulated units, and is a mean over the fits counted "
    "in its `_n_used` column, with its Monte Carlo standard error. Correlation measures how "
    "closely the estimates track the true effects across units, and RMSE their typical "
    "distance from them. The calibration slope is the least-squares slope of the true effects "
    "on the estimates: at 1, a difference between two units' estimates matches, on average, "
    "the difference between their true effects; above 1, the estimates understate those "
    "differences, and below 1 they overstate them. Scenario NULL gives every unit the same "
    "true effect, so only its RMSE is reported."
)


@dataclass(frozen=True, eq=False)
class CateSectionData:
    """Per-unit CATE recovery, one row per method and scenario with a scored fit.

    `eq=False`: holds a DataFrame. `replications` and the cell counts describe the
    store the rows came from, which need not be the main study's.
    """

    table: pd.DataFrame
    replications: int
    cells_done: int
    cells_expected: int
    note: str

    def prose_fragments(self) -> tuple[str, ...]:
        """Included: the note. Excluded: `table`, numeric data."""
        return (self.note,)


def cate_section_data(
    rows: pd.DataFrame, status: RunStatus, replications: int
) -> CateSectionData | None:
    """The CATE section over `rows`, or `None` when no fitted row was scored.

    The numbers are `summarise`'s own, pooled per method and scenario, so they are
    exactly what `bench report` prints for the same rows. It pools every row of a
    scenario (`by=["scenario"]`), so `_cate_counts`' "replications per scenario" is
    accurate only while `CateAxes` has one configuration per scenario, as it does today.
    """
    if rows.empty:
        return None
    summary = summarise(rows, by=["scenario"])
    counts = [f"{metric}_n_used" for metric in CATE_METRIC_COLUMNS]
    scored = summary[summary[counts].sum(axis=1) > 0]
    if scored.empty:
        return None
    columns = [
        "method",
        "scenario",
        *(
            f"{metric}{suffix}"
            for metric in CATE_METRIC_COLUMNS
            for suffix in ("", "_mcse", "_n_used")
        ),
    ]
    return CateSectionData(
        table=scored[columns].reset_index(drop=True),
        replications=replications,
        cells_done=status.done,
        cells_expected=status.expected,
        note=CATE_NOTE,
    )


_P7_UNITS, _P7_PERIODS = PANEL_PROTOCOLS["P7"]

P7_NOTE = (
    f"Scenario P7 has seven adoption cohorts whose effects differ across cohorts, and runs at "
    f"its own protocol of {_P7_UNITS} units over {_P7_PERIODS} periods, the size at which a "
    "hierarchical model's cross-cohort variance can be identified. It is reported apart from "
    "scenarios P1 to P6, which ran at 150 units over 7 and 10 periods, so its figures are not "
    "comparable with theirs cell for cell. Each number is a mean over the fits counted in its "
    "_n_used column, with its Monte Carlo standard error, and coverage is that of each "
    "estimator's primary interval, whose kind the interval column names."
)


@dataclass(frozen=True, eq=False)
class P7SectionData:
    """Scenario P7's ATT accuracy and primary-interval coverage, one row per estimator.

    `eq=False`: holds a DataFrame. `replications` and the cell counts describe the
    store the rows came from.
    """

    table: pd.DataFrame
    replications: int
    cells_done: int
    cells_expected: int
    note: str

    def prose_fragments(self) -> tuple[str, ...]:
        """Included: the note. Excluded: `table`, numeric data."""
        return (self.note,)


def _primary_coverage(record: Mapping[Hashable, Any]) -> tuple[str, float, float, int]:
    """(kind, value, MCSE, n_used) of a summary row's primary interval, the first
    `_PRIMARY_COVERAGE` basis it populates, exactly as the coverage table reads it."""
    for kind, column in _PRIMARY_COVERAGE:
        value = record.get(column)
        if isinstance(value, float) and np.isfinite(value):
            return (
                kind,
                value,
                float(record[f"{column}_mcse"]),
                int(record[f"{column}_n_used"]),
            )
    return ("", float("nan"), float("nan"), 0)


def p7_section_data(
    rows: pd.DataFrame, status: RunStatus, replications: int
) -> P7SectionData | None:
    """The P7 section over `rows`, or `None` when no estimator fitted on P7.

    Pooled per method with `summarise`, so the numbers are exactly what `bench
    report` prints for the same rows.
    """
    if rows.empty:
        return None
    summary = summarise(rows, by=["scenario"])
    summary = summary[summary["n_fitted"] > 0]
    if summary.empty:
        return None
    records = []
    for record in summary.to_dict("records"):
        kind, coverage, coverage_mcse, coverage_n = _primary_coverage(record)
        records.append(
            {
                "method": record["method"],
                "bias": record["bias"], "bias_mcse": record["bias_mcse"],
                "bias_n_used": record["bias_n_used"],
                "rmse": record["rmse"], "rmse_mcse": record["rmse_mcse"],
                "rmse_n_used": record["rmse_n_used"],
                "interval": kind, "coverage": coverage, "coverage_mcse": coverage_mcse,
                "coverage_n_used": coverage_n,
            }
        )
    return P7SectionData(
        table=pd.DataFrame(records),
        replications=replications,
        cells_done=status.done,
        cells_expected=status.expected,
        note=P7_NOTE,
    )


@dataclass(frozen=True, eq=False)
class BenchmarkEvidence:
    """Everything the two entry points need, gathered once by `benchmark_evidence_from_run`."""

    expectations: tuple[ExpectationResult, ...]
    tables: dict[str, pd.DataFrame]
    # ATE part of the main store's status (`_split_status`); E8 is evaluated on the whole store's.
    status: RunStatus
    replications: int
    scenarios_run: tuple[str, ...]
    estimators_run: tuple[str, ...]
    not_covered: tuple[tuple[str, str], ...]
    figures: tuple[ReportFigure, ...]
    limitations: tuple[str, ...]
    narratives: dict[str, str]
    exploratory_note: str
    cate_section: CateSectionData | None = None
    p7_section: P7SectionData | None = None


def _exploratory_note(by_size: pd.DataFrame) -> str:
    """Describe the analytic-versus-bootstrap comparison with no direction claimed --
    the study design's "deliberately not pre-registered" honoured literally: this
    states what is in the table, not a conclusion about it."""
    if by_size.empty:
        return (
            "The analytic-versus-bootstrap coverage comparison across sample sizes has no rows "
            "to report in this run."
        )
    return (
        "Analytic and bootstrap interval coverage were also compared across the two "
        "cross-sectional sample sizes the grid varies; that comparison is the "
        "coverage__scenario_by_size table in the study's own output and is not reproduced "
        "here. It was not pre-registered as an expectation, and no direction is claimed."
    )


_HEATMAP_CAPTIONS = {
    "coverage_primary": (
        "Coverage of each estimator's primary interval, by method and scenario, pooled over "
        "each scenario's baseline slice. Lighter cells have higher coverage; grey cells have "
        "no fitted result."
    )
}


def _laid_out(draw: Callable[[Axes], Axes]) -> Callable[[Axes], Axes]:
    """Wrap a draw function so its figure lays itself out once drawn.

    `figure_to_inline_svg`/`figure_to_file` save a default-margin figure, which clips a
    long tick label such as `callaway_santanna`; `bench study` avoids that by saving with
    `bbox_inches="tight"`, which the shared renderers do not do.
    """

    def drawn(ax: Axes) -> Axes:
        result = draw(ax)
        figure = ax.get_figure()
        if isinstance(figure, Figure):
            figure.tight_layout()
        return result

    return drawn


def _report_figures(frame: pd.DataFrame) -> tuple[ReportFigure, ...]:
    """The curated figure set, each wrapping an existing `cp_figures.benchmark` draw
    function. A profile selection whose scenario is absent from `frame` is skipped,
    like a missing curated table."""
    pooled = summarise(baseline_slice(frame), by=["scenario"])
    pooled["coverage_primary"] = _primary_values(pooled)
    per_cell = summarise(frame)
    present = {str(s) for s in frame["scenario"].unique()}
    figures = [
        ReportFigure(
            caption=_HEATMAP_CAPTIONS[metric],
            draw=_laid_out(partial(draw_heatmap, heatmap_data(pooled, metric))),
        )
        for metric in REPORT_HEATMAP_METRICS
    ]
    for scenario, axis, metric in REPORT_PROFILE_SELECTIONS:
        if scenario not in present:
            continue
        figures.append(
            ReportFigure(
                caption=f"{metric} on scenario {scenario}, varying {axis} one value at a time.",
                draw=_laid_out(
                    partial(
                        draw_axis_profile,
                        axis_profile_data(per_cell, metric, axis=axis, scenario=scenario),
                    )
                ),
            )
        )
    return tuple(figures)


def _split_status(status: RunStatus) -> tuple[RunStatus, RunStatus, RunStatus]:
    """`status` as an (ATE part, CATE part, long-panel part) triple, split by scenario.

    A study store that carries the CATE or P7 arm counts its cells in one ledger; the
    overview counts ATE cells and each section counts its own (design §6.1).
    Orphan shards are not attributable to a scenario, so they stay on the ATE part.
    It classifies by scenario name (`CATE_SCENARIOS` and `PANEL_PROTOCOLS` keys), so an
    ATE scenario that someone later names RICH, NULL, NOISY or P7 would be misfiled.
    """
    ate = {
        k: v
        for k, v in status.by_scenario.items()
        if k not in CATE_SCENARIOS and k not in PANEL_PROTOCOLS
    }
    cate = {k: v for k, v in status.by_scenario.items() if k in CATE_SCENARIOS}
    long_panel = {k: v for k, v in status.by_scenario.items() if k in PANEL_PROTOCOLS}

    def part(by_scenario: dict[str, tuple[int, int]], orphans: int) -> RunStatus:
        return RunStatus(
            expected=sum(total for _, total in by_scenario.values()),
            done=sum(done for done, _ in by_scenario.values()),
            orphans=orphans,
            by_scenario=by_scenario,
        )

    return part(ate, status.orphans), part(cate, 0), part(long_panel, 0)


def benchmark_evidence_from_run(
    directory: Path,
    *,
    cate_directory: Path | None = None,
    p7_directory: Path | None = None,
    limitations: tuple[str, ...] = (),
    narratives: Mapping[str, str] | None = None,
) -> BenchmarkEvidence:
    """Build `BenchmarkEvidence` from a finished, complete sharded study.

    Refuses an incomplete run by reusing `RunStore.read_frame`'s own refusal
    (`IncompleteRunError`). `limitations` and `narratives` are the two places a caller
    states something the stored data cannot: no text is baked in here, including the
    `hierarchical_did`/P6 exclusion, which the generating command states explicitly.
    A narrative keyed to an expectation id that does not exist is refused, not ignored.

    Every existing view is built from the study's rows whose `shape` is not `"cate"`:
    the expectations, tables and figures are ATE views registered on the ATE grid,
    the precedent expectation E7 follows (design
    `docs/specs/2026-09-30-cate-report-section-design.md` §6.2). CATE rows feed only
    the CATE section. They come from the study itself or, when `cate_directory` is
    given, from a complete `cate`-tier run; both at once is refused, since that
    would average two protocols together. The same holds for the fixed-protocol
    long-panel rows (`shape == "long_panel"`, scenario P7), which feed only the P7
    section and come from the study itself or from a complete `p7`-tier run in
    `p7_directory`.
    """
    store = RunStore.open(directory)
    full = store.read_frame()
    status = store.status()
    is_cate = full["shape"] == "cate"
    is_long = full["shape"] == "long_panel"
    frame = full[~(is_cate | is_long)].reset_index(drop=True)
    cate_rows = full[is_cate].reset_index(drop=True)
    long_rows = full[is_long].reset_index(drop=True)
    ate_status, cate_status, p7_status = _split_status(status)
    cate_replications = p7_replications = store.replications

    if cate_directory is not None:
        cate_store = RunStore.open(cate_directory)
        if cate_store.identity.tier != "cate":
            raise ValueError(
                f"{cate_directory} is a {cate_store.identity.tier!r} run; the CATE section "
                "reads a 'cate' run"
            )
        if not cate_rows.empty:
            raise ValueError(
                f"{directory} already carries CATE rows and {cate_directory} is a CATE run; "
                "reading both would average two protocols together"
            )
        cate_rows = cate_store.read_frame()
        cate_status, cate_replications = cate_store.status(), cate_store.replications

    if p7_directory is not None:
        p7_store = RunStore.open(p7_directory)
        if p7_store.identity.tier != "p7":
            raise ValueError(
                f"{p7_directory} is a {p7_store.identity.tier!r} run; the P7 section "
                "reads a 'p7' run"
            )
        if not long_rows.empty:
            raise ValueError(
                f"{directory} already carries P7 rows and {p7_directory} is a P7 run; "
                "reading both would average two protocols together"
            )
        long_rows = p7_store.read_frame()
        p7_status, p7_replications = p7_store.status(), p7_store.replications

    results = evaluate_expectations(frame, status)

    narratives = dict(narratives or {})
    unknown = sorted(set(narratives) - {r.id for r in results})
    if unknown:
        raise ValueError(f"narratives name unknown expectation ids: {unknown}")

    tables = headline_tables(frame)
    return BenchmarkEvidence(
        expectations=results,
        tables=tables,
        status=ate_status,
        replications=store.replications,
        scenarios_run=tuple(sorted(frame["scenario"].unique())),
        estimators_run=tuple(sorted(frame["method"].unique())),
        not_covered=not_covered_for(
            pd.concat([frame, cate_rows, long_rows], ignore_index=True)
        ),
        figures=_report_figures(frame),
        limitations=limitations,
        narratives=narratives,
        exploratory_note=_exploratory_note(
            tables.get("coverage__scenario_by_size", pd.DataFrame())
        ),
        cate_section=cate_section_data(cate_rows, cate_status, cate_replications),
        p7_section=p7_section_data(long_rows, p7_status, p7_replications),
    )


# -- prose assembly and the language lint -----------------------------------------


def _section_data(
    evidence: BenchmarkEvidence,
) -> tuple[OverviewData, HeadlineFindingsData, MethodComparisonsData, LimitationsData]:
    """The four always-present sections, rebuilt from `evidence` in one place so the prose
    the lint sees cannot drift from the prose rendered. The optional CATE and P7 sections are added
    separately, in `document_prose_fragments` and both renderers."""
    return (
        overview_data(
            len(evidence.estimators_run),
            len(evidence.scenarios_run),
            evidence.status.expected,
            evidence.replications,
        ),
        headline_findings_data(evidence.expectations, evidence.narratives),
        method_comparisons_data(evidence.tables, evidence.exploratory_note),
        limitations_data(evidence.not_covered, evidence.limitations),
    )


def document_prose_fragments(evidence: BenchmarkEvidence) -> list[tuple[str, str]]:
    """Every fragment of reader-facing prose a report built from `evidence` shows,
    tagged with its section, in rendering order. Mirrors
    `cp_report.sections.document_prose_fragments`; a figure caption is its own fragment
    because it reaches the reader as a `<figcaption>` or an image's alt text."""
    overview, findings, comparisons, limitations = _section_data(evidence)
    fragments = [("Overview", text) for text in overview.prose_fragments()]
    fragments += [("Headline Findings", text) for text in findings.prose_fragments()]
    fragments += [("Figures", figure.caption) for figure in evidence.figures]
    fragments += [("Method Comparisons", text) for text in comparisons.prose_fragments()]
    if evidence.cate_section is not None:
        fragments += [
            ("Heterogeneous Effects (CATE)", text)
            for text in evidence.cate_section.prose_fragments()
        ]
    if evidence.p7_section is not None:
        fragments += [("Scenario P7", text) for text in evidence.p7_section.prose_fragments()]
    fragments += [("Limitations", text) for text in limitations.prose_fragments()]
    return fragments


# -- HTML -------------------------------------------------------------------------

_STYLE = (
    "body{font-family:sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;}"
    "table{border-collapse:collapse;width:100%;margin:1rem 0;}"
    "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;}"
    "figure{margin:1rem 0;}figcaption{font-style:italic;}"
)


def _html_cell(value: object) -> str:
    return escape(format_value(value)) if value is not None else "—"


def _dataframe_to_html(frame: pd.DataFrame) -> str:
    header = "".join(f"<th>{escape(str(c))}</th>" for c in frame.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_html_cell(v)}</td>" for v in row) + "</tr>"
        for row in frame.itertuples(index=False, name=None)
    )
    return f"<table><tr>{header}</tr>{rows}</table>"


def render_overview_html(data: OverviewData) -> str:
    return (
        "<section><h2>Overview</h2>"
        f"<p>{data.n_estimators} estimators, {data.n_scenarios} scenarios, "
        f"{data.n_cells} cells, {data.replications} replications per configuration "
        f"(coverage MCSE ~= {data.coverage_mcse:.3f}).</p></section>"
    )


def render_headline_findings_html(data: HeadlineFindingsData) -> str:
    rows = "".join(
        f"<tr><td>{escape(row.id)}</td><td>{escape(row.verdict)}</td>"
        f"<td>{escape(row.narrative)}</td></tr>"
        for row in data.rows
    )
    summary = f"Met: {data.n_met} · Failed: {data.n_failed} · Inconclusive: {data.n_inconclusive}"
    return (
        "<section><h2>Headline Findings</h2>"
        f"<p>{summary}</p>"
        "<table><tr><th>ID</th><th>Verdict</th><th>Explanation</th></tr>"
        f"{rows}</table></section>"
    )


def render_method_comparisons_html(data: MethodComparisonsData) -> str:
    tables = "".join(
        f"<h3>{escape(name)}</h3>{_dataframe_to_html(frame)}" for name, frame in data.tables.items()
    )
    coverage_note = f"<p>{escape(data.coverage_note)}</p>" if data.coverage_note else ""
    return (
        "<section><h2>Method Comparisons</h2>"
        f"{tables}{coverage_note}<p><em>{escape(data.exploratory_note)}</em></p></section>"
    )


def _cate_counts(data: CateSectionData) -> str:
    return (
        f"{data.replications} replications per scenario, "
        f"{data.cells_done} of {data.cells_expected} cells."
    )


def render_cate_section_html(data: CateSectionData) -> str:
    return (
        "<section><h2>Heterogeneous Effects (CATE)</h2>"
        f"<p>{escape(_cate_counts(data))}</p>"
        f"{_dataframe_to_html(data.table)}<p>{escape(data.note)}</p></section>"
    )


def _p7_counts(data: P7SectionData) -> str:
    return (
        f"{data.replications} replications per estimator, "
        f"{data.cells_done} of {data.cells_expected} cells."
    )


def render_p7_section_html(data: P7SectionData) -> str:
    return (
        "<section><h2>Scenario P7</h2>"
        f"<p>{escape(_p7_counts(data))}</p>"
        f"{_dataframe_to_html(data.table)}<p>{escape(data.note)}</p></section>"
    )


def render_limitations_html(data: LimitationsData) -> str:
    not_covered = "".join(
        f"<li>{escape(name)}: {escape(reason)}</li>" for name, reason in data.not_covered
    )
    statements = "".join(f"<li>{escape(s)}</li>" for s in data.statements)
    return f"<section><h2>Limitations</h2><ul>{not_covered}{statements}</ul></section>"


def _render_figures_html(figures: tuple[ReportFigure, ...]) -> str:
    if not figures:
        return ""
    body = "".join(
        f"<figure>{figure_to_inline_svg(figure)}<figcaption>{escape(figure.caption)}"
        "</figcaption></figure>"
        for figure in figures
    )
    return f"<section><h2>Figures</h2>{body}</section>"


def render_html_benchmark_report(evidence: BenchmarkEvidence) -> str:
    """Render the report as one self-contained HTML document.

    Lints before rendering, and carries its own `require_clean_prose` call rather than
    sharing one with the Markdown entry point, so a guard removed from one format
    cannot mask a gap in the other.
    """
    overview, findings, comparisons, limitations = _section_data(evidence)
    require_clean_prose(document_prose_fragments(evidence))

    body = "".join(
        [
            render_overview_html(overview),
            render_headline_findings_html(findings),
            _render_figures_html(evidence.figures),
            render_method_comparisons_html(comparisons),
            render_cate_section_html(evidence.cate_section) if evidence.cate_section else "",
            render_p7_section_html(evidence.p7_section) if evidence.p7_section else "",
            render_limitations_html(limitations),
        ]
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>CausalPulse Benchmark Study Report</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>CausalPulse Benchmark Study Report</h1>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


# -- Markdown ---------------------------------------------------------------------

_DOCUMENT_NAME = "benchmark-report.md"


def _md_cell(text: str) -> str:
    """A table cell: a pipe or a newline would end the row early."""
    return text.replace("|", "\\|").replace("\n", " ")


def render_overview_markdown(data: OverviewData) -> str:
    return (
        "## Overview\n\n"
        f"{data.n_estimators} estimators, {data.n_scenarios} scenarios, "
        f"{data.n_cells} cells, {data.replications} replications per configuration "
        f"(coverage MCSE ~= {data.coverage_mcse:.3f})."
    )


def render_headline_findings_markdown(data: HeadlineFindingsData) -> str:
    summary = f"Met: {data.n_met} · Failed: {data.n_failed} · Inconclusive: {data.n_inconclusive}"
    rows = [f"| {row.id} | {row.verdict} | {_md_cell(row.narrative)} |" for row in data.rows]
    table = "\n".join(["| ID | Verdict | Explanation |", "| --- | --- | --- |", *rows])
    return f"## Headline Findings\n\n{summary}\n\n{table}"


def render_method_comparisons_markdown(data: MethodComparisonsData) -> str:
    parts = ["## Method Comparisons"]
    for name, frame in data.tables.items():
        parts.extend([f"### {name}", markdown_table(frame).rstrip("\n")])
    if data.coverage_note:
        parts.append(data.coverage_note)
    parts.append(f"*{data.exploratory_note}*")
    return "\n\n".join(parts)


def render_cate_section_markdown(data: CateSectionData) -> str:
    return "\n\n".join(
        [
            "## Heterogeneous Effects (CATE)",
            _cate_counts(data),
            markdown_table(data.table).rstrip("\n"),
            data.note,
        ]
    )


def render_p7_section_markdown(data: P7SectionData) -> str:
    return "\n\n".join(
        [
            "## Scenario P7",
            _p7_counts(data),
            markdown_table(data.table).rstrip("\n"),
            data.note,
        ]
    )


def render_limitations_markdown(data: LimitationsData) -> str:
    lines = [f"- {name}: {reason}" for name, reason in data.not_covered]
    lines.extend(f"- {statement}" for statement in data.statements)
    return "\n".join(["## Limitations", "", *lines])


def _render_figures_markdown(figures: tuple[ReportFigure, ...], paths: list[Path]) -> str:
    if not figures:
        return ""
    links = "\n".join(
        f"![{figure.caption}]({path.name})" for figure, path in zip(figures, paths, strict=True)
    )
    return f"## Figures\n\n{links}"


def write_markdown_benchmark_report(evidence: BenchmarkEvidence, directory: Path) -> Path:
    """Write the report as one Markdown document plus its SVG figures under `directory`.

    Lints before writing anything, and `directory` must already exist -- the two
    guarantees of `cp_report.markdown.write_markdown_report`, for the same reasons: a
    refusal must never leave a half-written directory, and this never silently `mkdir`s
    a path a caller may not have meant to create.
    """
    overview, findings, comparisons, limitations = _section_data(evidence)
    require_clean_prose(document_prose_fragments(evidence))

    if not directory.is_dir():
        raise FileNotFoundError(
            f"write_markdown_benchmark_report requires an existing directory, got {directory!r}"
        )

    figure_paths = [figure_to_file(figure, directory) for figure in evidence.figures]
    document = "\n\n".join(
        part
        for part in [
            "# CausalPulse Benchmark Study Report",
            render_overview_markdown(overview),
            render_headline_findings_markdown(findings),
            _render_figures_markdown(evidence.figures, figure_paths),
            render_method_comparisons_markdown(comparisons),
            render_cate_section_markdown(evidence.cate_section) if evidence.cate_section else "",
            render_p7_section_markdown(evidence.p7_section) if evidence.p7_section else "",
            render_limitations_markdown(limitations),
        ]
        if part
    )
    path = directory / _DOCUMENT_NAME
    path.write_text(document, encoding="utf-8")
    return path
