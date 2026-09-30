"""From a dataset to one evidence report per estimate.

Every piece already existed -- routing, fitting, validity scoring,
cross-method comparison, both renderers -- and nothing chained them: the
command line stopped at estimates, and `Evidence` was only ever built in
tests. `analyse` is that chain. The command line's `analyze --report` and
the backend worker both call it, so neither carries a copy.

Each estimator is fitted separately. `run_analysis` fits all of its
estimators in one expression, so one raising estimator would discard every
other fit. A failure here is recorded, not raised, and the manifest names
only the estimators that succeeded -- a manifest naming a failed estimator
is one `replay` could not honour.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from cp_causal.comparison import MethodComparison, compare
from cp_causal.manifest import run_analysis
from cp_causal.router import RoutingDecision, route
from cp_causal.validity import score_validity
from cp_core.dataset import AssignmentProvenance, CausalDataset
from cp_core.manifest import AnalysisManifest, DatasetReference
from cp_core.results import CausalValidityScore, CounterfactualPath, EffectEstimate
from cp_core.values import EstimatorConfig
from cp_figures.effects import (
    counterfactual_path_data,
    draw_counterfactual_path,
    draw_forest_plot,
    forest_plot_data,
)

from cp_report.evidence import Evidence, ReportFigure
from cp_report.figures import strip_xml_preamble
from cp_report.html import render_html_report
from cp_report.markdown import write_markdown_report


@dataclass(frozen=True)
class FitFailure:
    """An estimator that raised, and its message exactly as raised."""

    method: str
    error: str


def select_estimators(
    dataset: CausalDataset, requested: Sequence[str] | None
) -> tuple[tuple[RoutingDecision, ...], tuple[str, ...]]:
    """Route the dataset, then settle which estimators to fit.

    `requested=None` means every estimator the router judges eligible. A
    request is refused whole, before anything is fitted, if it names an
    unknown or an ineligible estimator -- with the router's own reasons,
    unreworded.
    """
    decisions = route(dataset)
    by_name = {d.estimator_name: d for d in decisions}
    if requested is None:
        names = tuple(d.estimator_name for d in decisions if d.eligible)
        if not names:
            reasons = "; ".join(f"{d.estimator_name}: {', '.join(d.reasons)}" for d in decisions)
            raise ValueError(f"no registered estimator is eligible for this dataset ({reasons})")
        return decisions, names
    if not requested:
        raise ValueError(
            "a request must name at least one estimator; pass none at all to let "
            "the router choose"
        )
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"unknown estimator(s): {unknown}; registered: {sorted(by_name)}")
    names = tuple(dict.fromkeys(requested))
    ineligible = [by_name[name] for name in names if not by_name[name].eligible]
    if ineligible:
        reasons = "; ".join(f"{d.estimator_name}: {', '.join(d.reasons)}" for d in ineligible)
        raise ValueError(f"requested estimator(s) are not eligible for this dataset: {reasons}")
    return decisions, names


def fit_each(
    dataset: CausalDataset,
    reference: DatasetReference,
    names: Sequence[str],
    config: EstimatorConfig,
) -> tuple[tuple[EffectEstimate, ...], tuple[FitFailure, ...], AnalysisManifest]:
    """Fit each estimator on its own, so one failure cannot discard the rest."""
    estimates: list[EffectEstimate] = []
    succeeded: list[str] = []
    failures: list[FitFailure] = []
    for name in names:
        try:
            _, (estimate,) = run_analysis(dataset, reference, [(name, config)])
        except Exception as exc:
            failures.append(FitFailure(method=name, error=str(exc) or type(exc).__name__))
            continue
        estimates.append(estimate)
        succeeded.append(name)
    if not estimates:
        detail = "; ".join(f"{f.method}: {f.error}" for f in failures)
        raise ValueError(f"every requested estimator failed ({detail})")
    manifest = AnalysisManifest.capture(
        dataset=reference,
        schema=dataset.schema,
        estimators=[(name, config) for name in succeeded],
    )
    return tuple(estimates), tuple(failures), manifest


@dataclass(frozen=True, eq=False)
class EstimateReport:
    """One estimate, its own score, and its report in both formats.

    `assets` maps each Markdown figure's filename to its SVG text. The
    Markdown links to its figures by bare filename, so a consumer that keeps
    only `markdown` is left with broken images; the HTML inlines its SVG.
    """

    estimate: EffectEstimate
    score: CausalValidityScore
    html: str
    markdown: str
    assets: Mapping[str, str]


@dataclass(frozen=True, eq=False)
class ComparisonGroup:
    """Estimates that share an estimand and a scale, compared -- or why not.

    `compare` refuses a mixed estimand or scale (an ATE and an ATT are
    different quantities) and needs at least two estimates, so a routed set
    is compared within groups, and a group of one says so.
    """

    estimand: str
    scale: str
    methods: tuple[str, ...]
    comparison: MethodComparison | None
    not_compared_reason: str | None

    def to_dict(self) -> dict[str, object]:
        base: dict[str, object] = {
            "estimand": self.estimand,
            "scale": self.scale,
            "methods": list(self.methods),
        }
        if self.comparison is None:
            base["not_compared_reason"] = self.not_compared_reason
            return base
        low, high = self.comparison.point_range
        base.update(
            agreement=self.comparison.agreement,
            verdict=self.comparison.verdict,
            point_range=[float(low), float(high)],
            dispersion=float(self.comparison.dispersion),
        )
        return base


@dataclass(frozen=True, eq=False)
class AnalysisOutcome:
    decisions: tuple[RoutingDecision, ...]
    reports: tuple[EstimateReport, ...]
    failures: tuple[FitFailure, ...]
    comparisons: tuple[ComparisonGroup, ...]
    manifest: AnalysisManifest

    @property
    def partial(self) -> bool:
        """Some requested estimators failed while at least one succeeded."""
        return bool(self.failures)


def _figures(estimate: EffectEstimate) -> tuple[ReportFigure, ...]:
    figures = [
        ReportFigure(
            caption=f"Point estimate and interval for {estimate.method}.",
            draw=partial(draw_forest_plot, forest_plot_data((estimate,))),
        )
    ]
    if isinstance(estimate.counterfactual, CounterfactualPath):
        figures.append(
            ReportFigure(
                caption=(
                    "Observed outcome against the counterfactual path "
                    f"{estimate.method} estimated."
                ),
                draw=partial(
                    draw_counterfactual_path, counterfactual_path_data(estimate.counterfactual)
                ),
            )
        )
    return tuple(figures)


def render_estimate_report(
    estimate: EffectEstimate, provenance: AssignmentProvenance
) -> EstimateReport:
    """Score one estimate and render its report.

    The assumption table and the band come from the same tuple:
    `checks=score.checks`, which `score_validity` read from this estimate.
    `Evidence` cannot enforce that agreement itself (see its docstring).
    """
    score = score_validity(estimate, provenance=provenance)
    evidence = Evidence(
        estimates=(estimate,),
        score=score,
        checks=score.checks,
        limitations=(),
        figures=_figures(estimate),
    )
    html = render_html_report(evidence)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        document = write_markdown_report(evidence, directory)
        markdown = document.read_text(encoding="utf-8")
        # `write_markdown_report` writes each figure via `figure_to_file`, which
        # (unlike `figure_to_inline_svg`) keeps the XML declaration and DOCTYPE,
        # since the file it writes is a whole standalone document. Per spec §6,
        # this same asset is later served one-per-URL as `image/svg+xml` and
        # written back to disk by `causalpulse analyze --report` -- so the
        # preamble is stripped here with `strip_xml_preamble` so a stored asset
        # has one shape whether it is served, written, or inlined. The result
        # is still a valid standalone SVG file: the root `<svg>` element
        # declares its own namespaces, so dropping the XML declaration and
        # DOCTYPE in front of it loses nothing a consumer needs.
        assets = {
            path.name: strip_xml_preamble(path.read_text(encoding="utf-8"))
            for path in sorted(directory.iterdir())
            if path != document
        }
    return EstimateReport(
        estimate=estimate, score=score, html=html, markdown=markdown, assets=assets
    )


def comparison_groups(
    pairs: Sequence[tuple[EffectEstimate, CausalValidityScore]],
) -> tuple[ComparisonGroup, ...]:
    grouped: dict[tuple[str, str], list[tuple[EffectEstimate, CausalValidityScore]]] = {}
    for estimate, score in pairs:
        grouped.setdefault((estimate.estimand, estimate.scale), []).append((estimate, score))
    groups = []
    for (estimand, scale), members in sorted(grouped.items()):
        methods = tuple(estimate.method for estimate, _ in members)
        if len(members) < 2:
            groups.append(
                ComparisonGroup(
                    estimand=estimand,
                    scale=scale,
                    methods=methods,
                    comparison=None,
                    not_compared_reason=(
                        f"only one method estimated the {estimand} on the {scale} scale, "
                        "so there is no second estimate to compare it with"
                    ),
                )
            )
            continue
        comparison = compare(
            tuple(estimate for estimate, _ in members), tuple(score for _, score in members)
        )
        groups.append(
            ComparisonGroup(
                estimand=estimand,
                scale=scale,
                methods=methods,
                comparison=comparison,
                not_compared_reason=None,
            )
        )
    return tuple(groups)


def analyse(
    dataset: CausalDataset,
    reference: DatasetReference,
    estimators: Sequence[str] | None = None,
    random_seed: int = 0,
) -> AnalysisOutcome:
    """Route, fit each estimator, and report on each estimate separately.

    Raises if the request is refused (unknown or ineligible estimators) or
    if every estimator fails; otherwise returns, recording any per-estimator
    failure on the outcome.
    """
    decisions, names = select_estimators(dataset, estimators)
    config = EstimatorConfig(random_seed=random_seed)
    estimates, failures, manifest = fit_each(dataset, reference, names, config)
    provenance = dataset.schema.assignment_provenance
    reports = tuple(render_estimate_report(estimate, provenance) for estimate in estimates)
    return AnalysisOutcome(
        decisions=decisions,
        reports=reports,
        failures=failures,
        comparisons=comparison_groups([(r.estimate, r.score) for r in reports]),
        manifest=manifest,
    )
