"""An analysis as a fact table.

The one module in the package that imports the result types.

A value that is absent gets no fact. A p-value that was never computed, the
statistic of a check that cannot be tested, a comparison that was not run:
none of them appears, so a narrative that cites one fails to resolve and
cannot render as a zero or a blank.
"""

from __future__ import annotations

import math
import re
from typing import cast

from cp_causal.router import RoutingDecision
from cp_core.results import estimate_summary
from cp_report.pipeline import AnalysisOutcome, ComparisonGroup, EstimateReport, FitFailure

from cp_llm.facts import Fact, FactTable

_SEGMENT = re.compile(r"[a-z][a-z0-9_]*")


def _segment(name: str, what: str) -> str:
    """`name` as one id segment, or a refusal.

    Every registered estimator and every assumption check is named in
    snake_case today. A name that is not would need escaping, and an escaped
    id can alias another; refusing is the version that cannot.
    """
    if _SEGMENT.fullmatch(name) is None:
        raise ValueError(
            f"{what} {name!r} cannot be a fact id segment; expected lowercase letters, "
            "digits and underscores, starting with a letter"
        )
    return name


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _estimate_facts(report: EstimateReport) -> list[Fact]:
    estimate, score = report.estimate, report.score
    summary = estimate_summary(estimate)
    method = _segment(estimate.method, "method")
    prefix = f"est.{method}"
    subject = f"{prefix}.method"
    facts = [Fact(subject, "name", estimate.method, None)]

    for field in ("estimand", "scale", "ci_basis"):
        facts.append(Fact(f"{prefix}.{field}", "label", str(summary[field]), subject))
    point = _finite(summary["point"])
    for field in ("point", "se", "ci_lower", "ci_upper"):
        number = point if field == "point" else _finite(summary[field])
        if number is not None:
            facts.append(Fact(f"{prefix}.{field}", "number", number, subject))
    level = _finite(summary["ci_level"])
    if level is not None:
        facts.append(Fact(f"{prefix}.ci_level", "level", level, subject))
    p_value = _finite(summary["p_value"])
    if p_value is not None:
        facts.append(Fact(f"{prefix}.p_value", "p_value", p_value, subject))
    for field in ("n_treated", "n_control"):
        facts.append(Fact(f"{prefix}.{field}", "count", int(cast(int, summary[field])), subject))

    if point is not None:
        direction = "positive" if point > 0 else "negative" if point < 0 else "zero"
        facts.append(Fact(f"{prefix}.direction", "label", direction, subject))
    # No finiteness guard: Interval refuses non-finite bounds at construction.
    excludes = estimate.ci.lower > 0 or estimate.ci.upper < 0
    facts.append(
        Fact(
            f"{prefix}.interval_excludes_zero",
            "label",
            "excludes zero" if excludes else "includes zero",
            subject,
        )
    )

    facts.append(Fact(f"score.{method}.band", "label", score.band, subject))
    facts.append(Fact(f"score.{method}.ceiling", "label", score.ceiling, subject))
    for index, line in enumerate(score.rationale):
        facts.append(Fact(f"score.{method}.rationale.{index}", "text", line, subject))

    # `score.checks`, not `estimate.assumptions`: the band was computed from
    # this tuple, so the table and the band cannot describe different checks.
    for check in score.checks:
        base = f"check.{method}.{_segment(check.name, 'check')}"
        facts.append(Fact(f"{base}.name", "name", check.name, subject))
        facts.append(Fact(f"{base}.kind", "label", check.kind, subject))
        facts.append(Fact(f"{base}.status", "label", check.status, subject))
        for field, raw in (("statistic", check.statistic), ("threshold", check.threshold)):
            number = _finite(raw)
            if number is not None:
                facts.append(Fact(f"{base}.{field}", "number", number, subject))
        facts.append(Fact(f"{base}.interpretation", "text", check.interpretation, subject))
        if check.remedy is not None:
            facts.append(Fact(f"{base}.remedy", "text", check.remedy, subject))
    return facts


def _comparison_facts(group: ComparisonGroup) -> list[Fact]:
    prefix = f"cmp.{_segment(group.estimand.lower(), 'estimand')}.{_segment(group.scale, 'scale')}"
    facts = [Fact(f"{prefix}.n_methods", "count", len(group.methods), None)]
    comparison = group.comparison
    if comparison is None:
        assert group.not_compared_reason is not None
        facts.append(
            Fact(f"{prefix}.not_compared_reason", "text", group.not_compared_reason, None)
        )
        return facts
    low, high = comparison.point_range
    facts.extend(
        [
            Fact(f"{prefix}.agreement", "label", comparison.agreement, None),
            Fact(f"{prefix}.verdict", "text", comparison.verdict, None),
            Fact(f"{prefix}.point_low", "number", float(low), None),
            Fact(f"{prefix}.point_high", "number", float(high), None),
            Fact(f"{prefix}.dispersion", "number", float(comparison.dispersion), None),
        ]
    )
    return facts


def _routing_facts(decision: RoutingDecision) -> list[Fact]:
    prefix = f"route.{_segment(decision.estimator_name, 'estimator')}"
    subject = f"{prefix}.estimator"
    facts = [
        Fact(subject, "name", decision.estimator_name, None),
        Fact(
            f"{prefix}.eligible",
            "label",
            "eligible" if decision.eligible else "not eligible",
            subject,
        ),
    ]
    for index, reason in enumerate(decision.reasons):
        facts.append(Fact(f"{prefix}.reasons.{index}", "text", reason, subject))
    return facts


def _failure_facts(failure: FitFailure) -> list[Fact]:
    prefix = f"fail.{_segment(failure.method, 'method')}"
    subject = f"{prefix}.method"
    return [
        Fact(subject, "name", failure.method, None),
        Fact(f"{prefix}.error", "text", failure.error, subject),
    ]


def fact_table(outcome: AnalysisOutcome) -> FactTable:
    """Every citable fact about one analysis, in a stable order."""
    facts = [
        Fact(
            "analysis.assignment_provenance",
            "label",
            outcome.manifest.schema.assignment_provenance,
            None,
        ),
        Fact("analysis.n_estimates", "count", len(outcome.reports), None),
        Fact("analysis.n_failed", "count", len(outcome.failures), None),
    ]
    for report in outcome.reports:
        facts.extend(_estimate_facts(report))
    for group in outcome.comparisons:
        facts.extend(_comparison_facts(group))
    for decision in outcome.decisions:
        facts.extend(_routing_facts(decision))
    for failure in outcome.failures:
        facts.extend(_failure_facts(failure))
    return FactTable(tuple(facts))
