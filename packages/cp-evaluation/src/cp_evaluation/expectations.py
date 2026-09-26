"""The study's pre-registered expectations, scored against a finished run.

Written into `docs/specs/2026-09-21-benchmark-study-design.md` (section 8) before
any result existed, so a result cannot be fitted to them afterwards. Each is scored
**met**, **failed** or **inconclusive**, and a failed one is reported with its
measurement, never loosened. Tolerances are in pooled MCSE, and no verdict is
offered on fewer than `MIN_N_USED` pooled rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeGuard

import numpy as np
import pandas as pd

from cp_evaluation.metrics import summarise
from cp_evaluation.store import RunStatus
from cp_evaluation.study import baseline_slice, format_value

Verdict = Literal["met", "failed", "inconclusive"]

# Fewer pooled rows than this and an expectation gets no verdict: an MCSE over a
# handful of rows is not a measurement of anything.
MIN_N_USED = 40

# How many pooled MCSE an expectation may sit from its target and still be met.
TOLERANCE_MCSE = 2.0

@dataclass(frozen=True)
class Measurement:
    label: str
    value: float
    mcse: float
    n_used: int


@dataclass(frozen=True)
class ExpectationResult:
    id: str
    statement: str
    rule: str
    verdict: Verdict
    measurements: tuple[Measurement, ...]
    note: str = ""


def _measure(pooled: pd.DataFrame, method: str, scenario: str, metric: str) -> Measurement | None:
    rows = pooled[(pooled["method"] == method) & (pooled["scenario"] == scenario)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return Measurement(
        label=f"{method} on {scenario}: {metric}",
        value=float(row[metric]),
        mcse=float(row[f"{metric}_mcse"]),
        n_used=int(row[f"{metric}_n_used"]),
    )


def _usable(measurement: Measurement | None) -> TypeGuard[Measurement]:
    return (
        measurement is not None
        and measurement.n_used >= MIN_N_USED
        and bool(np.isfinite(measurement.value))
        and bool(np.isfinite(measurement.mcse))
    )


def _combine(verdicts: Sequence[Verdict]) -> Verdict:
    if "failed" in verdicts:
        return "failed"
    if "inconclusive" in verdicts or not verdicts:
        return "inconclusive"
    return "met"


def _near(measurement: Measurement, target: float) -> Verdict:
    within = abs(measurement.value - target) <= TOLERANCE_MCSE * measurement.mcse
    return "met" if within else "failed"


def _target_expectation(
    id_: str,
    statement: str,
    rule: str,
    pooled: pd.DataFrame,
    methods: Sequence[str],
    scenario: str,
    metric: str,
    target: float,
) -> ExpectationResult:
    measurements = [_measure(pooled, m, scenario, metric) for m in methods]
    verdicts: list[Verdict] = [
        _near(m, target) if _usable(m) else "inconclusive" for m in measurements
    ]
    return ExpectationResult(
        id=id_,
        statement=statement,
        rule=rule,
        verdict=_combine(verdicts),
        measurements=tuple(m for m in measurements if m is not None),
    )


def _more_biased(
    id_: str,
    statement: str,
    rule: str,
    pooled: pd.DataFrame,
    worse: str,
    better: str,
    scenario: str,
) -> ExpectationResult:
    a = _measure(pooled, worse, scenario, "bias")
    b = _measure(pooled, better, scenario, "bias")
    measurements = tuple(m for m in (a, b) if m is not None)
    if not (_usable(a) and _usable(b)):
        return ExpectationResult(id_, statement, rule, "inconclusive", measurements)
    holds = abs(a.value) > abs(b.value) and abs(a.value) > TOLERANCE_MCSE * a.mcse
    return ExpectationResult(id_, statement, rule, "met" if holds else "failed", measurements)


def _hierarchical_coverage(baseline: pd.DataFrame) -> ExpectationResult:
    statement = "hierarchical_did's credible-interval coverage is not below 0.90, pooled"
    rule = "failed only if pooled coverage_posterior + 2 MCSE < 0.90"
    rows = baseline[baseline["method"] == "hierarchical_did"]
    if rows.empty:
        return ExpectationResult("E5", statement, rule, "inconclusive", ())
    pooled = summarise(rows, by=[])
    row = pooled.iloc[0]
    measurement = Measurement(
        label="hierarchical_did pooled: coverage_posterior",
        value=float(row["coverage_posterior"]),
        mcse=float(row["coverage_posterior_mcse"]),
        n_used=int(row["coverage_posterior_n_used"]),
    )
    if not _usable(measurement):
        return ExpectationResult("E5", statement, rule, "inconclusive", (measurement,))
    failed = measurement.value + TOLERANCE_MCSE * measurement.mcse < 0.90
    return ExpectationResult("E5", statement, rule, "failed" if failed else "met", (measurement,))


def _no_fits_on_missing_outcomes(frame: pd.DataFrame) -> ExpectationResult:
    statement = "No method has any fitted row on scenario H (missing outcomes)"
    rule = "n_fitted = 0 for every method on H"
    rows = frame[frame["scenario"] == "H"]
    if rows.empty:
        return ExpectationResult("E6", statement, rule, "inconclusive", ())
    fitted = rows[rows["status"] == "fitted"]
    measurement = Measurement(
        "fitted rows on H, all methods", float(len(fitted)), float("nan"), len(rows)
    )
    verdict: Verdict = "met" if fitted.empty else "failed"
    return ExpectationResult("E6", statement, rule, verdict, (measurement,))


def _slowest_is_hierarchical(frame: pd.DataFrame) -> ExpectationResult:
    statement = "hierarchical_did has the largest mean runtime_seconds of the estimators"
    rule = "rank 1 by mean runtime_seconds over each method's fitted rows"
    per_method = summarise(frame, by=[])
    per_method = per_method[per_method["runtime_seconds_n_used"] > 0]
    if "hierarchical_did" not in set(per_method["method"]):
        return ExpectationResult("E7", statement, rule, "inconclusive", ())
    ranked = per_method.sort_values("runtime_seconds", ascending=False)
    measurements = tuple(
        Measurement(
            f"{r['method']}: runtime_seconds",
            float(r["runtime_seconds"]),
            float(r["runtime_seconds_mcse"]),
            int(r["runtime_seconds_n_used"]),
        )
        for _, r in ranked.head(3).iterrows()
    )
    top = str(ranked.iloc[0]["method"])
    return ExpectationResult(
        "E7", statement, rule, "met" if top == "hierarchical_did" else "failed", measurements
    )


def _store_is_complete(status: RunStatus) -> ExpectationResult:
    statement = "The store is complete and consistent"
    rule = "every expected cell has an intact shard in the ledger, and no orphan shards remain"
    ok = status.remaining == 0 and status.orphans == 0
    measurements = (
        Measurement("cells expected", float(status.expected), float("nan"), status.expected),
        Measurement("cells done", float(status.done), float("nan"), status.done),
        Measurement("orphan shards", float(status.orphans), float("nan"), status.orphans),
    )
    return ExpectationResult("E8", statement, rule, "met" if ok else "failed", measurements)


RANDOMISATION_ESTIMATORS = ("difference_in_means", "lin_regression", "ipw", "aipw")


def evaluate_expectations(frame: pd.DataFrame, status: RunStatus) -> tuple[ExpectationResult, ...]:
    """Score E1 to E8 against a finished run. See the design document's section 8."""
    baseline = baseline_slice(frame)
    pooled = summarise(baseline, by=["scenario"])
    return (
        _target_expectation(
            "E1",
            "False-positive rate under scenario B is about 0.05 for the four estimators "
            "that assume randomisation or unconfoundedness",
            "0.05 within 2 pooled MCSE, for each of the four",
            pooled,
            RANDOMISATION_ESTIMATORS,
            "B",
            "false_positive_rate",
            0.05,
        ),
        _target_expectation(
            "E2",
            "difference_in_means analytic coverage on scenario A (randomised) is about 0.95",
            "0.95 within 2 pooled MCSE",
            pooled,
            ("difference_in_means",),
            "A",
            "coverage_analytic",
            0.95,
        ),
        _more_biased(
            "E3",
            "On confounded scenario D, difference_in_means is biased and aipw is less so",
            "|bias(difference_in_means)| > |bias(aipw)| and exceeds 2 MCSE",
            pooled,
            "difference_in_means",
            "aipw",
            "D",
        ),
        _more_biased(
            "E4",
            "On staggered heterogeneous-effect scenario P4, twfe is biased and "
            "callaway_santanna is less so",
            "|bias(twfe)| > |bias(callaway_santanna)| and exceeds 2 MCSE",
            pooled,
            "twfe",
            "callaway_santanna",
            "P4",
        ),
        _hierarchical_coverage(baseline),
        _no_fits_on_missing_outcomes(frame),
        _slowest_is_hierarchical(frame),
        _store_is_complete(status),
    )


def expectations_markdown(results: Sequence[ExpectationResult]) -> str:
    counts = {v: sum(r.verdict == v for r in results) for v in ("met", "failed", "inconclusive")}
    lines = [
        "# Pre-registered expectations",
        "",
        f"{counts['met']} met, {counts['failed']} failed, {counts['inconclusive']} inconclusive.",
        "",
    ]
    for r in results:
        lines += [f"## {r.id} - {r.verdict.upper()}", "", r.statement, "", f"Rule: {r.rule}", ""]
        if r.measurements:
            lines += [
                "| measurement | value | MCSE | n |",
                "| --- | --- | --- | --- |",
                *(
                    f"| {m.label} | {format_value(m.value)} | {format_value(m.mcse)} | {m.n_used} |"
                    for m in r.measurements
                ),
                "",
            ]
        if r.note:
            lines += [r.note, ""]
    return "\n".join(lines)
