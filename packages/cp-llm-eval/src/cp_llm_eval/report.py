"""The study's report: rates of each misreading, the judge's measured error, the comparison.

`build_report` computes everything as plain data (the JSON written beside the
markdown); `render_markdown` only formats it. Every rate is clustered by case
(`cp_llm_eval.stats`). A judge-graded rate is reported over judged narratives
only, next to the judge's sensitivity and specificity for that category.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cp_llm_eval.corpus import CorpusEntry
from cp_llm_eval.deterministic import D1_UNPARSED, DETERMINISTIC_CATEGORIES
from cp_llm_eval.graded import GradedNarrative, GradedStatement
from cp_llm_eval.judge import JUDGE_CATEGORIES
from cp_llm_eval.labels import Label
from cp_llm_eval.runner import INPUT_PRICE, OUTPUT_PRICE
from cp_llm_eval.stats import clustered_rate, paired_difference, wilson

PRIMARY_ENDPOINT = "D4 per narrative, first attempts"
BASELINE_VARIANT = "baseline"


def _has(statement: GradedStatement, category: str) -> bool:
    if category in DETERMINISTIC_CATEGORIES:
        return any(f.category == category for f in statement.deterministic)
    if category in JUDGE_CATEGORIES:
        return statement.judge is not None and any(f.category == category for f in statement.judge)
    if category == "any_D":
        return any(_has(statement, c) for c in DETERMINISTIC_CATEGORIES)
    if category == "any_J":
        return any(_has(statement, c) for c in JUDGE_CATEGORIES)
    raise ValueError(f"unknown category {category!r}")


def _rate_or_none(groups: Iterable[tuple[int, int]]) -> dict[str, object] | None:
    groups = list(groups)
    if sum(d for _, d in groups) == 0:
        return None
    return asdict(clustered_rate(groups))


def _by_case(narratives: Iterable[GradedNarrative]) -> dict[str, list[GradedNarrative]]:
    cases: dict[str, list[GradedNarrative]] = defaultdict(list)
    for narrative in narratives:
        cases[narrative.case_id].append(narrative)
    return cases


def _final_rates(narratives: Sequence[GradedNarrative], category: str) -> dict[str, object]:
    """Per statement and per narrative, over verified narratives (judged ones for a J category)."""
    judge_side = category in JUDGE_CATEGORIES or category == "any_J"
    pool = [n for n in narratives if n.statements and (n.judged or not judge_side)]
    cases = _by_case(pool)
    per_statement = [
        (
            sum(_has(s, category) for n in group for s in n.statements),
            sum(len(n.statements) for n in group),
        )
        for group in cases.values()
    ]
    per_narrative = [
        (sum(any(_has(s, category) for s in n.statements) for n in group), len(group))
        for group in cases.values()
    ]
    return {
        "per_statement": _rate_or_none(per_statement),
        "per_narrative": _rate_or_none(per_narrative),
    }


def _first_attempt_rate(
    narratives: Sequence[GradedNarrative], category: str
) -> dict[str, object] | None:
    pool = [n for n in narratives if n.first_attempt is not None]
    groups = []
    for group in _by_case(pool).values():
        hits = 0
        for narrative in group:
            assert narrative.first_attempt is not None
            hits += any(f.category == category for flags in narrative.first_attempt for f in flags)
        groups.append((hits, len(group)))
    return _rate_or_none(groups)


def _first_attempt_share(narrative: GradedNarrative, category: str) -> float:
    assert narrative.first_attempt is not None
    return float(any(f.category == category for flags in narrative.first_attempt for f in flags))


def _comparison(
    narratives: Sequence[GradedNarrative], variant: str
) -> dict[str, object] | None:
    """`variant` minus baseline, paired by case: the primary endpoint, then the rest."""
    base = _by_case(n for n in narratives if n.variant == BASELINE_VARIANT)
    other = _by_case(n for n in narratives if n.variant == variant)

    def paired(score: Callable[[list[GradedNarrative]], float | None]) -> dict[str, object] | None:
        pairs = []
        for case_id in sorted(set(base) & set(other)):
            before, after = score(base[case_id]), score(other[case_id])
            if before is not None and after is not None:
                pairs.append((before, after))
        return asdict(paired_difference(pairs)) if len(pairs) >= 2 else None

    def first_attempt_d4(group: list[GradedNarrative]) -> float | None:
        pool = [n for n in group if n.first_attempt is not None]
        return sum(_first_attempt_share(n, "D4") for n in pool) / len(pool) if pool else None

    def final(category: str) -> Callable[[list[GradedNarrative]], float | None]:
        judge_side = category in JUDGE_CATEGORIES or category == "any_J"

        def score(group: list[GradedNarrative]) -> float | None:
            pool = [n for n in group if n.statements and (n.judged or not judge_side)]
            if not pool:
                return None
            return sum(any(_has(s, category) for s in n.statements) for n in pool) / len(pool)

        return score

    return {
        "variant": variant,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary": paired(first_attempt_d4),
        "exploratory": {
            category: paired(final(category))
            for category in (*DETERMINISTIC_CATEGORIES, *JUDGE_CATEGORIES, "any_D", "any_J")
        },
    }


def _labelled_as(label: Label, category: str) -> bool:
    """Whether the person labelled the statement a misreading of `category`."""
    if label.verdict != "misreading":
        return False
    if category == "any_J":
        return label.category in JUDGE_CATEGORIES
    return label.category == category


def _validation(
    narratives: Sequence[GradedNarrative], labels: Sequence[Label]
) -> dict[str, object]:
    """The judge's sensitivity and specificity, and each grader's precision, against labels."""
    statements = {s.statement_id: s for n in narratives for s in n.statements}
    labelled = [
        (label, statements[label.statement_id])
        for label in labels
        if label.statement_id in statements
    ]

    def interval(successes: int, n: int) -> dict[str, object] | None:
        return asdict(wilson(successes, n)) if n else None

    judged = [(label, s) for label, s in labelled if s.judge is not None]
    judge: dict[str, object] = {}
    for category in (*JUDGE_CATEGORIES, "any_J"):
        positives = [(label, s) for label, s in judged if _labelled_as(label, category)]
        negatives = [(label, s) for label, s in judged if not _labelled_as(label, category)]
        judge[category] = {
            "sensitivity": interval(sum(_has(s, category) for _, s in positives), len(positives)),
            "specificity": interval(
                sum(not _has(s, category) for _, s in negatives), len(negatives)
            ),
        }
    graders: dict[str, object] = {}
    for category in DETERMINISTIC_CATEGORIES:
        flagged = [label for label, s in labelled if _has(s, category)]
        misread = sum(_labelled_as(label, category) for label in flagged)
        graders[category] = {"precision": interval(misread, len(flagged))}
    return {"labelled": len(labelled), "judge": judge, "graders": graders}


def _run(
    narratives: Sequence[GradedNarrative],
    interpretations: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    def spent(records: Iterable[Mapping[str, Any]]) -> tuple[int, float]:
        calls = [c for r in records for c in r["calls"]]
        dollars = sum(
            c["input_tokens"] * INPUT_PRICE + c["output_tokens"] * OUTPUT_PRICE for c in calls
        )
        return len(calls), dollars

    variants: dict[str, object] = {}
    for variant in sorted({n.variant for n in narratives}):
        group = [n for n in narratives if n.variant == variant]
        records = [r for r in interpretations if r["key"]["variant"] == variant]
        calls, dollars = spent(records)
        variants[variant] = {
            "interpretations": len(group),
            "verified": sum(n.outcome == "verified" for n in group),
            "refused": sum(n.outcome == "refused" for n in group),
            "no_answer": sum(n.outcome == "no_answer" for n in group),
            "verified_first_attempt": sum(
                n.outcome == "verified" and n.attempts == 1 for n in group
            ),
            "calls": calls,
            "cost": dollars,
        }
    judge_calls, judge_cost = spent(verdicts)
    return {
        "variants": variants,
        "judge": {
            "verdicts": len(verdicts),
            "judge_errors": sum(v["outcome"] == "judge_error" for v in verdicts),
            "calls": judge_calls,
            "cost": judge_cost,
        },
    }


def _d4_truth(narratives: Sequence[GradedNarrative]) -> dict[str, int]:
    counts = {"true": 0, "false": 0, "undetermined": 0}
    for narrative in narratives:
        for statement in narrative.statements:
            for flag in statement.deterministic:
                if flag.category == "D4" and flag.truth is not None:
                    counts[flag.truth] += 1
    return counts


def _unparsed(narratives: Sequence[GradedNarrative]) -> int:
    return sum(
        any(f.category == D1_UNPARSED for f in s.deterministic)
        for n in narratives
        for s in n.statements
    )


def _check_one_setting_per_variant(interpretations: Sequence[Mapping[str, Any]]) -> None:
    """Raise if a variant name covers records of more than one model, effort or prompt."""
    settings: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for record in interpretations:
        key = record["key"]
        settings[key["variant"]].add((key["model"], key["effort"], key["prompt_version"]))
    for variant, found in sorted(settings.items()):
        if len(found) > 1:
            raise ValueError(
                f"variant {variant!r} has records from {len(found)} settings of "
                f"(model, effort, prompt_version): {sorted(found)}; report one setting per variant"
            )


def build_report(
    entries: Sequence[CorpusEntry],
    narratives: Sequence[GradedNarrative],
    labels: Sequence[Label],
    interpretations: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """The report as plain data. Raises `ValueError` if one variant mixes settings."""
    _check_one_setting_per_variant(interpretations)
    categories = (*DETERMINISTIC_CATEGORIES, *JUDGE_CATEGORIES, "any_D", "any_J")
    variants = sorted({n.variant for n in narratives})
    designs = sorted({e.design for e in entries})
    by_variant = {v: [n for n in narratives if n.variant == v] for v in variants}
    return {
        "corpus": [
            {
                "case_id": e.case_id,
                "design": e.design,
                "group": e.group,
                "reason": e.reason,
                "requirements": list(e.requirements),
            }
            for e in entries
        ],
        "run": _run(narratives, interpretations, verdicts),
        "rates": {v: {c: _final_rates(by_variant[v], c) for c in categories} for v in variants},
        "by_design": {
            v: {
                d: {
                    c: _final_rates([n for n in by_variant[v] if n.design == d], c)["per_narrative"]
                    for c in categories
                }
                for d in designs
            }
            for v in variants
        },
        "first_attempt": {
            v: {c: _first_attempt_rate(by_variant[v], c) for c in DETERMINISTIC_CATEGORIES}
            for v in variants
        },
        "d4_truth": {v: _d4_truth(by_variant[v]) for v in variants},
        "d1_unparsed": {v: _unparsed(by_variant[v]) for v in variants},
        # The judge and its prompt are the same for every variant, so every
        # labelled statement measures it, whichever prompt produced it.
        "validation": _validation(narratives, labels),
        "comparison": [
            _comparison(narratives, v) for v in variants if v != BASELINE_VARIANT
        ],
    }


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _rate_text(rate: Mapping[str, Any] | None) -> str:
    if rate is None:
        return "-"
    cases = f"{rate['clusters']} case" + ("" if rate["clusters"] == 1 else "s")
    if rate["upper_bound"] is not None:
        return (
            f"0% (at most {_percent(rate['upper_bound'])}, 0/{rate['denominator']}, {cases})"
        )
    spread = f", SE {_percent(rate['se'])}" if rate["se"] is not None else ""
    return (
        f"{_percent(rate['rate'])} ({rate['numerator']}/{rate['denominator']}{spread}, {cases})"
    )


def _interval_text(interval: Mapping[str, Any] | None) -> str:
    if interval is None:
        return "-"
    return (
        f"{_percent(interval['estimate'])} "
        f"[{_percent(interval['lower'])}, {_percent(interval['upper'])}], n={interval['n']}"
    )


def _difference_text(difference: Mapping[str, Any] | None) -> str:
    if difference is None:
        return "-"
    return (
        f"{100 * difference['mean']:+.1f} points "
        f"[{100 * difference['lower']:+.1f}, {100 * difference['upper']:+.1f}], "
        f"{difference['pairs']} cases"
    )


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """The report as markdown. Formatting only: every number comes from `report`."""
    lines = ["# How often a verified interpretation misreads its analysis", ""]
    corpus = report["corpus"]
    lines += [f"Corpus: {len(corpus)} cases.", ""]
    lines += _table(
        ("Case", "Design", "Group", "Why"),
        ((c["case_id"], c["design"], c["group"], c["reason"]) for c in corpus),
    )

    run = report["run"]
    lines += ["", "## The run", ""]
    lines += _table(
        ("Prompt", "Interpretations", "Verified", "First attempt", "Refused", "No answer",
         "Calls", "Cost"),
        (
            (name, str(v["interpretations"]), str(v["verified"]),
             str(v["verified_first_attempt"]), str(v["refused"]), str(v["no_answer"]),
             str(v["calls"]), f"${v['cost']:.2f}")
            for name, v in run["variants"].items()
        ),
    )
    judge = run["judge"]
    lines += [
        "",
        f"Judge: {judge['verdicts']} verdicts, {judge['judge_errors']} unusable, "
        f"{judge['calls']} calls, ${judge['cost']:.2f}.",
    ]

    validation = report["validation"]
    for variant, by_category in report["rates"].items():
        lines += ["", f"## Final narratives: {variant}", ""]
        rows = []
        for category, rate in by_category.items():
            measured = validation["judge"].get(category, {})
            rows.append(
                (category, _rate_text(rate["per_statement"]), _rate_text(rate["per_narrative"]),
                 _interval_text(measured.get("sensitivity")),
                 _interval_text(measured.get("specificity")))
            )
        lines += _table(
            ("Category", "Per statement", "Per narrative", "Judge sensitivity",
             "Judge specificity"),
            rows,
        )
        truth = report["d4_truth"][variant]
        lines += [
            "",
            f"D4 summaries: {truth['true']} true, {truth['false']} false, "
            f"{truth['undetermined']} undetermined. Comparisons D1 left to the judge: "
            f"{report['d1_unparsed'][variant]}.",
        ]

    lines += ["", "## By design (per narrative)"]
    for variant, designs in report["by_design"].items():
        names = list(next(iter(designs.values())))
        lines += ["", f"### {variant}", ""]
        lines += _table(
            ("Design", *names),
            ((design, *(_rate_text(rates[c]) for c in names)) for design, rates in designs.items()),
        )

    lines += ["", "## First attempts (per narrative)", ""]
    lines += _table(
        ("Prompt", *DETERMINISTIC_CATEGORIES),
        (
            (variant, *(_rate_text(rates[c]) for c in DETERMINISTIC_CATEGORIES))
            for variant, rates in report["first_attempt"].items()
        ),
    )

    lines += ["", "## The graders against the labels", ""]
    lines += [f"{validation['labelled']} labelled statements.", ""]
    lines += _table(
        ("Grader", "Precision"),
        ((c, _interval_text(m["precision"])) for c, m in validation["graders"].items()),
    )

    for comparison in report["comparison"]:
        lines += [
            "",
            f"## Comparison: {comparison['variant']} minus baseline",
            "",
            f"Primary endpoint, fixed before any paid call ({comparison['primary_endpoint']}): "
            f"{_difference_text(comparison['primary'])}. Normal 95% interval, paired by case.",
            "",
            "Exploratory, final narratives, per narrative:",
            "",
        ]
        lines += _table(
            ("Category", "Difference"),
            ((c, _difference_text(d)) for c, d in comparison["exploratory"].items()),
        )
    return "\n".join(lines) + "\n"


def write_report(report: Mapping[str, Any], out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    markdown, data = out / "report.md", out / "report.json"
    markdown.write_text(render_markdown(report), encoding="utf-8")
    data.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return markdown, data
