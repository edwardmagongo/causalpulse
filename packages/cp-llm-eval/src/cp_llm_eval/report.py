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

from cp_llm_eval.arms import ARMS
from cp_llm_eval.attribution import D0, D0_AMBIGUOUS, D0_ROUNDED
from cp_llm_eval.corpus import CorpusEntry
from cp_llm_eval.deterministic import D1_UNPARSED, DETERMINISTIC_CATEGORIES
from cp_llm_eval.graded import UNVERIFIED, GradedNarrative, GradedStatement
from cp_llm_eval.judge import JUDGE_CATEGORIES, judge_prompt_version
from cp_llm_eval.labels import Label, population, stratum
from cp_llm_eval.prereg import score_p0, score_p1
from cp_llm_eval.runner import INPUT_PRICE, OUTPUT_PRICE
from cp_llm_eval.stats import clustered_rate, paired_difference, wilson_weighted

PRIMARY_ENDPOINT = "D4 per narrative, first attempts"
BASELINE_VARIANT = "baseline"
WEIGHTING_NOTE = (
    "The judge's sensitivity and specificity and each grader's precision are weighted "
    "by stratum (D0-flagged, judge-flagged, grader-only, unflagged) back to all judged "
    "statements, with Wilson intervals at the effective sample size n_eff; they are "
    "noisy, because each unflagged label stands for many statements."
)
# Reported beside D0 and deliberately not in `DETERMINISTIC_CATEGORIES`: neither
# is a misreading. A rounded value is the right quantity at the model's own
# precision, and an ambiguous one matches several facts, so it is counted in
# neither direction -- but a high ambiguity rate would undercut what D1-D5 read
# on the ungrounded arm, which is why it is printed rather than dropped.
ATTRIBUTION_CATEGORIES: tuple[str, ...] = (D0_ROUNDED, D0_AMBIGUOUS)


def _has(statement: GradedStatement, category: str) -> bool:
    if category in DETERMINISTIC_CATEGORIES or category in ATTRIBUTION_CATEGORIES:
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
    shared = sorted(set(base) & set(other))
    paired_cases: dict[str, int] = {}

    def paired(
        name: str, score: Callable[[list[GradedNarrative]], float | None]
    ) -> dict[str, object] | None:
        pairs = []
        for case_id in shared:
            before, after = score(base[case_id]), score(other[case_id])
            if before is not None and after is not None:
                pairs.append((before, after))
        paired_cases[name] = len(pairs)
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

    primary = paired("primary", first_attempt_d4)
    exploratory = {
        category: paired(category, final(category))
        for category in (*DETERMINISTIC_CATEGORIES, *JUDGE_CATEGORIES, "any_D", "any_J")
    }
    return {
        "variant": variant,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary": primary,
        "exploratory": exploratory,
        # Cases with narratives under both prompts, and how many of them each
        # figure could pair (both sides had something to score).
        "shared_cases": len(shared),
        "paired_cases": paired_cases,
    }


def _labelled_as(label: Label, category: str) -> bool:
    """Whether the person labelled the statement a misreading of `category`."""
    if label.verdict != "misreading":
        return False
    if category == "any_J":
        return label.category in JUDGE_CATEGORIES
    return label.category == category


def _weighted(pairs: Sequence[tuple[bool, float]]) -> dict[str, object] | None:
    """A weighted proportion of `(success, weight)` pairs, its Wilson interval at
    Kish's effective sample size, the labelled count and that size."""
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    n_eff = total**2 / sum(w**2 for _, w in pairs)
    p = sum(w for success, w in pairs if success) / total
    interval = wilson_weighted(min(1.0, p), n_eff)
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "labelled": len(pairs),
        "n_eff": n_eff,
    }


def _validation(
    narratives: Sequence[GradedNarrative], labels: Sequence[Label]
) -> dict[str, object]:
    """The judge's sensitivity and specificity, and each grader's precision, against labels.

    The labels are a stratified sample (`labels.sample`: judge-flagged,
    grader-only, unflagged, about a third each), so each labelled statement is
    weighted by its stratum's population over its labelled count, the
    population being the statements the sample is drawn from.
    """
    statements = {s.statement_id: s for s in population(narratives)}
    labelled = [
        (label, statements[label.statement_id])
        for label in labels
        if label.statement_id in statements
    ]
    sizes: dict[int, int] = defaultdict(int)
    for s in statements.values():
        sizes[stratum(s)] += 1
    counts: dict[int, int] = defaultdict(int)
    for _, s in labelled:
        counts[stratum(s)] += 1
    weighted = [(label, s, sizes[stratum(s)] / counts[stratum(s)]) for label, s in labelled]

    judge: dict[str, object] = {}
    for category in (*JUDGE_CATEGORIES, "any_J"):
        judge[category] = {
            "sensitivity": _weighted(
                [(_has(s, category), w) for label, s, w in weighted
                 if _labelled_as(label, category)]
            ),
            "specificity": _weighted(
                [(not _has(s, category), w) for label, s, w in weighted
                 if not _labelled_as(label, category)]
            ),
        }
    graders: dict[str, object] = {}
    for category in DETERMINISTIC_CATEGORIES:
        graders[category] = {
            "precision": _weighted(
                [(_labelled_as(label, category), w) for label, s, w in weighted
                 if _has(s, category)]
            )
        }
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
            # The ungrounded arm's outcome. It has no verifier, so none of its
            # records is verified or refused; without this column its
            # interpretations would be counted in the total and nowhere else.
            "unverified": sum(n.outcome == UNVERIFIED for n in group),
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


def _denominators(narratives: Sequence[GradedNarrative], variant: str) -> dict[str, object]:
    """What this arm's rates are over, and whether it could have refused.

    A grounded arm can refuse, and does: `baseline` refused 1 of 100. The
    ungrounded arm structurally cannot, so its denominator is every narrative
    it produced. Three rates printed side by side would look like three
    measurements of the same thing, and they are not.
    """
    arm = ARMS.get(variant)
    return {
        "interpretations": len(narratives),
        "with_a_narrative": sum(bool(n.statements) for n in narratives),
        "judged": sum(n.judged for n in narratives),
        # None for a variant no longer in the registry, rather than a guess.
        "can_refuse": None if arm is None else arm.grounded,
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


def _j_rates(narratives: Sequence[GradedNarrative]) -> dict[str, object]:
    """Each arm's per-narrative J-rate, for one judge's verdicts."""
    variants = sorted({n.variant for n in narratives})
    return {
        v: {
            c: _final_rates([n for n in narratives if n.variant == v], c)["per_narrative"]
            for c in (*JUDGE_CATEGORIES, "any_J")
        }
        for v in variants
    }


def build_report(
    entries: Sequence[CorpusEntry],
    narratives: Sequence[GradedNarrative],
    labels: Sequence[Label],
    interpretations: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    under_other_judges: Mapping[str, Sequence[GradedNarrative]] | None = None,
) -> dict[str, object]:
    """The report as plain data. Raises `ValueError` if one variant mixes settings.

    `narratives` carry the verdicts of the judge prompt this code holds.
    `under_other_judges` maps an older judge prompt version to the same
    interpretations graded under that judge, so the report can print both sets
    of J-rates side by side. A revised judge invalidates every stored verdict
    by version, so without this the published rates would simply vanish from
    the report rather than being shown as what they are: a different
    instrument, not a different arm.
    """
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
        # Narratives whose model gave no narrative at all on the first call:
        # left out of every first-attempt figure.
        "first_attempt_missing": {
            v: sum(n.first_attempt is None for n in by_variant[v]) for v in variants
        },
        # D0 and its two companions, which are not misreadings; see
        # `ATTRIBUTION_CATEGORIES`.
        "attribution": {
            v: {c: _final_rates(by_variant[v], c) for c in ATTRIBUTION_CATEGORIES}
            for v in variants
        },
        "denominators": {v: _denominators(by_variant[v], v) for v in variants},
        "d4_truth": {v: _d4_truth(by_variant[v]) for v in variants},
        "d1_unparsed": {v: _unparsed(by_variant[v]) for v in variants},
        # The judge and its prompt are the same for every variant, so every
        # labelled statement measures it, whichever prompt produced it.
        "validation": _validation(narratives, labels),
        # The arm's own pre-registered endpoints, scored by `prereg`. P0 gates
        # P1: if the matcher finds an unsupported value in a grounded
        # narrative, where `render.substitute` put every value, the instrument
        # is wrong and P1 says nothing.
        "prereg": {
            "p0": asdict(score_p0(narratives)),
            "p1": asdict(score_p1(narratives)),
        },
        "judges": {
            "current": judge_prompt_version(),
            "j_rates": {
                judge_prompt_version(): _j_rates(narratives),
                **{v: _j_rates(ns) for v, ns in (under_other_judges or {}).items()},
            },
        },
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
    if rate.get("lower_bound") is not None:
        # At three units or fewer the bound is 0 and says nothing.
        bound = f"at least {_percent(rate['lower_bound'])}, " if rate["lower_bound"] > 0 else ""
        return f"100% ({bound}{rate['numerator']}/{rate['denominator']}, {cases})"
    if rate["se"] is not None:
        spread = f", SE {_percent(rate['se'])}"
    elif rate["clusters"] >= 2:
        # Every case has the same proportion: a clustered SE would be 0.
        spread = ", no spread across cases"
    else:
        spread = ""
    # One case gives no clustered standard error at all.
    alone = ", so no SE" if rate["se"] is None and rate["clusters"] < 2 else ""
    return (
        f"{_percent(rate['rate'])} "
        f"({rate['numerator']}/{rate['denominator']}{spread}, {cases}{alone})"
    )


def _interval_text(interval: Mapping[str, Any] | None) -> str:
    if interval is None:
        return "-"
    return (
        f"{_percent(interval['estimate'])} "
        f"[{_percent(interval['lower'])}, {_percent(interval['upper'])}], "
        f"{interval['labelled']} labelled, n_eff {interval['n_eff']:.1f}"
    )


def _difference_text(difference: Mapping[str, Any] | None) -> str:
    if difference is None:
        return "-"
    if difference.get("degenerate"):
        return (
            f"{100 * difference['mean']:+.1f} points "
            f"(no spread across {difference['pairs']} cases; no interval)"
        )
    return (
        f"{100 * difference['mean']:+.1f} points "
        f"[{100 * difference['lower']:+.1f}, {100 * difference['upper']:+.1f}], "
        f"{difference['pairs']} cases"
    )


def _interval_kind(difference: Mapping[str, Any] | None) -> str:
    """Names the interval, when the difference has one."""
    if difference is None or difference.get("degenerate"):
        return ""
    return " Normal 95% interval."


def _denominator_text(denominators: Mapping[str, Any]) -> str:
    """What the arm's rates are over, and whether a refusal could have shrunk it."""
    over = f"{denominators['with_a_narrative']} of {denominators['interpretations']}"
    can_refuse = denominators["can_refuse"]
    if can_refuse is None:
        return f"{over}, verifier unknown"
    return f"{over}, {'can refuse' if can_refuse else 'cannot refuse'}"


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
        ("Prompt", "Interpretations", "Verified", "First attempt", "Unverified", "Refused",
         "No answer", "Calls", "Cost"),
        (
            (name, str(v["interpretations"]), str(v["verified"]),
             str(v["verified_first_attempt"]), str(v["unverified"]), str(v["refused"]),
             str(v["no_answer"]), str(v["calls"]), f"${v['cost']:.2f}")
            for name, v in run["variants"].items()
        ),
    )
    judge = run["judge"]
    lines += [
        "",
        f"Judge: {judge['verdicts']} verdicts, {judge['judge_errors']} unusable, "
        f"{judge['calls']} calls, ${judge['cost']:.2f}.",
    ]

    prereg = report["prereg"]
    p0, p1 = prereg["p0"], prereg["p1"]
    lines += ["", "## Pre-registered endpoints", ""]
    lines += [
        f"P0 (the matcher finds no unsupported value in a grounded narrative) "
        f"{'passed' if p0['passed'] else 'FAILED'}: {p0['unsupported']} of "
        f"{p0['statements']} grounded statements. P1 (the ungrounded arm's "
        f"per-narrative unsupported-value rate, against {_percent(p1['threshold'])}) "
        f"is {p1['verdict']}: {_rate_text(p1['rate'])}. `prereg.md` holds both in full.",
        "",
    ]

    lines += ["", "## Values the fact table does not support", ""]
    lines += [
        "D0 is the share of numeric spans that match no fact in the table. On a grounded "
        "arm it is zero by construction -- every value there was substituted from the "
        "table -- so a non-zero D0 on a grounded arm means the matcher is at fault, not "
        "the model. The two companion figures are not misreadings: a rounded value is "
        "the right quantity at the model's own precision, and an ambiguous one matches "
        "several facts and is counted in neither direction.",
        "",
    ]
    lines += _table(
        ("Prompt", "D0 per statement", "D0 per narrative", "Rounded", "Ambiguous", "Over"),
        (
            (
                variant,
                _rate_text(report["rates"][variant][D0]["per_statement"]),
                _rate_text(report["rates"][variant][D0]["per_narrative"]),
                _rate_text(report["attribution"][variant][D0_ROUNDED]["per_statement"]),
                _rate_text(report["attribution"][variant][D0_AMBIGUOUS]["per_statement"]),
                _denominator_text(report["denominators"][variant]),
            )
            for variant in report["rates"]
        ),
    )

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
    missing = ", ".join(f"{v} {n}" for v, n in report["first_attempt_missing"].items())
    lines += ["", f"Narratives with no first attempt, excluded from these figures: {missing}."]

    lines += ["", "## The graders against the labels", ""]
    lines += [
        f"{validation['labelled']} labelled statements. {WEIGHTING_NOTE}",
        "",
    ]
    lines += _table(
        ("Grader", "Precision"),
        ((c, _interval_text(m["precision"])) for c, m in validation["graders"].items()),
    )

    judges = report["judges"]
    if len(judges["j_rates"]) > 1:
        lines += ["", "## The same narratives under each judge (per narrative)", ""]
        lines += [
            "One judge prompt per column block, named by its version. Rates from "
            "different judge prompts are different instruments and are not comparable "
            f"with each other; {judges['current']} is the prompt this report's other "
            "tables use.",
            "",
        ]
        for version, by_variant in judges["j_rates"].items():
            current = " (current)" if version == judges["current"] else ""
            lines += ["", f"### Judge {version}{current}", ""]
            judged = (*JUDGE_CATEGORIES, "any_J")
            lines += _table(
                ("Prompt", *judged),
                (
                    (variant, *(_rate_text(rates[c]) for c in judged))
                    for variant, rates in by_variant.items()
                ),
            )

    for comparison in report["comparison"]:
        shared, paired = comparison["shared_cases"], comparison["paired_cases"]
        over = {name: f"{n} of {shared} shared cases" for name, n in paired.items()}
        lines += [
            "",
            f"## Comparison: {comparison['variant']} minus baseline",
            "",
            f"Primary endpoint, fixed before any paid call ({comparison['primary_endpoint']}): "
            f"{_difference_text(comparison['primary'])}.{_interval_kind(comparison['primary'])} "
            f"Paired by case over {over['primary']}.",
            "",
            "Exploratory, final narratives, per narrative:",
            "",
        ]
        lines += _table(
            ("Category", "Difference", "Paired"),
            ((c, _difference_text(d), over[c]) for c, d in comparison["exploratory"].items()),
        )
    return "\n".join(lines) + "\n"


def write_report(report: Mapping[str, Any], out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    markdown, data = out / "report.md", out / "report.json"
    markdown.write_text(render_markdown(report), encoding="utf-8")
    data.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return markdown, data
