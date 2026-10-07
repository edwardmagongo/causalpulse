"""The hand-labelled sample that the judge and the graders are measured against.

`export_csv` writes a seeded, stratified sample of final statements for a
person to label blind: the file holds the statement, what it cites and the
part of the table around it, and never the judge's verdict or a grader's
flag. `save_sample` records which statements were exported. `import_csv`
reads the labelled file back and refuses it unless it labels exactly that
exported sample, every row from the allowed sets.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from cp_llm.facts import FactTable
from cp_llm.render import format_fact, placeholder_ids

from cp_llm_eval.deterministic import DETERMINISTIC_CATEGORIES
from cp_llm_eval.graded import GradedNarrative, GradedStatement
from cp_llm_eval.judge import JUDGE_CATEGORIES

COLUMNS = (
    "statement_id",
    "case_id",
    "statement",
    "cited_facts",
    "context",
    "verdict",
    "category",
    "note",
)
VERDICTS = ("ok", "misreading")
CATEGORIES = (*DETERMINISTIC_CATEGORIES, *JUDGE_CATEGORIES)
DEFAULT_SIZE = 150


class LabelError(ValueError):
    """The labelled file cannot be used; the message lists every problem found."""


@dataclass(frozen=True)
class Label:
    statement_id: str
    verdict: str
    category: str
    note: str


def _stratum(statement: GradedStatement) -> int:
    """0 if the judge flagged it, 1 if only a grader did, 2 if nothing did."""
    if statement.judge:
        return 0
    if any(f.category in DETERMINISTIC_CATEGORIES for f in statement.deterministic):
        return 1
    return 2


def sample(
    narratives: Sequence[GradedNarrative], size: int = DEFAULT_SIZE, seed: int = 0
) -> list[GradedStatement]:
    """About a third each from judge-flagged, grader-flagged and unflagged statements.

    Drawn from judged final narratives only, since a statement the judge never
    saw cannot measure it. A stratum with too few statements gives all it has
    and the shortfall is drawn from what is left. The result is shuffled, so
    the order of the rows does not reveal the stratum.
    """
    rng = random.Random(seed)
    statements = [s for n in narratives if n.judged for s in n.statements]
    chosen: list[GradedStatement] = []
    for index in range(3):
        stratum = [s for s in statements if _stratum(s) == index]
        rng.shuffle(stratum)
        chosen.extend(stratum[: size // 3])
    taken = {s.statement_id for s in chosen}
    left = [s for s in statements if s.statement_id not in taken]
    rng.shuffle(left)
    chosen.extend(left[: max(0, size - len(chosen))])
    rng.shuffle(chosen)
    return chosen


def _cited(statement: GradedStatement, facts: FactTable) -> str:
    parts = []
    for fact_id in dict.fromkeys(placeholder_ids(statement.raw)):
        fact = facts.get(fact_id)
        if fact is not None:
            parts.append(f"{fact_id} = {format_fact(fact)}")
    return "; ".join(parts)


def _context(statement: GradedStatement, facts: FactTable) -> str:
    """Each cited method's estimate, band and check statuses, as a reader would look them up."""
    methods = []
    for fact_id in placeholder_ids(statement.raw):
        fact = facts.get(fact_id)
        if fact is None:
            continue
        subject = fact.id if fact.kind == "name" and fact.id.startswith("est.") else fact.subject
        if subject is not None and subject.startswith("est.") and subject not in methods:
            methods.append(subject)
    lines = []
    for subject in methods:
        method = subject.split(".")[1]

        def shown(fact_id: str) -> str:
            fact = facts.get(fact_id)
            return format_fact(fact) if fact is not None else "-"

        checks = [
            f"{shown(f.id.rsplit('.', 1)[0] + '.name')}={format_fact(f)}"
            for f in facts
            if f.id.startswith(f"check.{method}.") and f.id.endswith(".status")
        ]
        lines.append(
            f"{method}: point {shown(f'est.{method}.point')} "
            f"[{shown(f'est.{method}.ci_lower')}, {shown(f'est.{method}.ci_upper')}], "
            f"band {shown(f'score.{method}.band')} (ceiling {shown(f'score.{method}.ceiling')}), "
            f"checks {', '.join(checks) or 'none'}"
        )
    return " | ".join(lines)


def export_csv(
    statements: Sequence[GradedStatement], facts_by_case: Mapping[str, FactTable], path: Path
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for statement in statements:
            facts = facts_by_case[statement.case_id]
            writer.writerow(
                {
                    "statement_id": statement.statement_id,
                    "case_id": statement.case_id,
                    "statement": statement.rendered,
                    "cited_facts": _cited(statement, facts),
                    "context": _context(statement, facts),
                    "verdict": "",
                    "category": "",
                    "note": "",
                }
            )


def save_sample(statements: Sequence[GradedStatement], path: Path) -> None:
    """The exported statement ids, in export order, for `import_csv` to check against."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [statement.statement_id for statement in statements]
    path.write_text(json.dumps(ids, indent=1) + "\n", encoding="utf-8")


def load_sample(path: Path) -> list[str]:
    return [str(statement_id) for statement_id in json.loads(path.read_text(encoding="utf-8"))]


def import_csv(path: Path, sample: Sequence[str]) -> list[Label]:
    """Every row's label, or a `LabelError` naming each unusable row or missing statement."""
    known = set(sample)
    problems = []
    labels = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise LabelError(f"{path} has no column(s) {sorted(missing)}")
        seen: set[str] = set()
        for line, row in enumerate(reader, start=2):
            statement_id = (row["statement_id"] or "").strip()
            verdict = (row["verdict"] or "").strip().lower()
            category = (row["category"] or "").strip().upper()
            if statement_id not in known:
                problems.append(f"line {line}: unknown statement_id {statement_id!r}")
            if statement_id in seen:
                problems.append(f"line {line}: {statement_id} is labelled twice")
            seen.add(statement_id)
            if verdict not in VERDICTS:
                problems.append(f"line {line}: verdict {verdict!r} is not one of {VERDICTS}")
            if verdict == "misreading" and category not in CATEGORIES:
                problems.append(f"line {line}: a misreading needs a category from {CATEGORIES}")
            if verdict == "ok" and category:
                problems.append(f"line {line}: a statement labelled ok has category {category!r}")
            labels.append(Label(statement_id, verdict, category, (row["note"] or "").strip()))
    absent = [statement_id for statement_id in sample if statement_id not in seen]
    if absent:
        problems.append(f"{len(absent)} sampled statement(s) not labelled: {absent}")
    if problems:
        raise LabelError("; ".join(problems))
    return labels


def save(labels: Sequence[Label], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(label) for label in labels]
    path.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")


def load(path: Path) -> list[Label]:
    if not path.exists():
        return []
    return [Label(**record) for record in json.loads(path.read_text(encoding="utf-8"))]
