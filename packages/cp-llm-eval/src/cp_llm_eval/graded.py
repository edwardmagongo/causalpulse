"""Every stored interpretation joined to its grades, as the report and the labels read it.

The deterministic graders run here, each time, because they are free and a
corrected grader must reach every stored narrative. The judge's verdicts are
read from the store, matched to their interpretation by key and to the
current judge prompt by version, so a verdict from an older judge prompt is
never mixed in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cp_llm.facts import FactTable
from cp_llm.narrative import Narrative

from cp_llm_eval.corpus import CorpusEntry, Design
from cp_llm_eval.deterministic import Flag, grade_narrative
from cp_llm_eval.judge import Finding, judge_prompt_version, rendered_statements
from cp_llm_eval.runner import interpretation_key
from cp_llm_eval.store import digest


@dataclass(frozen=True)
class GradedStatement:
    statement_id: str
    interpretation: str
    case_id: str
    section: str
    position: int
    raw: str
    rendered: str
    deterministic: tuple[Flag, ...]
    # None when the narrative was not judged, or the judge's reply was unusable.
    judge: tuple[Finding, ...] | None


@dataclass(frozen=True)
class GradedNarrative:
    interpretation: str
    case_id: str
    design: Design
    variant: str
    replicate: int
    outcome: str
    attempts: int
    # The final, verified narrative's statements; empty unless `outcome` is verified.
    statements: tuple[GradedStatement, ...]
    # The deterministic flags of the first attempt's statements, or None if
    # the model gave no narrative at all.
    first_attempt: tuple[tuple[Flag, ...], ...] | None
    judge_error: bool

    @property
    def judged(self) -> bool:
        return bool(self.statements) and all(s.judge is not None for s in self.statements)


def _first_attempt(
    record: Mapping[str, object], facts: FactTable
) -> tuple[tuple[Flag, ...], ...] | None:
    attempts = record["attempts"]
    assert isinstance(attempts, list)
    if not attempts:
        return None
    narrative = Narrative.model_validate(attempts[0]["narrative"])
    return tuple(grade.flags for grade in grade_narrative(narrative, facts))


def grade_records(
    entries: Sequence[CorpusEntry],
    interpretations: Sequence[Mapping[str, object]],
    verdicts: Sequence[Mapping[str, object]],
    judge_model: str,
) -> tuple[GradedNarrative, ...]:
    """Join each interpretation record to its case, its grades and its verdict."""
    by_case = {entry.case_id: entry for entry in entries}
    version = judge_prompt_version()
    by_interpretation = {
        str(v["interpretation"]): v
        for v in verdicts
        if v["judge_model"] == judge_model and v["judge_prompt_version"] == version
    }
    graded = []
    for record in interpretations:
        key_fields = record["key"]
        assert isinstance(key_fields, dict)
        key = interpretation_key(record)
        entry = by_case[str(key_fields["case_id"])]
        verdict = by_interpretation.get(key)
        judged = verdict is not None and verdict["outcome"] == "judged"
        statements: tuple[GradedStatement, ...] = ()
        if record["outcome"] == "verified":
            interpretation = record["interpretation"]
            assert isinstance(interpretation, dict)
            narrative = Narrative.model_validate(interpretation["narrative"])
            grades = grade_narrative(narrative, entry.facts)
            rendered = rendered_statements(dict(record))
            findings: list[list[dict[str, object]]] | None = None
            if judged and verdict is not None:
                stored = verdict["statements"]
                assert isinstance(stored, list)
                findings = stored
            statements = tuple(
                GradedStatement(
                    statement_id=digest({"interpretation": key, "position": grade.position}),
                    interpretation=key,
                    case_id=entry.case_id,
                    section=grade.section,
                    position=grade.position,
                    raw=grade.statement,
                    rendered=rendered[grade.position][1],
                    deterministic=grade.flags,
                    judge=(
                        tuple(Finding.model_validate(f) for f in findings[grade.position])
                        if findings is not None
                        else None
                    ),
                )
                for grade in grades
            )
        attempts = record["attempts"]
        replicate = key_fields["replicate"]
        assert isinstance(attempts, list) and isinstance(replicate, int)
        graded.append(
            GradedNarrative(
                interpretation=key,
                case_id=entry.case_id,
                design=entry.design,
                variant=str(key_fields["variant"]),
                replicate=replicate,
                outcome=str(record["outcome"]),
                attempts=len(attempts),
                statements=statements,
                first_attempt=_first_attempt(record, entry.facts),
                judge_error=verdict is not None and verdict["outcome"] == "judge_error",
            )
        )
    return tuple(graded)
