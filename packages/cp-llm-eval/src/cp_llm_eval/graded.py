"""Every stored interpretation joined to its grades, as the report and the labels read it.

The deterministic graders run here, each time, because they are free and a
corrected grader must reach every stored narrative. The judge's verdicts are
read from the store, matched to their interpretation by key and to the
current judge prompt by version, so a verdict from an older judge prompt is
never mixed in.

A narrative from the ungrounded arm holds values rather than placeholders, so
it is matched against the fact table first and D1-D5 read the reconstructed
form (`attribution.grade_reconstructed`). Without that the graders would find
nothing to read in it and the arm would score a perfect zero on every
category -- not "no misreadings" but "no input".

The matcher runs on every arm, so that its flags mean the same thing
everywhere: on an unverified narrative over the text the model wrote, and on
a verified one over the rendered text a reader is shown. On a verified
narrative it must find nothing, because every value there came from
`render.substitute` and so is some fact's rendered form exactly -- which is
what the control arm's P0 gate checks before a call of it is billed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cp_llm.facts import FactTable
from cp_llm.narrative import Narrative

from cp_llm_eval.attribution import flags as attribution_flags
from cp_llm_eval.attribution import grade_reconstructed
from cp_llm_eval.corpus import CorpusEntry, Design, check_facts_match
from cp_llm_eval.deterministic import Flag, StatementGrade, grade_narrative
from cp_llm_eval.judge import Finding, judge_prompt_version, rendered_statements
from cp_llm_eval.runner import interpretation_key
from cp_llm_eval.store import digest
from cp_llm_eval.ungrounded import UNVERIFIED

__all__ = ["UNVERIFIED", "GradedNarrative", "GradedStatement", "grade_records"]


@dataclass(frozen=True)
class GradedStatement:
    statement_id: str
    interpretation: str
    case_id: str
    section: str
    position: int
    raw: str
    rendered: str
    # D1-D5, and the matcher's own flags for the values in this statement.
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
    # The narrative the arm ended on: the verified one, or the ungrounded
    # arm's single unverified reply. Empty when the model refused or did not
    # answer, since there is then no narrative to grade.
    statements: tuple[GradedStatement, ...]
    # The deterministic flags of the first attempt's statements, or None if
    # the model gave no narrative at all. D1-D5 only on a grounded arm: an
    # attempt that was never verified was never rendered, so there is no
    # rendered text for the matcher to read.
    first_attempt: tuple[tuple[Flag, ...], ...] | None
    judge_error: bool

    @property
    def judged(self) -> bool:
        return bool(self.statements) and all(s.judge is not None for s in self.statements)


def _grades(
    narrative: Narrative, facts: FactTable, outcome: str
) -> tuple[StatementGrade, ...]:
    """Grade `narrative` the way this arm's narratives are written.

    An unverified narrative holds the model's own values, so it is matched
    against the table and graded on the reconstructed form; a verified one
    already cites its facts and is graded as it stands.
    """
    if outcome == UNVERIFIED:
        return grade_reconstructed(narrative, facts)
    return grade_narrative(narrative, facts)


def _flags(
    grade: StatementGrade, rendered: str, facts: FactTable, outcome: str
) -> tuple[Flag, ...]:
    """One statement's flags: D1-D5, and the matcher's on the text a reader sees.

    On an unverified narrative the matcher has already run, over the text the
    model wrote, and its flags are in `grade.flags`. On a verified one the
    values are behind placeholders, so it runs here over the rendered form.
    """
    if outcome == UNVERIFIED:
        return grade.flags
    return grade.flags + attribution_flags(rendered, facts)


def _first_attempt(
    record: Mapping[str, object], facts: FactTable
) -> tuple[tuple[Flag, ...], ...] | None:
    attempts = record["attempts"]
    assert isinstance(attempts, list)
    if not attempts:
        return None
    narrative = Narrative.model_validate(attempts[0]["narrative"])
    grades = _grades(narrative, facts, str(record["outcome"]))
    return tuple(grade.flags for grade in grades)


def grade_records(
    entries: Sequence[CorpusEntry],
    interpretations: Sequence[Mapping[str, object]],
    verdicts: Sequence[Mapping[str, object]],
    judge_model: str,
) -> tuple[GradedNarrative, ...]:
    """Join each interpretation record to its case, its grades and its verdict.

    Raises `ValueError` if a record was made from a fact table other than the
    corpus entry's for its case.
    """
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
        check_facts_match(record, entry)
        verdict = by_interpretation.get(key)
        judged = verdict is not None and verdict["outcome"] == "judged"
        statements: tuple[GradedStatement, ...] = ()
        outcome = str(record["outcome"])
        if outcome in ("verified", UNVERIFIED):
            interpretation = record["interpretation"]
            assert isinstance(interpretation, dict)
            narrative = Narrative.model_validate(interpretation["narrative"])
            grades = _grades(narrative, entry.facts, outcome)
            # An unverified narrative was never substituted into, because its
            # values were never placeholders: it is already its own rendered
            # form, and that is what a reader of this arm sees.
            rendered = (
                narrative.statements()
                if outcome == UNVERIFIED
                else rendered_statements(dict(record))
            )
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
                    deterministic=_flags(
                        grade, rendered[grade.position][1], entry.facts, outcome
                    ),
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
                outcome=outcome,
                attempts=len(attempts),
                statements=statements,
                first_attempt=_first_attempt(record, entry.facts),
                judge_error=verdict is not None and verdict["outcome"] == "judge_error",
            )
        )
    return tuple(graded)
