from cp_llm_eval.graded import grade_records
from cp_llm_eval.judge import VERDICTS
from cp_llm_eval.runner import INTERPRETATIONS

from llm_eval.fakes import D4_STATEMENT


def _graded(entry, stored, judge_model="claude-opus-5-5"):
    return grade_records(
        [entry], stored.records(INTERPRETATIONS), stored.records(VERDICTS), judge_model
    )


def _by(graded, variant, replicate):
    (narrative,) = [n for n in graded if (n.variant, n.replicate) == (variant, replicate)]
    return narrative


def test_every_stored_interpretation_is_graded_with_its_outcome(entry, stored):
    graded = _graded(entry, stored)

    assert sorted((n.variant, n.replicate, n.outcome) for n in graded) == [
        ("baseline", 1, "verified"),
        ("baseline", 2, "verified"),
        ("cite-each-status", 1, "refused"),
        ("cite-each-status", 2, "verified"),
    ]
    assert {n.design for n in graded} == {"cross_section"}


def test_a_verified_narrative_carries_rendered_statements_and_the_judges_findings(entry, stored):
    narrative = _by(_graded(entry, stored), "baseline", 1)

    assert narrative.judged and not narrative.judge_error
    first, *rest = narrative.statements
    assert [f.category for f in first.judge] == ["J1"]
    assert all(s.judge == () for s in rest)
    assert all("{" not in s.rendered for s in narrative.statements)
    assert len({s.statement_id for s in narrative.statements}) == len(narrative.statements)


def test_a_refused_narrative_has_no_final_statements_but_a_graded_first_attempt(entry, stored):
    narrative = _by(_graded(entry, stored), "cite-each-status", 1)

    assert (narrative.statements, narrative.attempts) == ((), 2)
    assert narrative.first_attempt is not None
    assert not narrative.judged


def test_a_deterministic_flag_reaches_the_final_statement_it_belongs_to(entry, stored):
    narrative = _by(_graded(entry, stored), "cite-each-status", 2)

    (flagged,) = [s for s in narrative.statements if s.deterministic]
    assert flagged.raw == D4_STATEMENT
    assert [(f.category, f.truth) for f in flagged.deterministic] == [("D4", "true")]


def test_a_verdict_from_another_judge_model_is_not_used(entry, stored):
    graded = _graded(entry, stored, judge_model="some-other-model")

    assert not any(n.judged for n in graded)
    assert all(s.judge is None for n in graded for s in n.statements)


def test_a_verdict_from_an_older_judge_prompt_is_not_used(entry, stored, monkeypatch):
    from cp_llm_eval import graded as graded_module

    monkeypatch.setattr(graded_module, "judge_prompt_version", lambda: "a-newer-prompt")

    assert not any(n.judged for n in _graded(entry, stored))


def test_a_record_from_another_fact_table_is_refused(entry, stored):
    import pytest

    records = stored.records(INTERPRETATIONS)
    records[0] = {**records[0], "key": {**records[0]["key"], "facts_sha256": "0" * 64}}

    with pytest.raises(ValueError, match="t-d"):
        grade_records([entry], records, stored.records(VERDICTS), "claude-opus-5-5")
