from cp_llm_eval.attribution import D0, D0_ROUNDED
from cp_llm_eval.deterministic import grade_narrative
from cp_llm_eval.graded import UNVERIFIED, grade_records
from cp_llm_eval.judge import VERDICTS
from cp_llm_eval.runner import INTERPRETATIONS
from cp_llm_eval.runner import interpretation_key as _key

from llm_eval.fakes import (
    D4_STATEMENT,
    FABRICATED,
    ungrounded_narrative,
    ungrounded_record,
)


def _graded(entry, stored, judge_model="claude-opus-5-5"):
    return grade_records(
        [entry], stored.records(INTERPRETATIONS), stored.records(VERDICTS), judge_model
    )


def _categories(statement):
    return [flag.category for flag in statement.deterministic]


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

    (flagged,) = [s for s in narrative.statements if "D4" in _categories(s)]
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


# ------------------------------------------------- the ungrounded arm


def _ungrounded(entry, stored, narrative=None):
    narrative = narrative if narrative is not None else ungrounded_narrative(entry.facts)
    (graded,) = grade_records(
        [entry],
        [ungrounded_record(entry, narrative)],
        stored.records(VERDICTS),
        "claude-opus-5-5",
    )
    return graded


def test_an_ungrounded_narrative_is_graded_and_is_its_own_rendered_form(entry, stored):
    graded = _ungrounded(entry, stored)

    assert (graded.variant, graded.outcome) == ("ungrounded", UNVERIFIED)
    assert len(graded.statements) == 3
    # Nothing was substituted, because nothing was a placeholder.
    assert all(s.rendered == s.raw for s in graded.statements)
    assert all("{" not in s.raw for s in graded.statements)
    # No judge has seen this arm yet, so no statement carries a verdict.
    assert all(s.judge is None for s in graded.statements) and not graded.judged


def test_d1_runs_on_the_reconstructed_form_of_an_ungrounded_narrative(entry, stored):
    graded = _ungrounded(entry, stored)

    summary = graded.statements[0]
    assert "D1" in _categories(summary)
    (d1,) = [f for f in summary.deterministic if f.category == "D1"]
    assert "est.ipw.point" in d1.reason and "est.difference_in_means.point" in d1.reason


def test_without_the_reconstruction_the_same_narrative_scores_a_perfect_zero(entry):
    """The trap the wiring exists to avoid: no placeholders, so no input.

    Graded as a verified narrative would be, this prose yields nothing at all
    -- which would read as "no misreadings" in the report when what happened
    is that every grader found nothing to read.
    """
    narrative = ungrounded_narrative(entry.facts)

    unreconstructed = grade_narrative(narrative, entry.facts)

    assert [grade.flags for grade in unreconstructed] == [(), (), ()]


def test_a_fabricated_value_in_an_ungrounded_narrative_is_flagged_d0(entry, stored):
    graded = _ungrounded(entry, stored)

    estimates = graded.statements[1]
    (d0,) = [f for f in estimates.deterministic if f.category == D0]
    assert FABRICATED in d0.reason
    # The statement with the table's own values in it is not charged with one.
    assert D0 not in _categories(graded.statements[0])


def test_an_ungrounded_first_attempt_is_graded_the_same_way(entry, stored):
    graded = _ungrounded(entry, stored)

    assert graded.first_attempt is not None
    flagged = [f.category for flags in graded.first_attempt for f in flags]
    assert D0 in flagged and "D1" in flagged


def test_no_grounded_statement_carries_an_unsupported_value(entry, stored):
    """P0's property, through the path the study will run: the matcher on real
    rendered narratives finds nothing it could call invented.

    Every value in a verified narrative came from `render.substitute`, so it
    is some fact's rendered form exactly. A D0 or a D0-rounded here would be
    a defect in the matcher rather than a finding about the model.
    """
    graded = _graded(entry, stored)
    statements = [s for n in graded for s in n.statements]

    assert statements
    assert not [f for s in statements for f in s.deterministic if f.category in (D0, D0_ROUNDED)]


def test_the_matcher_reads_the_rendered_text_of_a_grounded_narrative(entry, stored):
    """The other half of P0: it can only pass meaningfully if the matcher ran.

    A grounded statement's values are behind placeholders, so the matcher is
    given the rendered text instead. Planting a value in that text that no
    fact supports must reach the statement as a D0 -- otherwise P0 would be
    measuring nothing and would pass on a matcher that never ran.
    """
    import copy

    records = [copy.deepcopy(r) for r in stored.records(INTERPRETATIONS)]
    (record,) = [r for r in records if r["outcome"] == "verified"][:1]
    record["interpretation"]["rendered"]["summary"][0] = (
        f"The effect was {FABRICATED} on the outcome scale."
    )

    graded = grade_records([entry], records, stored.records(VERDICTS), "claude-opus-5-5")
    (planted,) = [n for n in graded if n.interpretation == _key(record)]

    (d0,) = [f for f in planted.statements[0].deterministic if f.category == D0]
    assert FABRICATED in d0.reason
    # Only the statement whose rendered text was changed.
    assert not [f for s in planted.statements[1:] for f in s.deterministic if f.category == D0]
