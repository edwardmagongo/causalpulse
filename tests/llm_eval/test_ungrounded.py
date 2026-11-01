"""The arm with no verifier: one call, kept as the model wrote it.

The record this path writes is the one `graded.grade_records` reads, and
`llm_eval.fakes.ungrounded_record` is that shape written by hand. The last
test here holds the two together, so the fixture the grading tests are built
on cannot drift away from what the run path actually stores.
"""

from __future__ import annotations

import cp_llm.interpret
import cp_llm.render
import pytest
from cp_llm.interpret import ModelDidNotAnswerError, interpret
from cp_llm.prompt import prompt_version
from cp_llm.verify import verify
from cp_llm_eval.arms import UNGROUNDED
from cp_llm_eval.attribution import D0
from cp_llm_eval.graded import grade_records
from cp_llm_eval.runner import INTERPRETATIONS, CallBudget, max_calls, plan_jobs, run
from cp_llm_eval.store import Store
from cp_llm_eval.ungrounded import UNVERIFIED, ungrounded_prompt, write_unverified
from cp_llm_eval.variants import BASELINE
from llm.builders import good_narrative

from llm_eval.fakes import ReplyModel, ungrounded_narrative, ungrounded_record

MODEL, EFFORT = "claude-opus-5-5", "medium"


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store")


def _run(entry, store, script, budget=5):
    """One ungrounded job of `entry`, answered by `script`."""
    jobs = plan_jobs([entry], [UNGROUNDED], 1, MODEL, EFFORT)
    return run(jobs, lambda: ReplyModel(script), store, CallBudget(budget), workers=1)


def _categories(narrative):
    return [[flag.category for flag in s.deterministic] for s in narrative.statements]


# ------------------------------------------------- the call


def test_the_arm_makes_one_call_with_the_control_prompt(facts):
    narrative = ungrounded_narrative(facts)
    model = ReplyModel([narrative])

    result = write_unverified(facts, model)

    assert model.calls == 1
    assert model.systems == [ungrounded_prompt()]
    assert result.narrative == narrative


def test_a_reply_the_verifier_would_reject_is_kept_as_it_stands(facts):
    """What the arm is for. On a grounded arm this reply is a refusal."""
    narrative = ungrounded_narrative(facts)
    assert verify(narrative, facts), "this prose is only interesting while it breaks a rule"

    result = write_unverified(facts, ReplyModel([narrative]))

    assert result.narrative == narrative


def test_the_narrative_is_its_own_rendered_form(facts):
    narrative = ungrounded_narrative(facts)

    stored = write_unverified(facts, ReplyModel([narrative])).to_dict()

    # Nothing was substituted, because nothing was a placeholder.
    assert "rendered" not in stored
    assert stored["narrative"] == narrative.model_dump()
    assert len(stored["attempts"]) == 1


def test_nothing_is_ever_substituted(facts, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("an ungrounded narrative reached substitute()")

    monkeypatch.setattr(cp_llm.render, "substitute", never)
    monkeypatch.setattr(cp_llm.interpret, "substitute", never)

    write_unverified(facts, ReplyModel([ungrounded_narrative(facts)]))

    # And the guard is not vacuous: a grounded narrative does reach it.
    with pytest.raises(AssertionError, match="reached substitute"):
        interpret(facts, ReplyModel([good_narrative()]), system=BASELINE.system())


def test_a_reply_with_no_narrative_is_no_answer(facts):
    with pytest.raises(ModelDidNotAnswerError, match="refusal"):
        write_unverified(facts, ReplyModel(["refusal"]))


def test_the_version_recorded_is_the_control_prompts(facts):
    result = write_unverified(facts, ReplyModel([ungrounded_narrative(facts)]))

    assert result.prompt_version == prompt_version(UNGROUNDED.system())
    assert result.prompt_version != prompt_version(BASELINE.system())


# ------------------------------------------------- through the runner


def test_an_ungrounded_job_stores_one_unverified_record(entry, store):
    summary = _run(entry, store, [ungrounded_narrative(entry.facts)])

    assert (summary.stored, summary.calls, summary.transport_failures) == (1, 1, 0)
    (record,) = store.records(INTERPRETATIONS)
    assert record["outcome"] == UNVERIFIED
    assert record["key"]["variant"] == "ungrounded"
    assert record["key"]["prompt_version"] == prompt_version(UNGROUNDED.system())
    assert len(record["attempts"]) == 1
    assert "rendered" not in record["interpretation"]


def test_an_ungrounded_narrative_costs_one_call_where_a_grounded_one_holds_two(entry, store):
    assert (max_calls(UNGROUNDED), max_calls(BASELINE)) == (1, 2)

    summary = _run(entry, store, [ungrounded_narrative(entry.facts)], budget=1)

    # One call is the whole cost, so a budget of one is enough to run it,
    assert (summary.stored, summary.not_started) == (1, 0)
    # where a grounded job must hold two before it may start.
    jobs = plan_jobs([entry], [BASELINE], 1, MODEL, EFFORT)
    grounded = run(jobs, lambda: ReplyModel([good_narrative()]), store, CallBudget(1), workers=1)
    assert (grounded.stored, grounded.not_started) == (0, 1)


def test_a_transport_failure_stores_nothing_and_a_rerun_retries_it(entry, store):
    first = _run(entry, store, [None])
    assert (first.stored, first.transport_failures) == (0, 1)

    second = _run(entry, store, [ungrounded_narrative(entry.facts)])
    assert (second.stored, second.skipped) == (1, 0)


def test_a_refusal_is_stored_with_no_narrative_to_grade(entry, store):
    _run(entry, store, ["refusal"])

    (record,) = store.records(INTERPRETATIONS)
    assert (record["outcome"], record["attempts"]) == ("no_answer", [])
    (graded,) = grade_records([entry], [record], [], MODEL)
    # P1 excludes it from its denominator: it stated no value at all.
    assert graded.statements == () and graded.first_attempt is None


def test_the_record_the_run_path_writes_grades_as_the_fixture_does(entry, store):
    """What holds `fakes.ungrounded_record` honest.

    The grading tests are built on that hand-written record; this one runs the
    real path over the same narrative and asserts the two grade alike, flag
    for flag, so the fixture cannot quietly describe a shape nothing writes.
    """
    narrative = ungrounded_narrative(entry.facts)
    _run(entry, store, [narrative])

    (real,) = grade_records([entry], store.records(INTERPRETATIONS), [], MODEL)
    (fixture,) = grade_records([entry], [ungrounded_record(entry, narrative)], [], MODEL)

    assert _categories(real) == _categories(fixture)
    assert D0 in [category for statement in _categories(real) for category in statement]
