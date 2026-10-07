import json
from types import SimpleNamespace

import pytest
from cp_llm.model import ModelCallError
from cp_llm_eval.judge import (
    JUDGE_SYSTEM,
    VERDICTS,
    ClaudeJudge,
    Finding,
    JudgeJob,
    JudgeReply,
    StatementVerdict,
    judge_all,
    judge_message,
    judge_prompt_version,
    rendered_statements,
)
from cp_llm_eval.runner import INTERPRETATIONS, CallBudget, plan_jobs, run
from cp_llm_eval.store import Store
from cp_llm_eval.variants import BASELINE
from llm.builders import good_narrative

from llm_eval.fakes import ReplyJudge, ReplyModel


@pytest.fixture(scope="module")
def verified(entry, tmp_path_factory):
    """A verified interpretation of `entry`, stored by the runner, and its key."""
    store = Store(tmp_path_factory.mktemp("store"))
    jobs = plan_jobs([entry], [BASELINE], 1, "claude-opus-5-5", "medium")
    run(jobs, lambda: ReplyModel([good_narrative()]), store, CallBudget(5), workers=1)
    (record,) = store.records(INTERPRETATIONS)
    return jobs[0].key(), record


def _job(verified, entry):
    key, record = verified
    return JudgeJob(key, rendered_statements(record), entry.facts, "claude-opus-5-5")


def _clean(n):
    return JudgeReply(statements=[StatementVerdict(number=i, findings=[]) for i in range(1, n + 1)])


def test_the_judge_sees_rendered_values_and_numbered_statements(verified, entry):
    job = _job(verified, entry)

    message = judge_message(job.statements, entry.facts)

    assert "{" not in message.split("Statements:")[1]
    assert "1. [summary] " in message
    assert "- est.ipw.point (number) = " in message
    assert len(job.statements) == len(good_narrative().statements())


def test_a_verdict_is_stored_per_statement(verified, entry, tmp_path):
    store = Store(tmp_path)
    job = _job(verified, entry)

    def reply(n):
        verdicts = [StatementVerdict(number=i, findings=[]) for i in range(1, n + 1)]
        verdicts[1] = StatementVerdict(number=2, findings=[Finding(category="J1", reason="x")])
        return JudgeReply(statements=list(reversed(verdicts)))

    summary = judge_all([job], lambda: ReplyJudge(reply), store, CallBudget(5), workers=1)

    (record,) = store.records(VERDICTS)
    assert record["outcome"] == "judged"
    assert record["interpretation"] == job.interpretation
    assert record["judge_prompt_version"] == judge_prompt_version()
    assert record["statements"][0] == []
    assert record["statements"][1] == [{"category": "J1", "reason": "x"}]
    assert (summary.stored, summary.calls, summary.judge_errors) == (1, 1, 0)


@pytest.mark.parametrize(
    "reply_for",
    [
        lambda n: _clean(n - 1),
        lambda n: JudgeReply(
            statements=[*_clean(n).statements, StatementVerdict(number=n + 1, findings=[])]
        ),
        lambda n: None,
    ],
    ids=["a statement missing", "a statement that does not exist", "no verdict"],
)
def test_an_unusable_reply_is_stored_as_a_judge_error(reply_for, verified, entry, tmp_path):
    store = Store(tmp_path)

    job = _job(verified, entry)
    summary = judge_all([job], lambda: ReplyJudge(reply_for), store, CallBudget(5))

    (record,) = store.records(VERDICTS)
    assert record["outcome"] == "judge_error"
    assert "statements" not in record
    assert summary.judge_errors == 1


def test_a_transport_failure_stores_nothing_and_counts(verified, entry, tmp_path):
    store = Store(tmp_path)
    budget = CallBudget(5)

    def fail(n):
        raise ModelCallError("down")

    summary = judge_all([_job(verified, entry)], lambda: ReplyJudge(fail), store, budget)

    assert store.records(VERDICTS) == []
    assert (summary.transport_failures, budget.used) == (1, 1)


def test_a_judged_narrative_is_not_judged_again(verified, entry, tmp_path):
    store = Store(tmp_path)
    judge_all([_job(verified, entry)], lambda: ReplyJudge(_clean), store, CallBudget(5))
    judge = ReplyJudge(_clean)

    summary = judge_all([_job(verified, entry)], lambda: judge, store, CallBudget(5))

    assert (summary.skipped, judge.calls) == (1, 0)


def test_the_judge_stops_at_its_budget(verified, entry, tmp_path):
    job = _job(verified, entry)
    jobs = [JudgeJob(f"k{i}", job.statements, job.facts, job.model) for i in range(3)]

    summary = judge_all(jobs, lambda: ReplyJudge(_clean), Store(tmp_path), CallBudget(2))

    assert (summary.stored, summary.not_started) == (2, 1)


def test_an_unexpected_judge_error_stops_the_pool_and_is_raised(verified, entry, tmp_path):
    job = _job(verified, entry)
    jobs = [JudgeJob(f"k{i}", job.statements, job.facts, job.model) for i in range(10)]
    budget = CallBudget(100)

    def broken(n):
        raise RuntimeError("unexpected")

    judge = ReplyJudge(broken)
    with pytest.raises(RuntimeError, match="unexpected"):
        judge_all(jobs, lambda: judge, Store(tmp_path), budget, workers=1)

    assert (judge.calls, budget.used) == (1, 1)


def test_a_duplicated_judge_job_is_judged_once(verified, entry, tmp_path):
    job = _job(verified, entry)
    judge = ReplyJudge(_clean)

    summary = judge_all([job, job], lambda: judge, Store(tmp_path), CallBudget(5), workers=2)

    assert (judge.calls, summary.stored) == (1, 1)


def _response(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class _Client:
    def __init__(self, response=None, error=None):
        self.requests = []
        self._response, self._error = response, error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request):
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._response


def test_claude_judge_constrains_the_reply_to_the_schema_and_reads_it():
    client = _Client(_response(_clean(2).model_dump_json()))

    call = ClaudeJudge(client=client).judge(JUDGE_SYSTEM, "message")

    assert call.reply == _clean(2)
    (request,) = client.requests
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["system"] == JUDGE_SYSTEM
    assert "fallbacks" not in json.dumps(request, default=str)


def test_claude_judge_returns_no_reply_for_a_refusal_or_invalid_json():
    refused = ClaudeJudge(client=_Client(_response("I won't", "refusal"))).judge("s", "u")
    invalid = ClaudeJudge(client=_Client(_response("not json"))).judge("s", "u")

    assert (refused.reply, refused.stop_reason) == (None, "refusal")
    assert (invalid.reply, invalid.stop_reason) == (None, "invalid_reply")


def test_claude_judge_raises_an_sdk_error_as_a_model_call_error():
    import anthropic

    client = _Client(error=anthropic.AnthropicError("boom"))

    with pytest.raises(ModelCallError, match="boom"):
        ClaudeJudge(client=client).judge("s", "u")


def test_a_verdict_records_key_is_its_jobs_key(verified, entry, tmp_path):
    from cp_llm_eval.judge import verdict_key

    store = Store(tmp_path)
    job = _job(verified, entry)
    judge_all([job], lambda: ReplyJudge(_clean), store, CallBudget(5))

    (record,) = store.records(VERDICTS)
    assert verdict_key(record) == job.key()


def test_the_judge_prompt_does_not_mention_the_study_and_defines_every_category():
    assert "study" not in JUDGE_SYSTEM
    for number in range(1, 7):
        assert f"- J{number}:" in JUDGE_SYSTEM
