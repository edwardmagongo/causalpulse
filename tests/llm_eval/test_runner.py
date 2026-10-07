import pytest
from cp_llm.prompt import prompt_version
from cp_llm_eval.runner import (
    INTERPRETATIONS,
    Call,
    CallBudget,
    cost,
    plan_jobs,
    run,
)
from cp_llm_eval.store import Store
from cp_llm_eval.variants import BASELINE, CITE_EACH_STATUS
from llm.builders import good_narrative, swap

from llm_eval.fakes import ReplyModel

GOOD = good_narrative()
BARE = swap(GOOD, "validity", 0, GOOD.validity[0] + " That is 2 steps down.")


def _jobs(entry, replicates=1, variants=(BASELINE,)):
    return plan_jobs([entry], variants, replicates, "claude-opus-5-5", "medium")


def test_a_verified_interpretation_is_stored_with_its_key_and_calls(entry, tmp_path):
    store = Store(tmp_path)

    summary = run(_jobs(entry), lambda: ReplyModel([GOOD]), store, CallBudget(10), workers=1)

    (record,) = store.records(INTERPRETATIONS)
    assert record["outcome"] == "verified"
    assert record["key"]["case_id"] == "t-d"
    assert record["key"]["prompt_version"] == prompt_version()
    assert record["calls"] == [
        {
            "input_tokens": 1000,
            "output_tokens": 100,
            "stop_reason": "end_turn",
            "model": "scripted-model",
        }
    ]
    assert record["attempts"] == record["interpretation"]["attempts"]
    assert (summary.stored, summary.calls) == (1, 1)
    assert summary.cost == pytest.approx(1000 * 4e-6 + 100 * 20e-6)


def test_a_refusal_is_stored_with_both_attempts(entry, tmp_path):
    store = Store(tmp_path)

    run(_jobs(entry), lambda: ReplyModel([BARE]), store, CallBudget(10), workers=1)

    (record,) = store.records(INTERPRETATIONS)
    assert record["outcome"] == "refused"
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["violations"][0]["rule"] == "bare_value"
    assert "interpretation" not in record


def test_a_reply_with_no_narrative_is_stored_as_no_answer(entry, tmp_path):
    store = Store(tmp_path)

    run(_jobs(entry), lambda: ReplyModel(["refusal"]), store, CallBudget(10), workers=1)

    (record,) = store.records(INTERPRETATIONS)
    assert (record["outcome"], record["stop_reason"]) == ("no_answer", "refusal")
    assert record["calls"][0]["stop_reason"] == "refusal"


def test_a_no_answer_after_a_rejected_reply_keeps_the_first_attempt(entry, tmp_path):
    from cp_llm_eval.graded import grade_records

    store = Store(tmp_path)
    model = ReplyModel([BARE, "max_tokens"])

    run(_jobs(entry), lambda: model, store, CallBudget(10), workers=1)

    (record,) = store.records(INTERPRETATIONS)
    assert (record["outcome"], record["stop_reason"]) == ("no_answer", "max_tokens")
    assert len(record["attempts"]) == 1
    assert record["attempts"][0]["violations"][0]["rule"] == "bare_value"
    assert model.calls == 2
    (graded,) = grade_records([entry], store.records(INTERPRETATIONS), [], "claude-opus-5-5")
    assert graded.first_attempt is not None


def test_a_transport_failure_stores_nothing_and_still_counts(entry, tmp_path):
    store = Store(tmp_path)
    budget = CallBudget(10)

    summary = run(_jobs(entry), lambda: ReplyModel([None]), store, budget, workers=1)

    assert store.records(INTERPRETATIONS) == []
    assert (summary.transport_failures, summary.calls, budget.used) == (1, 1, 1)


def test_a_rerun_skips_what_is_stored_and_retries_what_failed(entry, tmp_path):
    store = Store(tmp_path)
    jobs = _jobs(entry, replicates=2)
    run(jobs[:1], lambda: ReplyModel([GOOD]), store, CallBudget(10), workers=1)
    model = ReplyModel([GOOD])

    summary = run(jobs, lambda: model, store, CallBudget(10), workers=1)

    assert (summary.skipped, summary.stored, model.calls) == (1, 1, 1)
    assert len(store.records(INTERPRETATIONS)) == 2


def test_the_budget_is_never_exceeded_by_an_interpretation_that_needs_a_retry(entry, tmp_path):
    store = Store(tmp_path)
    model = ReplyModel([BARE])
    budget = CallBudget(3)

    summary = run(_jobs(entry, replicates=3), lambda: model, store, budget, workers=1)

    # Each needs two calls: one fits in three, a second would need a fourth.
    assert (summary.stored, summary.not_started, model.calls, budget.used) == (1, 2, 2, 2)


def test_unused_reservations_are_returned(entry, tmp_path):
    store = Store(tmp_path)
    budget = CallBudget(4)
    model = ReplyModel([GOOD])

    summary = run(_jobs(entry, replicates=3), lambda: model, store, budget, workers=1)

    # Each reserves two and uses one. Released, the spare lets the third start
    # (2 used + 2 reserved = 4); kept, it would have needed 6.
    assert (summary.stored, budget.used) == (3, 3)


def test_concurrent_workers_store_every_job_once_within_budget(entry, tmp_path):
    store = Store(tmp_path)
    model = ReplyModel([GOOD])
    budget = CallBudget(100)

    jobs = _jobs(entry, replicates=8, variants=(BASELINE, CITE_EACH_STATUS))
    summary = run(jobs, lambda: model, store, budget, workers=4)

    assert summary.stored == 16
    assert len({r["key"]["replicate"] for r in store.records(INTERPRETATIONS)}) == 8
    assert budget.used == model.calls == 16


def test_each_variant_sends_its_own_prompt_and_is_keyed_apart(entry, tmp_path):
    store = Store(tmp_path)
    model = ReplyModel([GOOD])

    jobs = _jobs(entry, variants=(BASELINE, CITE_EACH_STATUS))
    run(jobs, lambda: model, store, CallBudget(10), workers=1)

    assert set(model.systems) == {BASELINE.system(), CITE_EACH_STATUS.system()}
    versions = {
        r["key"]["variant"]: r["key"]["prompt_version"] for r in store.records(INTERPRETATIONS)
    }
    assert versions == {
        "baseline": prompt_version(BASELINE.system()),
        "cite-each-status": prompt_version(CITE_EACH_STATUS.system()),
    }


def test_a_call_without_a_reservation_is_a_defect():
    with pytest.raises(RuntimeError, match="without a reservation"):
        CallBudget(5).spend()


def test_jobs_need_at_least_one_replicate(entry):
    with pytest.raises(ValueError, match="replicates"):
        _jobs(entry, replicates=0)


def test_cost_prices_input_and_output_tokens():
    assert cost([Call(1_000_000, 0, "end_turn", "m"), Call(0, 1_000_000, "end_turn", "m")]) == 24.0
