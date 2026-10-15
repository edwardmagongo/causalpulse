"""The interpretation job, run directly against the test database, with no network.

`cp_worker.jobs.build_model` is replaced by a scripted model, so every
outcome the job can store is reached without a real call.
"""

import hashlib
import json
import uuid

import pytest
from cp_api.db import AnalysisSummary, Dataset, Interpretation
from cp_llm.facts import FactTable
from cp_llm.interpret import VERIFIED_MEANS
from cp_llm.model import ModelCallError
from cp_worker import jobs
from cp_worker.jobs import facts_digest, run_analysis_job, run_interpretation_job
from cp_worker.worker import JobTimeLimitExceeded
from sqlalchemy import select
from worker.data import scenario_d_upload
from worker.narratives import ScriptedReplies, narrative_for

pytestmark = pytest.mark.services
PAIR = ["difference_in_means", "lin_regression"]


@pytest.fixture(autouse=True)
def interpreting(monkeypatch):
    """The worker writes interpretations only with `CP_INTERPRETATION=on`."""
    monkeypatch.setenv("CP_INTERPRETATION", "on")


def _analysed(sessions):
    """A succeeded analysis of scenario D by two methods; its id."""
    from cp_api.accounts import create_user
    from cp_api.db import Analysis

    content, schema = scenario_d_upload()
    with sessions() as session:
        owner = create_user(session, f"owner-{uuid.uuid4().hex[:8]}")
        dataset = Dataset(
            owner_id=owner.id, filename="d.csv", sha256=hashlib.sha256(content).hexdigest(),
            n_bytes=len(content), csv=content,
        )
        session.add(dataset)
        session.flush()
        analysis = Analysis(
            owner_id=owner.id, dataset_id=dataset.id, causal_schema=schema,
            requested_estimators=PAIR, random_seed=0, status="queued",
        )
        session.add(analysis)
        session.commit()
        key = analysis.id
    run_analysis_job(str(key))
    return key


def _interpretation_of(session, analysis_id):
    """The analysis's one interpretation row, or None: at most one exists in these tests."""
    return session.scalars(
        select(Interpretation).where(Interpretation.analysis_id == analysis_id)
    ).one_or_none()


def _queued(sessions, key):
    with sessions() as session:
        session.add(
            Interpretation(
                analysis_id=key, status="queued", job_id=f"interpretation-{key}-1",
                model="claude-opus-5-5", effort="medium",
            )
        )
        session.commit()


def _facts(sessions, key):
    with sessions() as session:
        stored = session.get(AnalysisSummary, key).facts
    return stored, FactTable.from_json(json.dumps(stored))


def _row(sessions, key):
    with sessions() as session:
        return _interpretation_of(session, key)


def _script(monkeypatch, *items):
    model = ScriptedReplies(items)
    monkeypatch.setattr(jobs, "build_model", lambda model_name, effort: model)
    return model


def test_a_succeeded_analysis_stores_its_fact_table(sessions, worker_env):
    key = _analysed(sessions)

    stored, facts = _facts(sessions, key)

    assert stored["schema_version"] == 1
    assert facts.get("est.difference_in_means.point") is not None
    assert facts.get("est.lin_regression.point") is not None


def test_a_fact_table_the_extractor_refuses_leaves_the_analysis_succeeded(
    sessions, worker_env, monkeypatch
):
    def refuse(outcome):
        raise ValueError("a name the table cannot hold")

    monkeypatch.setattr(jobs, "fact_table", refuse)

    key = _analysed(sessions)

    from cp_api.db import Analysis

    with sessions() as session:
        assert session.get(Analysis, key).status == "succeeded"
        assert session.get(AnalysisSummary, key).facts is None


def test_any_error_building_the_fact_table_leaves_the_analysis_succeeded(
    sessions, worker_env, monkeypatch
):
    def fail(outcome):
        raise KeyError("unexpected")

    monkeypatch.setattr(jobs, "fact_table", fail)

    key = _analysed(sessions)

    from cp_api.db import Analysis

    with sessions() as session:
        assert session.get(Analysis, key).status == "succeeded"
        assert session.get(AnalysisSummary, key).facts is None


def test_a_verified_interpretation_is_stored_with_its_sections_and_notice(
    sessions, worker_env, monkeypatch
):
    key = _analysed(sessions)
    stored, facts = _facts(sessions, key)
    _queued(sessions, key)
    _script(monkeypatch, narrative_for(facts))

    run_interpretation_job(str(key))

    row = _row(sessions, key)
    assert row.status == "verified"
    assert row.returned_model == "scripted-model"
    assert row.notice == VERIFIED_MEANS
    names = [section["name"] for section in row.rendered["sections"]]
    assert names == ["summary", "estimates", "validity", "agreement", "caveats"]
    assert row.rendered["sections"][1]["heading"] == "Estimates"
    assert row.rendered["limits"]
    statements = [s for section in row.rendered["sections"] for s in section["statements"]]
    assert statements and all("{" not in s for s in statements)
    assert len(row.attempts) == 1
    assert row.facts_sha256 == facts_digest(stored)
    assert row.started_at is not None and row.finished_at is not None
    assert row.prompt_version


def test_a_narrative_that_never_verifies_is_refused_with_its_attempts(
    sessions, worker_env, monkeypatch
):
    key = _analysed(sessions)
    _, facts = _facts(sessions, key)
    _queued(sessions, key)
    bad = narrative_for(facts).model_copy(update={"summary": ["It rose by 2 points."]})
    model = _script(monkeypatch, bad, bad)

    run_interpretation_job(str(key))

    row = _row(sessions, key)
    assert (row.status, model.calls, len(row.attempts)) == ("refused", 2, 2)
    assert row.attempts[0]["violations"]
    assert row.rendered is None and row.notice is None


def test_a_model_that_cannot_be_reached_fails_the_interpretation(
    sessions, worker_env, monkeypatch
):
    key = _analysed(sessions)
    _queued(sessions, key)
    _script(monkeypatch, ModelCallError("the model API call failed: APIConnectionError"))

    run_interpretation_job(str(key))

    row = _row(sessions, key)
    assert row.status == "failed"
    assert "APIConnectionError" in row.error


def test_no_answer_after_a_rejected_reply_fails_and_keeps_that_reply(
    sessions, worker_env, monkeypatch
):
    key = _analysed(sessions)
    _, facts = _facts(sessions, key)
    _queued(sessions, key)
    bad = narrative_for(facts).model_copy(update={"summary": ["It rose by 2 points."]})
    _script(monkeypatch, bad, "max_tokens")

    run_interpretation_job(str(key))

    row = _row(sessions, key)
    assert row.status == "failed"
    assert "max_tokens" in row.error
    assert len(row.attempts) == 1


def test_the_time_limit_fails_the_interpretation(sessions, worker_env, monkeypatch):
    key = _analysed(sessions)
    _queued(sessions, key)
    _script(monkeypatch, JobTimeLimitExceeded(1800))

    run_interpretation_job(str(key))

    row = _row(sessions, key)
    assert row.status == "failed"
    assert row.error == "the interpretation exceeded the 1800-second job limit and was stopped"


def test_an_unexpected_error_is_stored_as_a_failure_not_raised(sessions, worker_env, monkeypatch):
    key = _analysed(sessions)
    _queued(sessions, key)
    _script(monkeypatch, RuntimeError("boom"))

    run_interpretation_job(str(key))

    row = _row(sessions, key)
    assert (row.status, row.error) == ("failed", "boom")


def test_an_analysis_without_facts_fails_its_interpretation(sessions, worker_env, monkeypatch):
    key = _analysed(sessions)
    with sessions() as session:
        session.get(AnalysisSummary, key).facts = None
        session.commit()
    _queued(sessions, key)
    model = _script(monkeypatch)

    run_interpretation_job(str(key))

    row = _row(sessions, key)
    assert row.status == "failed"
    assert "no fact table" in row.error
    assert model.calls == 0


def test_a_refused_fact_table_is_stored_as_sql_null_not_json_null(
    sessions, worker_env, monkeypatch
):
    from sqlalchemy import select

    def refuse(outcome):
        raise ValueError("a name the table cannot hold")

    monkeypatch.setattr(jobs, "fact_table", refuse)
    key = _analysed(sessions)

    with sessions() as session:
        missing = session.scalar(
            select(AnalysisSummary.facts.is_(None)).where(AnalysisSummary.analysis_id == key)
        )
    assert missing is True


def test_a_job_on_a_row_that_is_not_queued_makes_no_model_call(
    sessions, worker_env, monkeypatch
):
    key = _analysed(sessions)
    _queued(sessions, key)
    with sessions() as session:
        row = _interpretation_of(session, key)
        row.status = "failed"
        row.error = "the queue was down"
        session.commit()
    before = _row(sessions, key)
    before = (before.status, before.error, before.job_id, before.started_at, before.finished_at)
    model = _script(monkeypatch, narrative_for(_facts(sessions, key)[1]))

    run_interpretation_job(str(key))

    row = _row(sessions, key)
    assert model.calls == 0
    assert (row.status, row.error, row.job_id, row.started_at, row.finished_at) == before


def test_a_job_that_is_not_the_one_the_row_names_makes_no_model_call(
    sessions, worker_env, redis_conn, monkeypatch
):
    from cp_worker.worker import make_worker
    from rq import Queue

    key = _analysed(sessions)
    _queued(sessions, key)
    model = _script(monkeypatch, narrative_for(_facts(sessions, key)[1]))
    Queue(jobs_queue_name(), connection=redis_conn).enqueue(
        "cp_worker.jobs.run_interpretation_job", str(key), job_id=f"interpretation-{key}-stale"
    )

    make_worker(redis_conn).work(burst=True)

    row = _row(sessions, key)
    assert model.calls == 0
    assert (row.status, row.job_id) == ("queued", f"interpretation-{key}-1")
    assert row.started_at is None and row.finished_at is None


def jobs_queue_name() -> str:
    from cp_api.app import QUEUE_NAME

    return QUEUE_NAME


def test_a_worker_with_interpretations_off_fails_the_interpretation_without_a_call(
    sessions, worker_env, monkeypatch
):
    key = _analysed(sessions)
    _queued(sessions, key)
    model = _script(monkeypatch)
    monkeypatch.delenv("CP_INTERPRETATION")

    run_interpretation_job(str(key))

    row = _row(sessions, key)
    assert (row.status, row.error) == ("failed", jobs.WORKER_INTERPRETATIONS_OFF)
    assert model.calls == 0
    assert row.finished_at is not None
