"""The interpretation job, run directly against the test database, with no network.

`cp_worker.jobs.build_model` is replaced by a scripted model, so every
outcome the job can store is reached without a real call.
"""

import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from cp_api.db import AnalysisSummary, Dataset, Interpretation
from cp_llm.facts import FactTable
from cp_llm.interpret import MEASURED_CHOICE, UNMEASURED_NOTE, VERIFIED_MEANS
from cp_llm.model import ModelCallError
from cp_llm.version import interpreter_version
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


def _queued(sessions, key, *, job_id=None, created_at=None, choice=MEASURED_CHOICE):
    """A queued interpretation of analysis `key`; its row id, the job's argument."""
    model, effort = choice
    with sessions() as session:
        row = Interpretation(
            analysis_id=key, status="queued", job_id=job_id or f"interpretation-{key}-1",
            model=model, effort=effort,
        )
        if created_at is not None:
            row.created_at = created_at
        session.add(row)
        session.commit()
        return row.id


def _facts(sessions, key):
    with sessions() as session:
        stored = session.get(AnalysisSummary, key).facts
    return stored, FactTable.from_json(json.dumps(stored))


def _row(sessions, row_id):
    with sessions() as session:
        return session.get(Interpretation, row_id)


def _columns(row):
    return tuple(getattr(row, column.name) for column in Interpretation.__table__.columns)


def _snapshot(sessions, analysis_id):
    """Every column of every interpretation of the analysis, ordered by id."""
    with sessions() as session:
        rows = session.scalars(
            select(Interpretation)
            .where(Interpretation.analysis_id == analysis_id)
            .order_by(Interpretation.id)
        ).all()
        return [_columns(row) for row in rows]


def _job_log(caplog):
    return [record for record in caplog.records if record.name == jobs.__name__]


def _script(monkeypatch, *items):
    """The scripted model every build returns; `model.built` lists each (model, effort) built."""
    model = ScriptedReplies(items)
    model.built = []

    def build(model_name, effort):
        model.built.append((model_name, effort))
        return model

    monkeypatch.setattr(jobs, "build_model", build)
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
    row_id = _queued(sessions, key)
    _script(monkeypatch, narrative_for(facts))

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
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
    assert row.interpreter_version == interpreter_version()


def test_a_narrative_that_never_verifies_is_refused_with_its_attempts(
    sessions, worker_env, monkeypatch
):
    key = _analysed(sessions)
    _, facts = _facts(sessions, key)
    row_id = _queued(sessions, key)
    bad = narrative_for(facts).model_copy(update={"summary": ["It rose by 2 points."]})
    model = _script(monkeypatch, bad, bad)

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert (row.status, model.calls, len(row.attempts)) == ("refused", 2, 2)
    assert row.attempts[0]["violations"]
    assert row.rendered is None and row.notice is None
    assert row.interpreter_version == interpreter_version()


def test_a_model_that_cannot_be_reached_fails_the_interpretation(
    sessions, worker_env, monkeypatch
):
    key = _analysed(sessions)
    row_id = _queued(sessions, key)
    _script(monkeypatch, ModelCallError("the model API call failed: APIConnectionError"))

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert row.status == "failed"
    assert "APIConnectionError" in row.error
    assert row.interpreter_version == interpreter_version()


def test_no_answer_after_a_rejected_reply_fails_and_keeps_that_reply(
    sessions, worker_env, monkeypatch
):
    key = _analysed(sessions)
    _, facts = _facts(sessions, key)
    row_id = _queued(sessions, key)
    bad = narrative_for(facts).model_copy(update={"summary": ["It rose by 2 points."]})
    _script(monkeypatch, bad, "max_tokens")

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert row.status == "failed"
    assert "max_tokens" in row.error
    assert len(row.attempts) == 1
    assert row.interpreter_version == interpreter_version()


def test_the_time_limit_fails_the_interpretation(sessions, worker_env, monkeypatch):
    key = _analysed(sessions)
    row_id = _queued(sessions, key)
    _script(monkeypatch, JobTimeLimitExceeded(1800))

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert row.status == "failed"
    assert row.error == "the interpretation exceeded the 1800-second job limit and was stopped"
    assert row.interpreter_version == interpreter_version()


def test_an_unexpected_error_is_stored_as_a_failure_not_raised(sessions, worker_env, monkeypatch):
    key = _analysed(sessions)
    row_id = _queued(sessions, key)
    _script(monkeypatch, RuntimeError("boom"))

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert (row.status, row.error) == ("failed", "boom")
    assert row.interpreter_version == interpreter_version()


def test_a_version_that_cannot_be_computed_fails_the_interpretation_with_no_version(
    sessions, worker_env, monkeypatch
):
    key = _analysed(sessions)
    row_id = _queued(sessions, key)
    model = _script(monkeypatch, narrative_for(_facts(sessions, key)[1]))

    def unknowable():
        raise RuntimeError("the rules could not be read")

    monkeypatch.setattr(jobs, "interpreter_version", unknowable)

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert (row.status, row.error) == ("failed", "the rules could not be read")
    assert row.interpreter_version is None
    assert row.finished_at is not None
    assert model.calls == 0


def test_an_analysis_without_facts_fails_its_interpretation(sessions, worker_env, monkeypatch):
    key = _analysed(sessions)
    with sessions() as session:
        session.get(AnalysisSummary, key).facts = None
        session.commit()
    row_id = _queued(sessions, key)
    model = _script(monkeypatch)

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert row.status == "failed"
    assert "no fact table" in row.error
    assert model.calls == 0
    assert row.interpreter_version == interpreter_version()


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
    row_id = _queued(sessions, key)
    with sessions() as session:
        row = session.get(Interpretation, row_id)
        row.status = "failed"
        row.error = "the queue was down"
        session.commit()
    before = _snapshot(sessions, key)
    model = _script(monkeypatch, narrative_for(_facts(sessions, key)[1]))

    run_interpretation_job(str(row_id))

    assert model.calls == 0
    assert _snapshot(sessions, key) == before


def test_a_job_that_is_not_the_one_the_row_names_makes_no_model_call(
    sessions, worker_env, redis_conn, monkeypatch
):
    from cp_worker.worker import make_worker
    from rq import Queue

    key = _analysed(sessions)
    row_id = _queued(sessions, key)
    model = _script(monkeypatch, narrative_for(_facts(sessions, key)[1]))
    Queue(jobs_queue_name(), connection=redis_conn).enqueue(
        "cp_worker.jobs.run_interpretation_job", str(row_id), job_id=f"interpretation-{key}-stale"
    )

    make_worker(redis_conn).work(burst=True)

    row = _row(sessions, row_id)
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
    row_id = _queued(sessions, key)
    model = _script(monkeypatch)
    monkeypatch.delenv("CP_INTERPRETATION")

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert (row.status, row.error) == ("failed", jobs.WORKER_INTERPRETATIONS_OFF)
    assert model.calls == 0
    assert row.finished_at is not None
    assert row.interpreter_version == interpreter_version()


def test_with_interpretations_off_a_withdrawn_choice_is_failed_as_off(
    sessions, worker_env, monkeypatch
):
    # Both would refuse the row; the off check comes first, so its sentence is the one stored.
    key = _analysed(sessions)
    row_id = _queued(sessions, key, choice=("claude-sonnet-5-5", "medium"))
    model = _script(monkeypatch)
    monkeypatch.delenv("CP_INTERPRETATION")

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert (row.status, row.error) == ("failed", jobs.WORKER_INTERPRETATIONS_OFF)
    assert (model.built, model.calls) == ([], 0)


def test_a_job_given_an_analysis_id_runs_its_newest_row_when_that_is_queued(
    sessions, worker_env, monkeypatch
):
    # A job enqueued before interpretations had their own ids carried the analysis id.
    key = _analysed(sessions)
    _, facts = _facts(sessions, key)
    earlier = datetime.now(UTC) - timedelta(hours=1)
    older_id = _queued(sessions, key, job_id=f"interpretation-{key}-0", created_at=earlier)
    with sessions() as session:
        older = session.get(Interpretation, older_id)
        older.status = "failed"
        older.error = "the queue was down"
        session.commit()
    newest_id = _queued(sessions, key)
    _script(monkeypatch, narrative_for(facts))

    run_interpretation_job(str(key))

    newest = _row(sessions, newest_id)
    assert newest.status == "verified"
    assert newest.interpreter_version == interpreter_version()
    older = _row(sessions, older_id)
    assert (older.status, older.error, older.started_at) == ("failed", "the queue was down", None)


def test_a_job_given_an_analysis_id_whose_newest_row_is_not_queued_writes_nothing(
    sessions, worker_env, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger=jobs.__name__)
    key = _analysed(sessions)
    row_id = _queued(sessions, key)
    with sessions() as session:
        row = session.get(Interpretation, row_id)
        row.status = "verified"
        session.commit()
    before = _snapshot(sessions, key)
    model = _script(monkeypatch, narrative_for(_facts(sessions, key)[1]))

    caplog.clear()
    run_interpretation_job(str(key))

    assert model.calls == 0
    assert _snapshot(sessions, key) == before
    logged = _job_log(caplog)
    assert len(logged) == 1 and str(key) in logged[0].getMessage()
    # Refused while resolving the argument, before the row's own check is reached.
    assert "names no interpretation" in logged[0].getMessage()


@pytest.mark.parametrize(
    "error", [RuntimeError("the database went away"), JobTimeLimitExceeded(1800)],
    ids=["unexpected", "time-limit"],
)
def test_a_job_that_fails_before_it_holds_its_row_raises(
    sessions, worker_env, monkeypatch, caplog, error
):
    # Raised, so RQ records the job failed and the API's lost-job check fails
    # the row on its next read, instead of once RQ forgets a finished job.
    caplog.set_level(logging.DEBUG, logger=jobs.__name__)
    key = _analysed(sessions)
    row_id = _queued(sessions, key)
    before = _snapshot(sessions, key)
    model = _script(monkeypatch, narrative_for(_facts(sessions, key)[1]))

    def unreachable(session, job_key):
        raise error

    monkeypatch.setattr(jobs, "_resolve", unreachable)

    with pytest.raises(type(error)):
        run_interpretation_job(str(row_id))

    assert model.calls == 0
    assert _snapshot(sessions, key) == before
    logged = _job_log(caplog)
    assert len(logged) == 1 and logged[0].exc_info is not None


def test_a_job_given_an_id_that_names_nothing_writes_nothing(
    sessions, worker_env, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger=jobs.__name__)
    key = _analysed(sessions)
    _queued(sessions, key)
    before = _snapshot(sessions, key)
    model = _script(monkeypatch, narrative_for(_facts(sessions, key)[1]))
    nothing = uuid.uuid4()

    caplog.clear()
    run_interpretation_job(str(nothing))

    assert model.calls == 0
    assert _snapshot(sessions, key) == before
    logged = _job_log(caplog)
    assert len(logged) == 1 and str(nothing) in logged[0].getMessage()
    # Skipped with a warning, not ended by an exception and its traceback.
    assert "names no interpretation" in logged[0].getMessage()
    assert (logged[0].levelno, logged[0].exc_info) == (logging.WARNING, None)


def test_with_two_rows_the_job_runs_exactly_the_one_it_was_given(
    sessions, worker_env, monkeypatch
):
    key = _analysed(sessions)
    _, facts = _facts(sessions, key)
    earlier = datetime.now(UTC) - timedelta(hours=1)
    # The older row is the one given, so the analysis's newest row would be the wrong one.
    given = _queued(sessions, key, job_id=f"interpretation-{key}-0", created_at=earlier)
    other = _queued(sessions, key)
    other_before = _columns(_row(sessions, other))
    model = _script(monkeypatch, narrative_for(facts))

    run_interpretation_job(str(given))

    row = _row(sessions, given)
    assert (row.status, row.job_id, model.calls) == ("verified", f"interpretation-{key}-0", 1)
    assert _columns(_row(sessions, other)) == other_before


# --- the operator's list ---------------------------------------------------

UNMEASURED = [("claude-sonnet-5-5", "medium"), ("claude-opus-5-5", "high")]


@pytest.mark.parametrize("choice", UNMEASURED, ids=["other-model", "other-effort"])
def test_a_choice_the_worker_does_not_offer_fails_without_a_model(
    sessions, worker_env, monkeypatch, choice
):
    # The setting is unset, so the worker offers claude-opus-5-5 at medium only.
    key = _analysed(sessions)
    row_id = _queued(sessions, key, choice=choice)
    model = _script(monkeypatch, narrative_for(_facts(sessions, key)[1]))

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert (row.status, row.error) == (
        "failed", "This model and effort are no longer offered on this server."
    )
    assert (model.built, model.calls) == ([], 0)
    assert row.notice is None and row.rendered is None
    assert row.finished_at is not None
    assert row.interpreter_version == interpreter_version()


def test_a_choice_withdrawn_from_a_longer_list_fails(sessions, worker_env, monkeypatch):
    monkeypatch.setenv("CP_INTERPRETATION_OPTIONS", "claude-sonnet-5-5:low, claude-opus-5-5:medium")
    key = _analysed(sessions)
    row_id = _queued(sessions, key, choice=("claude-sonnet-5-5", "medium"))
    model = _script(monkeypatch, narrative_for(_facts(sessions, key)[1]))

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert (row.status, row.error) == (
        "failed", "This model and effort are no longer offered on this server."
    )
    assert (model.built, model.calls) == ([], 0)


@pytest.mark.parametrize("choice", UNMEASURED, ids=["other-model", "other-effort"])
def test_a_verified_interpretation_from_an_unmeasured_choice_says_so(
    sessions, worker_env, monkeypatch, choice
):
    monkeypatch.setenv(
        "CP_INTERPRETATION_OPTIONS", f"claude-opus-5-5:medium,{choice[0]}:{choice[1]}"
    )
    key = _analysed(sessions)
    _, facts = _facts(sessions, key)
    row_id = _queued(sessions, key, choice=choice)
    model = _script(monkeypatch, narrative_for(facts))

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert row.status == "verified"
    assert model.built == [choice]
    assert row.notice == VERIFIED_MEANS + " " + UNMEASURED_NOTE


def test_a_verified_interpretation_from_the_measured_choice_has_no_note(
    sessions, worker_env, monkeypatch
):
    # The measured choice need not be the default.
    monkeypatch.setenv("CP_INTERPRETATION_OPTIONS", "claude-sonnet-5-5:low,claude-opus-5-5:medium")
    key = _analysed(sessions)
    _, facts = _facts(sessions, key)
    row_id = _queued(sessions, key)
    model = _script(monkeypatch, narrative_for(facts))

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert row.status == "verified"
    assert model.built == [MEASURED_CHOICE]
    assert row.notice == VERIFIED_MEANS


@pytest.mark.parametrize("outcome", ["refused", "failed"])
def test_an_unmeasured_choice_that_is_not_verified_stores_no_notice(
    sessions, worker_env, monkeypatch, outcome
):
    monkeypatch.setenv("CP_INTERPRETATION_OPTIONS", "claude-opus-5-5:medium,claude-sonnet-5-5:low")
    key = _analysed(sessions)
    _, facts = _facts(sessions, key)
    row_id = _queued(sessions, key, choice=("claude-sonnet-5-5", "low"))
    bad = narrative_for(facts).model_copy(update={"summary": ["It rose by 2 points."]})
    if outcome == "refused":
        _script(monkeypatch, bad, bad)
    else:
        _script(monkeypatch, ModelCallError("the model API call failed: APIConnectionError"))

    run_interpretation_job(str(row_id))

    row = _row(sessions, row_id)
    assert row.status == outcome
    assert row.notice is None


@pytest.mark.parametrize("outcome", ["verified", "refused", "failed", "unexpected"])
def test_a_row_the_api_marked_lost_during_the_call_keeps_what_the_api_wrote(
    sessions, worker_env, monkeypatch, caplog, outcome
):
    # The API's lost-job check can mark a running row failed while its model
    # call is still under way (RQ reports the job gone). The worker's outcome
    # must then not overwrite it: a request accepted after the failure would
    # otherwise leave two final interpretations from one entry, and a row the
    # user was shown as failed would change under them.
    from cp_api.app import WORKER_LOST
    from sqlalchemy import update

    key = _analysed(sessions)
    _, facts = _facts(sessions, key)
    row_id = _queued(sessions, key)
    bad = narrative_for(facts).model_copy(update={"summary": ["It rose by 2 points."]})
    replies = {
        "verified": [narrative_for(facts)],
        "refused": [bad, bad],
        "failed": [ModelCallError("the model API call failed: APIConnectionError")],
        "unexpected": [RuntimeError("boom")],
    }[outcome]
    model = _script(monkeypatch, *replies)
    build = jobs.build_model
    marked = []

    def build_then_mark_lost(model_name, effort):
        # Built after the worker committed `running`, before the call: the
        # API's write, as `_fail_lost_interpretations` makes it, lands here.
        with sessions() as session:
            session.execute(
                update(Interpretation)
                .where(
                    Interpretation.id == row_id,
                    Interpretation.status.in_(["queued", "running"]),
                )
                .values(status="failed", error=WORKER_LOST, finished_at=datetime.now(UTC))
            )
            session.commit()
        marked.append(_snapshot(sessions, key))
        return build(model_name, effort)

    monkeypatch.setattr(jobs, "build_model", build_then_mark_lost)
    caplog.set_level(logging.WARNING, logger=jobs.__name__)

    run_interpretation_job(str(row_id))

    assert model.calls == len(replies)
    assert _row(sessions, row_id).status == "failed"
    assert _snapshot(sessions, key) == marked[0]
    if outcome != "unexpected":
        # What was paid for and dropped is recorded nowhere else: the row keeps the API's write.
        spent = {
            "verified": "1 reply, 100 input and 40 output tokens",
            "refused": "2 replies, 200 input and 80 output tokens",
            "failed": "0 replies, 0 input and 0 output tokens",
        }[outcome]
        dropped = [r.getMessage() for r in _job_log(caplog) if str(row_id) in r.getMessage()]
        assert len(dropped) == 1
        assert spent in dropped[0]
        assert "claude-opus-5-5 at medium effort" in dropped[0]
