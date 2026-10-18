"""`POST /v1/analyses/{id}/interpretation` and the interpretation on the analysis.

Real PostgreSQL and Redis; the jobs run in this process with
`cp_worker.jobs.build_model` replaced, so no test reaches the network.
"""

import json
import threading
import time
import uuid
from datetime import timedelta

import pytest
from cp_api.app import (
    ALREADY_REFUSED,
    ALREADY_VERIFIED,
    BEING_WRITTEN,
    CURRENT_INTERPRETER_VERSION,
    INTERPRETATIONS_OFF,
    NAME_BOTH_OR_NEITHER,
    NO_FACTS,
    NOT_OFFERED,
    NOT_SUCCEEDED,
    SUBMISSION_GRACE,
    WORKER_LOST,
    create_app,
)
from cp_api.db import Analysis, AnalysisSummary, Interpretation, utcnow
from cp_api.limits import Limit, Limits
from cp_api.settings import InterpretationOption
from cp_llm.facts import FactTable
from cp_llm.model import ModelCallError
from cp_worker import jobs
from cp_worker.worker import make_worker
from fastapi.testclient import TestClient
from rq.job import Job
from sqlalchemy import func, select, text
from worker.data import scenario_d_upload
from worker.narratives import ScriptedReplies, narrative_for

pytestmark = pytest.mark.services
PAIR = ["difference_in_means", "lin_regression"]
# A version the interpreter has never had: an interpretation written under it is out of date.
OLDER = "0123456789ab"
# A server's list of two entries, the first the default, and an entry on neither.
A = InterpretationOption("claude-opus-5-5", "medium")
B = InterpretationOption("claude-sonnet-5-5", "low")
TWO = (A, B)


@pytest.fixture(autouse=True)
def interpreting(monkeypatch):
    """The worker writes interpretations only with `CP_INTERPRETATION=on`; each app here says
    for itself whether the API accepts them."""
    monkeypatch.setenv("CP_INTERPRETATION", "on")


@pytest.fixture
def make_client(sessions, redis_conn, worker_env, api_key_for, monkeypatch):
    def make(name="alice", interpretation=True, limits=None, options=None):
        # Without `options`, the app reads the setting, which no test sets: one
        # entry, `claude-opus-5-5:medium`. With them, the setting is set to the
        # same list, so the worker these tests drain offers what the app offers
        # (it fails a row whose entry its own list lacks).
        if options:
            monkeypatch.setenv(
                "CP_INTERPRETATION_OPTIONS", ",".join(f"{o.model}:{o.effort}" for o in options)
            )
        return TestClient(
            create_app(
                sessions=sessions, redis=redis_conn, limits=limits,
                interpretation=interpretation, interpretation_options=options,
            ),
            headers={"Authorization": f"Bearer {api_key_for(name)}"},
        )

    return make


def _rows(session, analysis_id):
    """The analysis's interpretation rows, newest first."""
    return list(
        session.scalars(
            select(Interpretation)
            .where(Interpretation.analysis_id == analysis_id)
            .order_by(Interpretation.created_at.desc(), Interpretation.id.desc())
        )
    )


def _newest(session, analysis_id):
    rows = _rows(session, analysis_id)
    return rows[0] if rows else None


def _columns(row):
    return {column.key: getattr(row, column.key) for column in Interpretation.__table__.columns}


# What the worker leaves on a row of each status, near enough for the API to serve it.
_FINISHED = {
    "verified": {
        "prompt_version": "p-planted",
        "rendered": {
            "sections": [{"name": "summary", "heading": "Summary", "statements": ["Planted."]}],
            "limits": "Planted limits.",
        },
        "notice": "Planted notice.",
        "attempts": [{"violations": []}],
    },
    "refused": {
        "prompt_version": "p-planted",
        "attempts": [{"violations": [{"rule": "bare_value"}]}] * 2,
    },
    "failed": {"error": "the model API call failed: planted"},
    "queued": {},
}


def _write(sessions, analysis_id, status, version, *, minutes_ago, entry=A):
    """Write one interpretation row from `entry` directly, in its own transaction, dated
    `minutes_ago` minutes back so that rows written by one test never tie on `created_at`;
    its id."""
    created = utcnow() - timedelta(minutes=minutes_ago)
    with sessions() as session:
        row = Interpretation(
            analysis_id=uuid.UUID(analysis_id),
            status=status,
            job_id=f"interpretation-{analysis_id}-{uuid.uuid4().hex[:8]}",
            model=entry.model,
            effort=entry.effort,
            interpreter_version=version,
            created_at=created,
            finished_at=None if status == "queued" else created + timedelta(seconds=30),
            **_FINISHED[status],
        )
        session.add(row)
        session.commit()
        return row.id


def _drain(redis_conn):
    make_worker(redis_conn).work(burst=True)


def _analysed(client, redis_conn):
    """A succeeded analysis of scenario D by two methods; its id."""
    content, schema = scenario_d_upload()
    dataset = client.post("/v1/datasets", files={"file": ("d.csv", content, "text/csv")}).json()
    body = {"dataset_id": dataset["id"], "schema": schema, "random_seed": 0, "estimators": PAIR}
    analysis_id = client.post("/v1/analyses", json=body).json()["id"]
    _drain(redis_conn)
    assert client.get(f"/v1/analyses/{analysis_id}").json()["status"] == "succeeded"
    return analysis_id


def _facts(sessions, analysis_id):
    with sessions() as session:
        stored = session.get(AnalysisSummary, analysis_id).facts
    return FactTable.from_json(json.dumps(stored))


def _script(monkeypatch, *items):
    model = ScriptedReplies(items)
    monkeypatch.setattr(jobs, "build_model", lambda model_name, effort: model)
    return model


def _ask(client, analysis_id, entry=None):
    """Ask for an interpretation: with no body, or naming `entry`."""
    if entry is None:
        return client.post(f"/v1/analyses/{analysis_id}/interpretation")
    return client.post(
        f"/v1/analyses/{analysis_id}/interpretation",
        json={"model": entry.model, "effort": entry.effort},
    )


def _undo(sessions, redis_conn, interpretation_id):
    """Delete an accepted request's row and its job, as if it had not been made."""
    with sessions() as session:
        row = session.get(Interpretation, uuid.UUID(interpretation_id))
        job_id = row.job_id
        session.delete(row)
        session.commit()
    Job.fetch(job_id, connection=redis_conn).delete()


def _offers_what_it_answers(client, sessions, redis_conn, analysis_id, options):
    """The analysis's `interpretation_options`, `interpretable` and `interpretable_reason`
    agree with what a request for each entry is answered; the analysis as read.

    Every entry is asked for in the state the analysis described: an accepted
    request is undone before the next is made, so the asking changes nothing.
    """
    detail = client.get(f"/v1/analyses/{analysis_id}").json()
    offered = detail["interpretation_options"]
    assert [(o["model"], o["effort"], o["default"]) for o in offered] == [
        (entry.model, entry.effort, index == 0) for index, entry in enumerate(options)
    ]
    for option in offered:
        asked = _ask(client, analysis_id, InterpretationOption(option["model"], option["effort"]))
        if option["available"]:
            assert option["reason"] is None, option
            assert asked.status_code == 202, (option, asked.json())
            assert (asked.json()["model"], asked.json()["effort"]) == (
                option["model"], option["effort"],
            )
            _undo(sessions, redis_conn, asked.json()["id"])
        else:
            assert (asked.status_code, asked.json()["detail"]) == (409, option["reason"]), option
    assert detail["interpretable"] is any(option["available"] for option in offered)
    assert detail["interpretable_reason"] == (
        None if detail["interpretable"] else offered[0]["reason"]
    )
    return detail


def _answers(detail):
    """Each entry's (available, reason), in the list's order."""
    return [(o["available"], o["reason"]) for o in detail["interpretation_options"]]


def _bad(facts):
    return narrative_for(facts).model_copy(update={"summary": ["It rose by 2 points."]})


def test_an_interpretation_is_queued_written_and_served(
    make_client, redis_conn, sessions, monkeypatch
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _script(monkeypatch, narrative_for(_facts(sessions, analysis_id)))

    asked = _ask(client, analysis_id)

    assert asked.status_code == 202
    assert asked.json()["status"] == "queued"
    assert client.get(f"/v1/analyses/{analysis_id}").json()["interpretation"]["status"] == "queued"
    _drain(redis_conn)
    detail = client.get(f"/v1/analyses/{analysis_id}").json()
    got = detail["interpretation"]
    assert got["status"] == "verified"
    assert [s["heading"] for s in got["sections"]][:2] == ["Summary", "Estimates"]
    assert got["limits"] and got["notice"].startswith("A verified interpretation means:")
    assert (got["model"], got["effort"]) == ("claude-opus-5-5", "medium")
    assert (got["attempts"], got["reason"], got["error"]) == (None, None, None)
    assert (got["interpreter_version"], got["out_of_date"]) == (CURRENT_INTERPRETER_VERSION, False)
    assert detail["earlier_interpretations"] == []
    assert (detail["interpretable"], detail["interpretable_reason"]) == (False, ALREADY_VERIFIED)


def test_a_server_with_interpretations_off_answers_403_and_offers_none(make_client, redis_conn):
    client = make_client(interpretation=False, options=TWO)
    analysis_id = _analysed(client, redis_conn)

    asked = _ask(client, analysis_id)
    # Before the body is judged: one naming half an entry is answered the same.
    half = client.post(f"/v1/analyses/{analysis_id}/interpretation", json={"model": A.model})

    assert (asked.status_code, asked.json()["detail"]) == (403, INTERPRETATIONS_OFF)
    assert (half.status_code, half.json()["detail"]) == (403, INTERPRETATIONS_OFF)
    detail = client.get(f"/v1/analyses/{analysis_id}").json()
    assert (detail["interpretable"], detail["interpretable_reason"]) == (False, None)
    assert detail["interpretation_options"] == []
    assert detail["interpretation"] is None


def test_someone_elses_analysis_is_404(make_client, redis_conn):
    analysis_id = _analysed(make_client("alice"), redis_conn)

    bob = make_client("bob", options=TWO)
    asked = _ask(bob, analysis_id)
    # Before the body is judged.
    unlisted = _ask(bob, analysis_id, InterpretationOption("claude-haiku-5-5", "low"))

    assert (asked.status_code, asked.json()["detail"]) == (404, f"no analysis {analysis_id}")
    assert (unlisted.status_code, unlisted.json()["detail"]) == (
        404, f"no analysis {analysis_id}",
    )


def test_an_analysis_that_has_not_finished_is_409(make_client):
    client = make_client()
    content, schema = scenario_d_upload()
    dataset = client.post("/v1/datasets", files={"file": ("d.csv", content, "text/csv")}).json()
    body = {"dataset_id": dataset["id"], "schema": schema, "random_seed": 0, "estimators": PAIR}
    analysis_id = client.post("/v1/analyses", json=body).json()["id"]

    asked = _ask(client, analysis_id)

    assert (asked.status_code, asked.json()["detail"]) == (409, NOT_SUCCEEDED)
    detail = client.get(f"/v1/analyses/{analysis_id}").json()
    assert (detail["interpretable"], detail["interpretable_reason"]) == (False, NOT_SUCCEEDED)


def test_an_analysis_without_a_fact_table_is_409(make_client, redis_conn, sessions):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    with sessions() as session:
        session.get(AnalysisSummary, analysis_id).facts = None
        session.commit()

    asked = _ask(client, analysis_id)

    assert (asked.status_code, asked.json()["detail"]) == (409, NO_FACTS)
    detail = client.get(f"/v1/analyses/{analysis_id}").json()
    assert (detail["interpretable"], detail["interpretable_reason"]) == (False, NO_FACTS)


def test_asking_while_one_is_being_written_is_409_and_the_analysis_says_so(
    make_client, redis_conn
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    assert _ask(client, analysis_id).status_code == 202
    detail = client.get(f"/v1/analyses/{analysis_id}").json()

    again = _ask(client, analysis_id)

    assert (again.status_code, again.json()["detail"]) == (409, BEING_WRITTEN)
    assert (detail["interpretable"], detail["interpretable_reason"]) == (False, BEING_WRITTEN)


def test_a_pending_one_is_409_even_over_an_out_of_date_one(make_client, redis_conn, sessions):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _write(sessions, analysis_id, "verified", OLDER, minutes_ago=10)
    assert _ask(client, analysis_id).status_code == 202

    again = _ask(client, analysis_id)

    assert (again.status_code, again.json()["detail"]) == (409, BEING_WRITTEN)
    with sessions() as session:
        assert [row.status for row in _rows(session, analysis_id)] == ["queued", "verified"]


def test_a_verified_interpretation_is_final_under_the_current_version(
    make_client, redis_conn, sessions, monkeypatch
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _script(monkeypatch, narrative_for(_facts(sessions, analysis_id)))
    _ask(client, analysis_id)
    _drain(redis_conn)
    detail = client.get(f"/v1/analyses/{analysis_id}").json()

    again = _ask(client, analysis_id)

    # The worker records the version it wrote under, and it is the API's current one.
    got = detail["interpretation"]
    assert (got["interpreter_version"], got["out_of_date"]) == (
        CURRENT_INTERPRETER_VERSION, False,
    )
    assert (again.status_code, again.json()["detail"]) == (409, ALREADY_VERIFIED)
    assert (detail["interpretable"], detail["interpretable_reason"]) == (False, ALREADY_VERIFIED)


def test_a_refused_interpretation_is_final_and_its_narrative_is_not_served(
    make_client, redis_conn, sessions, monkeypatch
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    bad = _bad(_facts(sessions, analysis_id))
    _script(monkeypatch, bad, bad)
    _ask(client, analysis_id)
    _drain(redis_conn)

    detail = client.get(f"/v1/analyses/{analysis_id}").json()
    got = detail["interpretation"]
    again = _ask(client, analysis_id)

    assert got["status"] == "refused"
    assert got["attempts"] == 2
    assert got["reason"] == (
        "The last attempt was refused because it wrote a number in its own words instead of "
        "citing it."
    )
    assert (got["sections"], got["limits"], got["notice"]) == (None, None, None)
    assert (got["interpreter_version"], got["out_of_date"]) == (
        CURRENT_INTERPRETER_VERSION, False,
    )
    assert (again.status_code, again.json()["detail"]) == (409, ALREADY_REFUSED)
    assert (detail["interpretable"], detail["interpretable_reason"]) == (False, ALREADY_REFUSED)


# Each kind of finished newest interpretation, under the current version, an
# older one and none: what a request is answered, or None for 202.
DECISIONS = [
    ("verified", CURRENT_INTERPRETER_VERSION, ALREADY_VERIFIED),
    ("verified", OLDER, None),
    ("verified", None, None),
    ("refused", CURRENT_INTERPRETER_VERSION, ALREADY_REFUSED),
    ("refused", OLDER, None),
    ("refused", None, None),
    ("failed", CURRENT_INTERPRETER_VERSION, None),
    ("failed", OLDER, None),
    ("failed", None, None),
]


@pytest.mark.parametrize(
    ("status", "version", "refusal"),
    DECISIONS,
    ids=[f"{status}-{'current' if version == CURRENT_INTERPRETER_VERSION else version}"
         for status, version, _ in DECISIONS],
)
def test_the_newest_interpretation_decides_whether_a_request_is_accepted(
    make_client, redis_conn, sessions, status, version, refusal
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    old_id = _write(sessions, analysis_id, status, version, minutes_ago=10)
    with sessions() as session:
        before = _columns(session.get(Interpretation, old_id))
    detail = client.get(f"/v1/analyses/{analysis_id}").json()

    asked = _ask(client, analysis_id)

    got = detail["interpretation"]
    assert (got["status"], got["interpreter_version"]) == (status, version)
    # Out of date only for a verdict: a failed one may be asked for again whatever its version.
    assert got["out_of_date"] is (
        status in ("verified", "refused") and version != CURRENT_INTERPRETER_VERSION
    )
    assert detail["earlier_interpretations"] == []
    # What the analysis offered is what the request was answered.
    assert (detail["interpretable"], detail["interpretable_reason"]) == (
        refusal is None, refusal,
    )
    with sessions() as session:
        rows = _rows(session, analysis_id)
        # The row that was there is untouched, whatever the answer.
        assert _columns(session.get(Interpretation, old_id)) == before
    if refusal is not None:
        assert (asked.status_code, asked.json()["detail"]) == (409, refusal)
        assert [row.id for row in rows] == [old_id]
        return
    assert asked.status_code == 202
    assert [row.id for row in rows][1:] == [old_id]
    new = rows[0]
    assert (new.status, new.interpreter_version, new.error) == ("queued", None, None)
    assert new.job_id != before["job_id"]
    # The job is given the new row's own id.
    assert list(Job.fetch(new.job_id, connection=redis_conn).args) == [str(new.id)]
    assert (asked.json()["status"], asked.json()["out_of_date"]) == ("queued", False)
    after = client.get(f"/v1/analyses/{analysis_id}").json()
    # A pending interpretation has no verdict yet, so it is not out of date.
    assert (after["interpretation"]["status"], after["interpretation"]["out_of_date"]) == (
        "queued", False,
    )
    assert [(e["status"], e["interpreter_version"]) for e in after["earlier_interpretations"]] == [
        (status, version),
    ]


def test_the_analysis_serves_the_newest_and_the_earlier_ones_newest_first(
    make_client, redis_conn, sessions
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _write(sessions, analysis_id, "failed", None, minutes_ago=30)
    _write(sessions, analysis_id, "verified", OLDER, minutes_ago=20)
    _write(sessions, analysis_id, "refused", OLDER, minutes_ago=10)
    _write(sessions, analysis_id, "verified", CURRENT_INTERPRETER_VERSION, minutes_ago=5)

    detail = client.get(f"/v1/analyses/{analysis_id}").json()

    got = detail["interpretation"]
    assert (got["status"], got["interpreter_version"], got["out_of_date"]) == (
        "verified", CURRENT_INTERPRETER_VERSION, False,
    )
    earlier = detail["earlier_interpretations"]
    assert [(e["status"], e["interpreter_version"], e["out_of_date"]) for e in earlier] == [
        ("refused", OLDER, True),
        ("verified", OLDER, True),
        ("failed", None, False),
    ]
    # Each in the same shape as the newest: a verified one with its sections,
    # a refused one with its reason only, a failed one with its error.
    assert earlier[1]["sections"][0]["statements"] == ["Planted."]
    assert (earlier[0]["sections"], earlier[0]["attempts"]) == (None, 2)
    assert earlier[2]["error"] == "the model API call failed: planted"
    assert (detail["interpretable"], detail["interpretable_reason"]) == (False, ALREADY_VERIFIED)


def test_an_analysis_with_no_interpretation_has_no_earlier_ones(make_client, redis_conn):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)

    detail = client.get(f"/v1/analyses/{analysis_id}").json()

    assert (detail["interpretation"], detail["earlier_interpretations"]) == (None, [])
    assert (detail["interpretable"], detail["interpretable_reason"]) == (True, None)
    # The list a server without the setting offers: one entry, the default.
    assert detail["interpretation_options"] == [
        {"model": "claude-opus-5-5", "effort": "medium", "default": True,
         "available": True, "reason": None},
    ]


def test_the_lost_job_check_on_a_read_looks_at_the_newest_pending_one_only(
    make_client, redis_conn, sessions
):
    # Only one can be pending in practice: a request is refused while any is.
    # Planted: two pending rows whose jobs never existed, under a newer failed
    # one. Any pending row holds every entry, so a read checks the newest
    # pending one, newest or not, and only that one: one queue read per read.
    client = make_client(options=TWO)
    analysis_id = _analysed(client, redis_conn)
    _write(sessions, analysis_id, "queued", None, minutes_ago=30)
    _write(sessions, analysis_id, "queued", None, minutes_ago=20, entry=B)
    _write(sessions, analysis_id, "failed", None, minutes_ago=10)

    detail = client.get(f"/v1/analyses/{analysis_id}").json()

    earlier = detail["earlier_interpretations"]
    assert [(e["status"], e["error"]) for e in earlier] == [
        ("failed", WORKER_LOST), ("queued", None),
    ]
    with sessions() as session:
        assert [row.status for row in _rows(session, analysis_id)] == [
            "failed", "failed", "queued",
        ]


def test_a_failed_interpretation_can_be_asked_for_again(
    make_client, redis_conn, sessions, monkeypatch
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    facts = _facts(sessions, analysis_id)
    _script(monkeypatch, ModelCallError("the model API call failed: APIConnectionError"))
    _ask(client, analysis_id)
    with sessions() as session:
        first_job = _newest(session, analysis_id).job_id
    _drain(redis_conn)
    failed = client.get(f"/v1/analyses/{analysis_id}").json()["interpretation"]
    assert (failed["status"], failed["error"]) == (
        "failed", "the model API call failed: APIConnectionError",
    )
    _script(monkeypatch, narrative_for(facts))

    again = _ask(client, analysis_id)
    _drain(redis_conn)

    assert again.status_code == 202
    assert again.json()["error"] is None
    with sessions() as session:
        assert _newest(session, analysis_id).job_id != first_job
    detail = client.get(f"/v1/analyses/{analysis_id}").json()
    assert detail["interpretation"]["status"] == "verified"
    assert [(e["status"], e["error"]) for e in detail["earlier_interpretations"]] == [
        ("failed", "the model API call failed: APIConnectionError"),
    ]


def test_each_request_gets_its_own_job(make_client, redis_conn, sessions, monkeypatch):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _script(monkeypatch, ModelCallError("down"))
    _ask(client, analysis_id)
    with sessions() as session:
        first = _newest(session, analysis_id).job_id
    _drain(redis_conn)

    _ask(client, analysis_id)

    with sessions() as session:
        second = _newest(session, analysis_id).job_id
    assert first != second
    assert first.startswith(f"interpretation-{analysis_id}-")


def test_the_daily_limit_refuses_with_its_sentence(make_client, redis_conn, sessions, monkeypatch):
    limits = Limits(interpretations_per_day=Limit("interpretations_per_day", 1, 86_400))
    client = make_client(limits=limits)
    first_id = _analysed(client, redis_conn)
    second_id = _analysed(client, redis_conn)
    assert _ask(client, first_id).status_code == 202
    # The analysis does not count the day's requests: it offers what only the limit refuses.
    detail = client.get(f"/v1/analyses/{second_id}").json()

    refused = _ask(client, second_id)

    assert (detail["interpretable"], detail["interpretable_reason"]) == (True, None)
    assert refused.status_code == 429
    assert refused.json()["detail"].startswith(
        "Too many interpretations today for this account. Try again in "
    )
    with sessions() as session:
        assert _rows(session, second_id) == []


def test_every_accepted_request_counts_against_the_daily_limit(
    make_client, redis_conn, sessions
):
    limits = Limits(interpretations_per_day=Limit("interpretations_per_day", 2, 86_400))
    client = make_client(limits=limits)
    analysis_id = _analysed(client, redis_conn)
    _write(sessions, analysis_id, "verified", OLDER, minutes_ago=10)
    assert _ask(client, analysis_id).status_code == 202
    with sessions() as session:
        newest = _newest(session, analysis_id)
        newest.status = "refused"
        newest.interpreter_version = OLDER
        newest.attempts = [{"violations": [{"rule": "bare_value"}]}]
        session.commit()
    assert _ask(client, analysis_id).status_code == 202
    with sessions() as session:
        newest = _newest(session, analysis_id)
        newest.status = "failed"
        newest.error = "planted"
        session.commit()

    refused = _ask(client, analysis_id)

    assert refused.status_code == 429
    with sessions() as session:
        assert [row.status for row in _rows(session, analysis_id)] == [
            "failed", "refused", "verified",
        ]


def test_a_pending_interpretation_whose_job_is_gone_fails_on_the_next_read(
    make_client, redis_conn, sessions
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _ask(client, analysis_id)
    redis_conn.flushdb()

    got = client.get(f"/v1/analyses/{analysis_id}").json()["interpretation"]

    assert (got["status"], got["error"]) == ("failed", WORKER_LOST)


def test_asking_again_frees_a_lost_one_older_than_the_grace_period(
    make_client, redis_conn, sessions
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _ask(client, analysis_id)
    redis_conn.flushdb()
    with sessions() as session:
        row = _newest(session, analysis_id)
        row.created_at = utcnow() - SUBMISSION_GRACE - timedelta(seconds=1)
        session.commit()

    again = _ask(client, analysis_id)

    assert again.status_code == 202
    with sessions() as session:
        assert [(row.status, row.error) for row in _rows(session, analysis_id)] == [
            ("queued", None), ("failed", WORKER_LOST),
        ]


def test_a_lost_one_inside_the_grace_period_still_holds_its_place(
    make_client, redis_conn, sessions
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _ask(client, analysis_id)
    redis_conn.flushdb()

    again = _ask(client, analysis_id)

    assert (again.status_code, again.json()["detail"]) == (409, BEING_WRITTEN)


def test_a_refused_row_is_never_served_as_a_narrative_whatever_it_holds(
    make_client, redis_conn, sessions, monkeypatch
):
    # The worker stores no rendered narrative for a refused interpretation;
    # the API must not serve one even if a row came to hold it.
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    bad = _bad(_facts(sessions, analysis_id))
    _script(monkeypatch, bad, bad)
    _ask(client, analysis_id)
    _drain(redis_conn)
    with sessions() as session:
        row = _newest(session, analysis_id)
        row.rendered = {"sections": [{"name": "summary", "heading": "Summary",
                                       "statements": ["It rose by 2 points."]}], "limits": "x"}
        row.notice = "planted"
        session.commit()

    got = client.get(f"/v1/analyses/{analysis_id}").json()["interpretation"]

    assert (got["status"], got["sections"], got["limits"], got["notice"]) == (
        "refused", None, None, None,
    )


def test_asking_again_keeps_the_failed_one_as_it_was(
    make_client, redis_conn, sessions, monkeypatch
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _script(monkeypatch, ModelCallError("down"))
    _ask(client, analysis_id)
    _drain(redis_conn)
    with sessions() as session:
        failed = _newest(session, analysis_id)
        assert failed.status == "failed"
        failed.attempts = [{"violations": []}]
        failed.prompt_version = "v-old"
        failed.facts_sha256 = "f" * 64
        failed.returned_model = "old-model"
        failed.started_at = utcnow()
        session.commit()
        before = _columns(failed)

    assert _ask(client, analysis_id).status_code == 202

    with sessions() as session:
        new, old = _rows(session, analysis_id)
        assert _columns(old) == before
        assert (new.status, new.job_id != before["job_id"]) == ("queued", True)
        for column in (
            "attempts", "prompt_version", "facts_sha256", "returned_model",
            "started_at", "finished_at", "error", "rendered", "notice", "interpreter_version",
        ):
            assert getattr(new, column) is None, column


def test_a_refused_request_does_not_spend_the_daily_allowance(
    make_client, redis_conn
):
    limits = Limits(interpretations_per_day=Limit("interpretations_per_day", 1, 86_400))
    client = make_client(limits=limits)
    content, schema = scenario_d_upload()
    dataset = client.post("/v1/datasets", files={"file": ("d.csv", content, "text/csv")}).json()
    body = {"dataset_id": dataset["id"], "schema": schema, "random_seed": 0, "estimators": PAIR}
    unfinished = client.post("/v1/analyses", json=body).json()["id"]
    assert _ask(client, unfinished).status_code == 409
    _drain(redis_conn)
    finished = _analysed(client, redis_conn)

    assert _ask(client, finished).status_code == 202


def test_a_request_waits_for_another_request_on_the_same_analysis(
    make_client, redis_conn, sessions
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _write(sessions, analysis_id, "verified", OLDER, minutes_ago=10)
    holder = sessions()
    # FOR NO KEY UPDATE, not FOR UPDATE: the insert's foreign key takes a
    # weaker lock that this one permits, so only a request that itself locks
    # the analysis row waits, and the test fails without that lock.
    holder.execute(
        select(Analysis.id).where(Analysis.id == analysis_id).with_for_update(key_share=True)
    )
    answers = []
    thread = threading.Thread(target=lambda: answers.append(_ask(client, analysis_id)))
    try:
        thread.start()
        thread.join(0.5)
        assert thread.is_alive(), "the request did not wait for the lock on the analysis"
    finally:
        holder.rollback()
        holder.close()
        thread.join(10)
    assert not thread.is_alive()
    assert answers[0].status_code == 202
    with sessions() as session:
        assert [row.status for row in _rows(session, analysis_id)] == ["queued", "verified"]


def _statuses(sessions, analysis_id):
    """The analysis's interpretation rows as (status, error), oldest first."""
    with sessions() as session:
        return [(row.status, row.error) for row in reversed(_rows(session, analysis_id))]


def test_concurrent_requests_over_a_lost_job_write_one_new_row(
    make_client, redis_conn, sessions, monkeypatch
):
    # The lost-job check commits, which releases the lock on the analysis;
    # the request must take it again before it reads the newest row and
    # writes a new one. Without that second lock, a request that arrives in
    # the window reads the same failed row, and both write a row: two jobs,
    # two paid calls.
    #
    # The two requests are driven into that window, not left to race. The
    # route reads the analysis's interpretations through `_interpretations`:
    # the first request reads them once before the lost-job check (read 1) and
    # once after it (read 2); the second request starts only during read 2 and
    # reads them once (read 3). Read 2 waits until the second request has
    # either read them or is waiting for the lock on the analysis, whichever
    # comes first; read 3 then waits until read 2 is done. "The newest row"
    # below is the newest of what each read returns.
    # - With the second lock, the first request holds the lock during read 2,
    #   so the second request waits for it; read 2 sees that wait and goes on,
    #   the first request writes and commits its row, and read 3 then sees it:
    #   one 202 and one 409.
    # - Without it, nothing holds the lock during read 2, so the second request
    #   takes the lock and makes read 3, which sees the failed row and holds
    #   the second request until read 2 has also seen it. Both are accepted:
    #   two 202s and two queued rows. Neither outcome depends on timing.
    from cp_api import app as app_module

    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _write(
        sessions, analysis_id, "queued", None,
        minutes_ago=SUBMISSION_GRACE.total_seconds() / 60 + 60,
    )
    real_read = app_module._interpretations
    reads = 0
    counting = threading.Lock()
    second_may_ask, second_has_read, first_has_read = (
        threading.Event(), threading.Event(), threading.Event(),
    )
    problems: list[str] = []

    def counted_read(session, key):
        nonlocal reads
        with counting:
            reads += 1
            read = reads
        if read == 2:
            # The first request, after its lost-job commit.
            second_may_ask.set()
            deadline = time.monotonic() + 30
            while not second_has_read.is_set():
                if _a_session_waits_for_a_lock(sessions, seconds=0.05):
                    break
                if time.monotonic() > deadline:
                    problems.append(
                        "the second request neither read the newest row nor waited for the lock"
                    )
                    break
            rows = real_read(session, key)
            first_has_read.set()
            return rows
        rows = real_read(session, key)
        if read == 3:
            # The second request: held until the first has made its read 2.
            second_has_read.set()
            if not first_has_read.wait(30):
                problems.append("the first request never read the newest row after its check")
        return rows

    monkeypatch.setattr(app_module, "_interpretations", counted_read)
    codes: dict[str, int] = {}
    sentences: dict[str, object] = {}
    errors: list[Exception] = []

    def ask(name, after=None):
        try:
            if after is not None and not after.wait(30):
                problems.append(
                    "the first request never took the lost-job path, so the second never asked"
                )
                return
            each = TestClient(client.app, headers=client.headers)
            answer = _ask(each, analysis_id)
            codes[name] = answer.status_code
            sentences[name] = answer.json().get("detail")
        except Exception as error:  # reported by the assertion below
            errors.append(error)

    second = threading.Thread(target=ask, args=("second", second_may_ask), daemon=True)
    first = threading.Thread(target=ask, args=("first",), daemon=True)
    second.start()
    first.start()
    first.join(90)
    second.join(90)

    rows = _statuses(sessions, analysis_id)
    assert not errors, f"requests raised {errors!r}; rows, oldest first: {rows}"
    assert not problems, f"{problems}; codes {codes}; rows, oldest first: {rows}"
    # The order is forced by counting the route's reads of the interpretations:
    # one before the lost-job check and one after it in the first request, one
    # in the second. If the route comes to read them more or fewer times, the
    # count no longer picks out those reads and the test would fall back to a
    # race, so it says so instead of passing.
    assert reads == 3, (
        f"the route read the interpretations {reads} times; this test forces the "
        "order of exactly three reads and must be changed with the route"
    )
    assert codes == {"first": 202, "second": 409}, (
        f"expected the first request accepted and the second refused, got {codes}; "
        f"rows, oldest first: {rows}"
    )
    assert [status for status, _ in rows] == ["failed", "queued"], (
        f"expected the lost row failed and one new row queued; rows, oldest first: {rows}"
    )
    assert rows[0][1] == WORKER_LOST
    assert sentences["second"] == BEING_WRITTEN


def _a_session_waits_for_a_lock(sessions, seconds=10.0):
    """Whether another session of this database comes to wait for a row lock it asked for
    with `FOR UPDATE`, within `seconds`."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with sessions() as session:
            waiting = session.scalar(text(
                "SELECT count(*) FROM pg_stat_activity"
                " WHERE datname = current_database() AND wait_event_type = 'Lock'"
                " AND position('FOR UPDATE' in query) > 0"
            ))
        if waiting:
            return True
        time.sleep(0.02)
    return False


def test_a_request_that_waited_for_the_lock_dates_its_row_after_the_holders(
    make_client, redis_conn, sessions
):
    # A request's transaction begins before it waits for the lock on the
    # analysis. If its row were dated by `now()`, the start of that
    # transaction, a failed row written meanwhile by the request holding the
    # lock would sort as newer, and a further request would be accepted while
    # this one's job is still pending.
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    holder = sessions()
    # The same lock mode as the test that a request waits for another, for
    # consistency; here a plain FOR UPDATE would serve as well. The request's
    # own lock on the analysis waits behind it either way.
    holder.execute(
        select(Analysis.id).where(Analysis.id == analysis_id).with_for_update(key_share=True)
    )
    answers = []
    thread = threading.Thread(target=lambda: answers.append(_ask(client, analysis_id)))
    try:
        thread.start()
        # Its transaction has begun once it waits for the lock.
        assert _a_session_waits_for_a_lock(sessions), (
            "the request did not wait for the lock on the analysis"
        )
        # What a concurrent request whose enqueue failed leaves: a failed row,
        # dated by the clock when it is written.
        holder.add(Interpretation(
            analysis_id=uuid.UUID(analysis_id),
            status="failed",
            job_id=f"interpretation-{analysis_id}-holder",
            model="claude-opus-5-5",
            effort="medium",
            error="planted: the queue was down",
            created_at=func.clock_timestamp(),
            finished_at=utcnow(),
        ))
        holder.commit()
    finally:
        holder.rollback()
        holder.close()
        thread.join(10)
    assert not thread.is_alive()
    assert answers[0].status_code == 202

    rows = _statuses(sessions, analysis_id)
    assert [status for status, _ in rows] == ["failed", "queued"], (
        "expected the request's queued row to be newer than the holder's failed row; "
        f"rows, oldest first: {rows}"
    )
    again = _ask(client, analysis_id)
    assert (again.status_code, again.json()["detail"]) == (409, BEING_WRITTEN), (
        f"a further request was answered {again.status_code} while a job is pending; "
        f"rows, oldest first: {_statuses(sessions, analysis_id)}"
    )


# --- the model and effort a request names -----------------------------------


def _entries(sessions, analysis_id):
    """The analysis's rows as (status, model, effort), newest first."""
    with sessions() as session:
        return [(row.status, row.model, row.effort) for row in _rows(session, analysis_id)]


@pytest.mark.parametrize(
    "sent",
    [
        {},
        {"json": {}},
        {"json": {"model": None, "effort": None}},
        {"content": b"null", "headers": {"content-type": "application/json"}},
    ],
    ids=["no-body", "empty-object", "both-null", "null"],
)
def test_a_request_naming_no_entry_is_written_by_the_default(
    make_client, redis_conn, sessions, sent
):
    # The default is the list's first entry, whatever it is: here not the
    # measured one.
    client = make_client(options=(B, A))
    analysis_id = _analysed(client, redis_conn)

    asked = client.post(f"/v1/analyses/{analysis_id}/interpretation", **sent)

    assert asked.status_code == 202
    assert (asked.json()["model"], asked.json()["effort"]) == (B.model, B.effort)
    assert _entries(sessions, analysis_id) == [("queued", B.model, B.effort)]
    got = client.get(f"/v1/analyses/{analysis_id}").json()["interpretation"]
    assert (got["model"], got["effort"]) == (B.model, B.effort)


def test_a_request_naming_an_entry_is_written_by_it(make_client, redis_conn, sessions):
    client = make_client(options=TWO)
    analysis_id = _analysed(client, redis_conn)

    asked = _ask(client, analysis_id, B)

    assert asked.status_code == 202
    assert (asked.json()["model"], asked.json()["effort"]) == (B.model, B.effort)
    assert _entries(sessions, analysis_id) == [("queued", B.model, B.effort)]
    with sessions() as session:
        row = _newest(session, analysis_id)
        assert list(Job.fetch(row.job_id, connection=redis_conn).args) == [str(row.id)]


def test_a_body_with_keys_besides_the_two_is_read_for_the_two(make_client, redis_conn, sessions):
    # As for an analysis's request: a key the body does not declare is ignored.
    client = make_client(options=TWO)
    analysis_id = _analysed(client, redis_conn)

    asked = client.post(
        f"/v1/analyses/{analysis_id}/interpretation",
        json={"model": B.model, "effort": B.effort, "colour": "red"},
    )

    assert asked.status_code == 202
    assert _entries(sessions, analysis_id) == [("queued", B.model, B.effort)]


def test_a_body_that_is_not_json_is_fastapis_422(make_client, redis_conn, sessions):
    client = make_client(options=TWO)
    analysis_id = _analysed(client, redis_conn)

    asked = client.post(
        f"/v1/analyses/{analysis_id}/interpretation",
        content=b"model=claude-opus-5-5",
        headers={"content-type": "application/json"},
    )

    assert asked.status_code == 422
    assert isinstance(asked.json()["detail"], list)
    with sessions() as session:
        assert _rows(session, analysis_id) == []


HALF = [{"model": A.model}, {"effort": A.effort}, {"model": A.model, "effort": None}]
UNLISTED = [
    {"model": A.model, "effort": "high"},
    {"model": B.model, "effort": A.effort},
    {"model": "claude-haiku-5-5", "effort": "medium"},
    {"model": A.model, "effort": "turbo"},
    {"model": "", "effort": ""},
]


@pytest.mark.parametrize(
    ("body", "sentence"),
    [(body, NAME_BOTH_OR_NEITHER) for body in HALF] + [(body, NOT_OFFERED) for body in UNLISTED],
)
def test_a_request_naming_half_an_entry_or_one_not_listed_is_422(
    make_client, redis_conn, sessions, body, sentence
):
    limits = Limits(interpretations_per_day=Limit("interpretations_per_day", 1, 86_400))
    client = make_client(options=TWO, limits=limits)
    analysis_id = _analysed(client, redis_conn)

    asked = client.post(f"/v1/analyses/{analysis_id}/interpretation", json=body)

    assert (asked.status_code, asked.json()["detail"]) == (422, sentence)
    with sessions() as session:
        assert _rows(session, analysis_id) == []
    # Refused before the day's count: the one request allowed is still there.
    assert _ask(client, analysis_id).status_code == 202


def test_the_body_is_judged_before_the_analysis(make_client):
    # The table's order: the 422s come before the 409 for an analysis not finished.
    client = make_client(options=TWO)
    content, schema = scenario_d_upload()
    dataset = client.post("/v1/datasets", files={"file": ("d.csv", content, "text/csv")}).json()
    body = {"dataset_id": dataset["id"], "schema": schema, "random_seed": 0, "estimators": PAIR}
    analysis_id = client.post("/v1/analyses", json=body).json()["id"]
    url = f"/v1/analyses/{analysis_id}/interpretation"

    half = client.post(url, json={"effort": "low"})
    unlisted = client.post(url, json={"model": "claude-haiku-5-5", "effort": "low"})
    listed = _ask(client, analysis_id, B)

    assert (half.status_code, half.json()["detail"]) == (422, NAME_BOTH_OR_NEITHER)
    assert (unlisted.status_code, unlisted.json()["detail"]) == (422, NOT_OFFERED)
    assert (listed.status_code, listed.json()["detail"]) == (409, NOT_SUCCEEDED)
    detail = client.get(f"/v1/analyses/{analysis_id}").json()
    assert _answers(detail) == [(False, NOT_SUCCEEDED), (False, NOT_SUCCEEDED)]
    assert (detail["interpretable"], detail["interpretable_reason"]) == (False, NOT_SUCCEEDED)


def test_an_analysis_without_a_fact_table_offers_no_entry(make_client, redis_conn, sessions):
    client = make_client(options=TWO)
    analysis_id = _analysed(client, redis_conn)
    with sessions() as session:
        session.get(AnalysisSummary, analysis_id).facts = None
        session.commit()

    detail = _offers_what_it_answers(client, sessions, redis_conn, analysis_id, TWO)

    assert _answers(detail) == [(False, NO_FACTS), (False, NO_FACTS)]
    assert detail["interpretable_reason"] == NO_FACTS


def test_a_final_interpretation_is_final_for_its_own_entry_only(
    make_client, redis_conn, sessions, monkeypatch
):
    # Verified from A; then B is accepted and A refused; then B is refused once verified.
    client = make_client(options=TWO)
    analysis_id = _analysed(client, redis_conn)
    facts = _facts(sessions, analysis_id)
    _script(monkeypatch, narrative_for(facts))
    assert _ask(client, analysis_id).status_code == 202
    _drain(redis_conn)

    after_a = _offers_what_it_answers(client, sessions, redis_conn, analysis_id, TWO)

    assert after_a["interpretation"]["status"] == "verified"
    assert _answers(after_a) == [(False, ALREADY_VERIFIED), (True, None)]
    assert (after_a["interpretable"], after_a["interpretable_reason"]) == (True, None)

    _script(monkeypatch, narrative_for(facts))
    asked = _ask(client, analysis_id, B)
    assert asked.status_code == 202
    while_b = _offers_what_it_answers(client, sessions, redis_conn, analysis_id, TWO)
    assert _answers(while_b) == [(False, BEING_WRITTEN), (False, BEING_WRITTEN)]
    _drain(redis_conn)

    after_b = _offers_what_it_answers(client, sessions, redis_conn, analysis_id, TWO)

    assert _answers(after_b) == [(False, ALREADY_VERIFIED), (False, ALREADY_VERIFIED)]
    assert (after_b["interpretable"], after_b["interpretable_reason"]) == (
        False, ALREADY_VERIFIED,
    )
    got, earlier = after_b["interpretation"], after_b["earlier_interpretations"]
    assert (got["status"], got["model"], got["effort"]) == ("verified", B.model, B.effort)
    assert [(e["status"], e["model"], e["effort"]) for e in earlier] == [
        ("verified", A.model, A.effort),
    ]
    assert _entries(sessions, analysis_id) == [
        ("verified", B.model, B.effort), ("verified", A.model, A.effort),
    ]


def test_a_refused_interpretation_is_final_for_its_own_entry_only(
    make_client, redis_conn, sessions, monkeypatch
):
    client = make_client(options=TWO)
    analysis_id = _analysed(client, redis_conn)
    facts = _facts(sessions, analysis_id)
    bad = _bad(facts)
    _script(monkeypatch, bad, bad)
    _ask(client, analysis_id, A)
    _drain(redis_conn)

    after_a = _offers_what_it_answers(client, sessions, redis_conn, analysis_id, TWO)

    assert after_a["interpretation"]["status"] == "refused"
    assert _answers(after_a) == [(False, ALREADY_REFUSED), (True, None)]

    _script(monkeypatch, narrative_for(facts))
    assert _ask(client, analysis_id, B).status_code == 202
    _drain(redis_conn)

    after_b = _offers_what_it_answers(client, sessions, redis_conn, analysis_id, TWO)

    assert _answers(after_b) == [(False, ALREADY_REFUSED), (False, ALREADY_VERIFIED)]
    # None is available, so the reason is the default entry's, not the newest row's.
    assert after_b["interpretation"]["model"] == B.model
    assert (after_b["interpretable"], after_b["interpretable_reason"]) == (
        False, ALREADY_REFUSED,
    )


@pytest.mark.parametrize(
    ("status", "version", "refusal"),
    DECISIONS,
    ids=[f"{status}-{'current' if version == CURRENT_INTERPRETER_VERSION else version}"
         for status, version, _ in DECISIONS],
)
def test_the_newest_from_an_entry_decides_for_that_entry_alone(
    make_client, redis_conn, sessions, status, version, refusal
):
    client = make_client(options=TWO)
    analysis_id = _analysed(client, redis_conn)
    _write(sessions, analysis_id, status, version, minutes_ago=10, entry=B)

    detail = _offers_what_it_answers(client, sessions, redis_conn, analysis_id, TWO)

    assert _answers(detail) == [(True, None), (refusal is None, refusal)]
    assert (detail["interpretable"], detail["interpretable_reason"]) == (True, None)


def test_an_entrys_newest_decides_for_it_whatever_another_entry_wrote_since(
    make_client, redis_conn, sessions
):
    # B's refused row is the analysis's newest; A's verified one decides for A,
    # and as neither is available the default's reason is the analysis's.
    client = make_client(options=TWO)
    analysis_id = _analysed(client, redis_conn)
    _write(sessions, analysis_id, "failed", None, minutes_ago=30, entry=B)
    _write(sessions, analysis_id, "verified", CURRENT_INTERPRETER_VERSION, minutes_ago=20)
    _write(sessions, analysis_id, "refused", CURRENT_INTERPRETER_VERSION, minutes_ago=10, entry=B)

    detail = _offers_what_it_answers(client, sessions, redis_conn, analysis_id, TWO)

    assert _answers(detail) == [(False, ALREADY_VERIFIED), (False, ALREADY_REFUSED)]
    assert (detail["interpretable"], detail["interpretable_reason"]) == (
        False, ALREADY_VERIFIED,
    )


def test_an_entry_out_of_date_or_failed_since_its_verdict_is_offered_again(
    make_client, redis_conn, sessions
):
    # Only the entry's newest counts: A's verified one under an older version,
    # and B's verdict followed by a failure.
    client = make_client(options=TWO)
    analysis_id = _analysed(client, redis_conn)
    _write(sessions, analysis_id, "verified", OLDER, minutes_ago=40)
    _write(sessions, analysis_id, "verified", CURRENT_INTERPRETER_VERSION, minutes_ago=30, entry=B)
    _write(sessions, analysis_id, "failed", CURRENT_INTERPRETER_VERSION, minutes_ago=20, entry=B)

    detail = _offers_what_it_answers(client, sessions, redis_conn, analysis_id, TWO)

    assert _answers(detail) == [(True, None), (True, None)]


@pytest.mark.parametrize("pending", [A, B], ids=["default", "other"])
def test_one_interpretation_at_a_time_whatever_its_entry(
    make_client, redis_conn, sessions, pending
):
    client = make_client(options=TWO)
    analysis_id = _analysed(client, redis_conn)
    assert _ask(client, analysis_id, pending).status_code == 202

    queued = _offers_what_it_answers(client, sessions, redis_conn, analysis_id, TWO)
    with sessions() as session:
        _newest(session, analysis_id).status = "running"
        session.commit()
    running = _offers_what_it_answers(client, sessions, redis_conn, analysis_id, TWO)

    for detail in (queued, running):
        assert _answers(detail) == [(False, BEING_WRITTEN), (False, BEING_WRITTEN)]
        assert (detail["interpretable"], detail["interpretable_reason"]) == (
            False, BEING_WRITTEN,
        )
    assert _entries(sessions, analysis_id) == [("running", pending.model, pending.effort)]


def test_a_lost_one_from_one_entry_frees_a_request_for_another(
    make_client, redis_conn, sessions
):
    client = make_client(options=TWO)
    analysis_id = _analysed(client, redis_conn)
    _ask(client, analysis_id, A)
    redis_conn.flushdb()
    # Inside the grace period it still holds every entry.
    inside = _ask(client, analysis_id, B)
    with sessions() as session:
        row = _newest(session, analysis_id)
        row.created_at = utcnow() - SUBMISSION_GRACE - timedelta(seconds=1)
        session.commit()

    again = _ask(client, analysis_id, B)

    assert (inside.status_code, inside.json()["detail"]) == (409, BEING_WRITTEN)
    assert again.status_code == 202
    with sessions() as session:
        assert [(row.status, row.model, row.error) for row in _rows(session, analysis_id)] == [
            ("queued", B.model, None), ("failed", A.model, WORKER_LOST),
        ]
