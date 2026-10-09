"""`POST /v1/analyses/{id}/interpretation` and the interpretation on the analysis.

Real PostgreSQL and Redis; the jobs run in this process with
`cp_worker.jobs.build_model` replaced, so no test reaches the network.
"""

import json
from datetime import timedelta

import pytest
from cp_api.app import (
    ALREADY_REFUSED,
    ALREADY_VERIFIED,
    BEING_WRITTEN,
    INTERPRETATIONS_OFF,
    NO_FACTS,
    NOT_FINISHED,
    WORKER_LOST,
    create_app,
)
from cp_api.db import AnalysisSummary, Interpretation
from cp_api.limits import Limit, Limits
from cp_llm.facts import FactTable
from cp_llm.model import ModelCallError
from cp_worker import jobs
from cp_worker.worker import make_worker
from fastapi.testclient import TestClient
from worker.data import scenario_d_upload
from worker.narratives import ScriptedReplies, narrative_for

pytestmark = pytest.mark.services
PAIR = ["difference_in_means", "lin_regression"]


@pytest.fixture
def make_client(sessions, redis_conn, worker_env, api_key_for):
    def make(name="alice", interpretation=True, limits=None):
        return TestClient(
            create_app(
                sessions=sessions, redis=redis_conn, limits=limits,
                interpretation=interpretation,
            ),
            headers={"Authorization": f"Bearer {api_key_for(name)}"},
        )

    return make


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


def _ask(client, analysis_id):
    return client.post(f"/v1/analyses/{analysis_id}/interpretation")


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
    assert got["model"] == "claude-opus-5-5"
    assert (got["attempts"], got["reason"], got["error"]) == (None, None, None)
    assert (detail["interpretable"], detail["interpretable_reason"]) == (True, None)


def test_a_server_with_interpretations_off_answers_403_and_offers_none(make_client, redis_conn):
    client = make_client(interpretation=False)
    analysis_id = _analysed(client, redis_conn)

    asked = _ask(client, analysis_id)

    assert (asked.status_code, asked.json()["detail"]) == (403, INTERPRETATIONS_OFF)
    detail = client.get(f"/v1/analyses/{analysis_id}").json()
    assert (detail["interpretable"], detail["interpretable_reason"]) == (False, None)
    assert detail["interpretation"] is None


def test_someone_elses_analysis_is_404(make_client, redis_conn):
    analysis_id = _analysed(make_client("alice"), redis_conn)

    asked = _ask(make_client("bob"), analysis_id)

    assert (asked.status_code, asked.json()["detail"]) == (404, f"no analysis {analysis_id}")


def test_an_analysis_that_has_not_finished_is_409(make_client):
    client = make_client()
    content, schema = scenario_d_upload()
    dataset = client.post("/v1/datasets", files={"file": ("d.csv", content, "text/csv")}).json()
    body = {"dataset_id": dataset["id"], "schema": schema, "random_seed": 0, "estimators": PAIR}
    analysis_id = client.post("/v1/analyses", json=body).json()["id"]

    asked = _ask(client, analysis_id)

    assert (asked.status_code, asked.json()["detail"]) == (409, NOT_FINISHED)
    detail = client.get(f"/v1/analyses/{analysis_id}").json()
    assert (detail["interpretable"], detail["interpretable_reason"]) == (False, NOT_FINISHED)


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


def test_asking_while_one_is_being_written_is_409(make_client, redis_conn):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    assert _ask(client, analysis_id).status_code == 202

    again = _ask(client, analysis_id)

    assert (again.status_code, again.json()["detail"]) == (409, BEING_WRITTEN)


def test_a_verified_interpretation_is_final(make_client, redis_conn, sessions, monkeypatch):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _script(monkeypatch, narrative_for(_facts(sessions, analysis_id)))
    _ask(client, analysis_id)
    _drain(redis_conn)

    again = _ask(client, analysis_id)

    assert (again.status_code, again.json()["detail"]) == (409, ALREADY_VERIFIED)


def test_a_refused_interpretation_is_final_and_its_narrative_is_not_served(
    make_client, redis_conn, sessions, monkeypatch
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    bad = _bad(_facts(sessions, analysis_id))
    _script(monkeypatch, bad, bad)
    _ask(client, analysis_id)
    _drain(redis_conn)

    got = client.get(f"/v1/analyses/{analysis_id}").json()["interpretation"]
    again = _ask(client, analysis_id)

    assert got["status"] == "refused"
    assert got["attempts"] == 2
    assert got["reason"] == "The last attempt broke 1 rule(s): bare_value."
    assert (got["sections"], got["limits"], got["notice"]) == (None, None, None)
    assert (again.status_code, again.json()["detail"]) == (409, ALREADY_REFUSED)


def test_a_failed_interpretation_can_be_asked_for_again(
    make_client, redis_conn, sessions, monkeypatch
):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    facts = _facts(sessions, analysis_id)
    _script(monkeypatch, ModelCallError("the model API call failed: APIConnectionError"))
    first = _ask(client, analysis_id).json()
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
        assert session.get(Interpretation, analysis_id).job_id != first.get("job_id", "")
    assert client.get(f"/v1/analyses/{analysis_id}").json()["interpretation"]["status"] == (
        "verified"
    )


def test_each_request_gets_its_own_job(make_client, redis_conn, sessions, monkeypatch):
    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _script(monkeypatch, ModelCallError("down"))
    _ask(client, analysis_id)
    with sessions() as session:
        first = session.get(Interpretation, analysis_id).job_id
    _drain(redis_conn)

    _ask(client, analysis_id)

    with sessions() as session:
        second = session.get(Interpretation, analysis_id).job_id
    assert first != second
    assert first.startswith(f"interpretation-{analysis_id}-")


def test_the_daily_limit_refuses_with_its_sentence(make_client, redis_conn, sessions, monkeypatch):
    limits = Limits(interpretations_per_day=Limit("interpretations_per_day", 1, 86_400))
    client = make_client(limits=limits)
    first_id = _analysed(client, redis_conn)
    second_id = _analysed(client, redis_conn)
    assert _ask(client, first_id).status_code == 202

    refused = _ask(client, second_id)

    assert refused.status_code == 429
    assert refused.json()["detail"].startswith(
        "Too many interpretations today for this account. Try again in "
    )
    with sessions() as session:
        assert session.get(Interpretation, second_id) is None


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
    from cp_api.app import SUBMISSION_GRACE
    from cp_api.db import utcnow

    client = make_client()
    analysis_id = _analysed(client, redis_conn)
    _ask(client, analysis_id)
    redis_conn.flushdb()
    with sessions() as session:
        row = session.get(Interpretation, analysis_id)
        row.created_at = utcnow() - SUBMISSION_GRACE - timedelta(seconds=1)
        session.commit()

    again = _ask(client, analysis_id)

    assert again.status_code == 202


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
        row = session.get(Interpretation, analysis_id)
        row.rendered = {"sections": [{"name": "summary", "heading": "Summary",
                                       "statements": ["It rose by 2 points."]}], "limits": "x"}
        row.notice = "planted"
        session.commit()

    got = client.get(f"/v1/analyses/{analysis_id}").json()["interpretation"]

    assert (got["status"], got["sections"], got["limits"], got["notice"]) == (
        "refused", None, None, None,
    )
