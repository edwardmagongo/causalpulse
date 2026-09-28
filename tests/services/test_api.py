import json
import re

import pytest
from cp_api import validation
from cp_api.app import WORKER_LOST, create_app
from cp_api.db import QUEUE_NAME, Analysis
from cp_causal.manifest import dataset_resolver, replay
from cp_causal.registry import get_estimator
from cp_core.manifest import AnalysisManifest
from cp_core.results import estimate_summary
from cp_simulation.scenarios import resolve_scenario_reference
from fastapi.testclient import TestClient
from redis import Redis
from rq import Queue, SimpleWorker
from rq.job import Job
from worker.data import scenario_d_upload

pytestmark = pytest.mark.services
PAIR = ["difference_in_means", "lin_regression"]


@pytest.fixture
def client(sessions, redis_conn, worker_env):
    return TestClient(create_app(sessions=sessions, redis=redis_conn))


def _drain(redis_conn):
    SimpleWorker([Queue(QUEUE_NAME, connection=redis_conn)], connection=redis_conn).work(
        burst=True
    )


def _upload(client, content, name="d.csv"):
    return client.post("/v1/datasets", files={"file": (name, content, "text/csv")})


def _submit(client, estimators=PAIR):
    content, schema = scenario_d_upload()
    dataset = _upload(client, content).json()
    body = {"dataset_id": dataset["id"], "schema": schema, "random_seed": 0}
    if estimators is not None:
        body["estimators"] = estimators
    return client.post("/v1/analyses", json=body)


def test_the_same_bytes_uploaded_twice_are_one_dataset(client):
    content, _ = scenario_d_upload()

    first = _upload(client, content)
    second = _upload(client, content, name="again.csv")

    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json()["id"] == second.json()["id"]
    assert "unit_id" in first.json()["columns"]


def test_an_oversized_upload_is_413(client, monkeypatch):
    monkeypatch.setattr(validation, "MAX_UPLOAD_BYTES", 10)

    assert _upload(client, b"a,b\n1,2\n3,4\n").status_code == 413


def test_a_schema_column_the_upload_lacks_is_422_before_anything_is_queued(client, redis_conn):
    content, schema = scenario_d_upload()
    dataset = _upload(client, content).json()

    response = client.post(
        "/v1/analyses",
        json={"dataset_id": dataset["id"], "schema": {**schema, "covariates": ["nope"]}},
    )

    assert response.status_code == 422
    assert "nope" in response.json()["detail"]
    assert Queue(QUEUE_NAME, connection=redis_conn).count == 0


def test_upload_to_report_end_to_end(client, redis_conn):
    submitted = _submit(client)
    assert submitted.status_code == 202
    analysis_id = submitted.json()["id"]

    _drain(redis_conn)
    body = client.get(f"/v1/analyses/{analysis_id}").json()

    assert body["status"] == "succeeded"
    assert [e["method"] for e in body["estimates"]] == PAIR
    assert all(e["band"] for e in body["estimates"])
    for method in PAIR:
        base = f"/v1/analyses/{analysis_id}/estimates/{method}"
        html = client.get(f"{base}/report.html")
        assert html.status_code == 200 and "<html" in html.text
        markdown = client.get(f"{base}/report.md")
        links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown.text)
        assert links
        for link in links:
            figure = client.get(f"{base}/{link}")
            assert figure.status_code == 200
            assert figure.headers["content-type"].startswith("image/svg+xml")
    summary = client.get(f"/v1/analyses/{analysis_id}/summary").json()
    assert summary["comparisons"][0]["methods"] == PAIR


def test_replaying_the_apis_manifest_reproduces_its_numbers_exactly(
    client, redis_conn, tmp_path, monkeypatch
):
    analysis_id = _submit(client).json()["id"]
    _drain(redis_conn)
    stored = {
        e["method"]: e["summary"]
        for e in client.get(f"/v1/analyses/{analysis_id}").json()["estimates"]
    }
    manifest_json = client.get(f"/v1/analyses/{analysis_id}/manifest").json()
    content, _ = scenario_d_upload()
    (tmp_path / manifest_json["dataset"]["path"]).write_bytes(content)
    monkeypatch.chdir(tmp_path)

    replayed = replay(
        AnalysisManifest.from_json(json.dumps(manifest_json)),
        dataset_resolver(resolve_scenario_reference),
    )

    for estimate in replayed:
        again = estimate_summary(estimate)
        for key in ("point", "se", "ci_lower", "ci_upper", "bootstrap_ci_lower",
                    "bootstrap_ci_upper", "p_value"):
            assert again[key] == stored[estimate.method][key], (estimate.method, key)


def test_an_ineligible_request_fails_with_the_routers_reasons(client, redis_conn):
    analysis_id = _submit(client, ["synthetic_control"]).json()["id"]

    _drain(redis_conn)
    body = client.get(f"/v1/analyses/{analysis_id}").json()

    assert body["status"] == "failed"
    assert "synthetic_control: requires panel data" in body["error"]


def test_a_failed_fit_is_partial_and_has_no_report(client, redis_conn, monkeypatch):
    def boom(self, dataset, config):
        raise RuntimeError("boom")

    monkeypatch.setattr(get_estimator("lin_regression"), "fit", boom)
    analysis_id = _submit(client).json()["id"]

    _drain(redis_conn)
    body = client.get(f"/v1/analyses/{analysis_id}").json()

    assert (body["status"], body["partial"]) == ("succeeded", True)
    report = client.get(f"/v1/analyses/{analysis_id}/estimates/lin_regression/report.html")
    assert report.status_code == 404
    assert "boom" in report.json()["detail"]


def test_a_job_that_vanished_is_reported_as_a_lost_worker(client, redis_conn, sessions):
    analysis_id = _submit(client).json()["id"]
    Job.fetch(analysis_id, connection=redis_conn).delete()

    body = client.get(f"/v1/analyses/{analysis_id}").json()

    assert (body["status"], body["error"]) == ("failed", WORKER_LOST)


def test_an_unreachable_queue_is_503_and_the_analysis_is_marked_failed(sessions, redis_conn):
    content, schema = scenario_d_upload()
    dataset = TestClient(create_app(sessions=sessions, redis=redis_conn)).post(
        "/v1/datasets", files={"file": ("d.csv", content, "text/csv")}
    ).json()
    broken = TestClient(
        create_app(sessions=sessions, redis=Redis.from_url("redis://localhost:1/0"))
    )

    response = broken.post("/v1/analyses", json={"dataset_id": dataset["id"], "schema": schema})

    assert response.status_code == 503
    with sessions() as session:
        (analysis,) = session.query(Analysis).all()
        assert analysis.status == "failed"


def test_healthz(client):
    assert client.get("/v1/healthz").json() == {"postgres": "ok", "redis": "ok"}
