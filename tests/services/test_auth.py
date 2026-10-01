"""Every route is authenticated, and every lookup is scoped to its owner.

The coverage tests walk the app's real route table rather than a hand-kept
list. A route added later without `CurrentUser`, or with a lookup that is not
owner-scoped, fails here by name. A route with a path parameter this file does
not know how to fill fails too, rather than being skipped.
"""

import pytest
from cp_api.accounts import revoke_key
from cp_api.app import create_app
from cp_api.auth import UNAUTHORIZED, parse_key
from cp_api.db import QUEUE_NAME
from cp_worker.worker import make_worker
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from rq import Queue
from worker.data import scenario_d_upload

OPEN_ROUTES = {("GET", "/v1/healthz")}
PAIR = ["difference_in_means", "lin_regression"]


def _routes():
    app = create_app(sessions=object(), redis=object())  # built only to list routes
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith("/v1"):
            for method in sorted(route.methods):
                yield method, route.path


ROUTES = sorted(_routes())


def _client(sessions, redis_conn, key=None):
    headers = {} if key is None else {"Authorization": f"Bearer {key}"}
    return TestClient(create_app(sessions=sessions, redis=redis_conn), headers=headers)


@pytest.fixture
def alices_analysis(sessions, redis_conn, worker_env, api_key_for):
    """A finished analysis of alice's, and every path parameter a route could need."""
    key = api_key_for("alice")
    alice = _client(sessions, redis_conn, key)
    content, schema = scenario_d_upload()
    dataset = alice.post("/v1/datasets", files={"file": ("d.csv", content, "text/csv")}).json()
    analysis_id = alice.post(
        "/v1/analyses",
        json={"dataset_id": dataset["id"], "schema": schema, "estimators": PAIR},
    ).json()["id"]
    make_worker(redis_conn).work(burst=True)
    markdown = alice.get(
        f"/v1/analyses/{analysis_id}/estimates/lin_regression/report.md"
    ).text
    figure = markdown.split("](", 1)[1].split(")", 1)[0]
    return {
        "key": key,
        "dataset_id": dataset["id"],
        "params": {"analysis_id": analysis_id, "method": "lin_regression", "filename": figure},
    }


def _request(client, method, path, fixture):
    try:
        url = path.format(**fixture["params"])
    except KeyError as exc:
        raise AssertionError(
            f"{method} {path} has a path parameter {exc} this test cannot fill; extend "
            "alices_analysis so the new route is checked rather than skipped"
        ) from None
    if (method, path) == ("POST", "/v1/datasets"):
        return client.post(url, files={"file": ("d.csv", b"a,b\n1,2\n", "text/csv")})
    if (method, path) == ("POST", "/v1/analyses"):
        return client.post(url, json={"dataset_id": fixture["dataset_id"], "schema": {}})
    return client.request(method, url)


@pytest.mark.parametrize(("method", "path"), [r for r in ROUTES if r not in OPEN_ROUTES])
def test_every_route_but_healthz_refuses_a_request_without_a_key(
    sessions, redis_conn, alices_analysis, method, path
):
    response = _request(_client(sessions, redis_conn), method, path, alices_analysis)

    assert response.status_code == 401, f"{method} {path} answered {response.status_code}"
    assert response.json() == {"detail": UNAUTHORIZED}
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    ("method", "path"), [r for r in ROUTES if "{analysis_id}" in r[1]]
)
def test_another_users_analysis_is_not_found_on_every_route(
    sessions, redis_conn, alices_analysis, api_key_for, method, path
):
    bob = _client(sessions, redis_conn, api_key_for("bob"))

    response = _request(bob, method, path, alices_analysis)

    assert response.status_code == 404, f"{method} {path} answered {response.status_code}"


def test_the_route_table_holds_no_surprises():
    # Makes a new route a deliberate addition to this file's checks.
    assert sorted([
        ("GET", "/v1/analyses/{analysis_id}"),
        ("GET", "/v1/analyses/{analysis_id}/estimates/{method}/report.html"),
        ("GET", "/v1/analyses/{analysis_id}/estimates/{method}/report.md"),
        ("GET", "/v1/analyses/{analysis_id}/estimates/{method}/{filename}"),
        ("GET", "/v1/analyses/{analysis_id}/manifest"),
        ("GET", "/v1/analyses/{analysis_id}/summary"),
        ("GET", "/v1/healthz"),
        ("POST", "/v1/analyses"),
        ("POST", "/v1/datasets"),
    ]) == ROUTES


def test_healthz_needs_no_key(sessions, redis_conn):
    assert _client(sessions, redis_conn).get("/v1/healthz").status_code == 200


def test_not_found_and_not_yours_are_indistinguishable(
    sessions, redis_conn, alices_analysis, api_key_for
):
    bob = _client(sessions, redis_conn, api_key_for("bob"))
    theirs = bob.get(f"/v1/analyses/{alices_analysis['params']['analysis_id']}")
    missing = bob.get("/v1/analyses/00000000-0000-0000-0000-000000000000")

    assert theirs.status_code == missing.status_code == 404
    assert theirs.json()["detail"].replace(
        alices_analysis["params"]["analysis_id"], "ID"
    ) == missing.json()["detail"].replace("00000000-0000-0000-0000-000000000000", "ID")


def test_the_same_bytes_from_two_users_are_two_datasets(sessions, redis_conn, api_key_for):
    content, _ = scenario_d_upload()
    upload = {"file": ("d.csv", content, "text/csv")}

    alice = _client(sessions, redis_conn, api_key_for("alice")).post("/v1/datasets", files=upload)
    bob = _client(sessions, redis_conn, api_key_for("bob")).post("/v1/datasets", files=upload)

    assert (alice.status_code, bob.status_code) == (201, 201)
    assert alice.json()["id"] != bob.json()["id"]


def test_an_analysis_of_another_users_dataset_is_not_found_and_nothing_is_queued(
    sessions, redis_conn, alices_analysis, api_key_for
):
    _, schema = scenario_d_upload()
    bob = _client(sessions, redis_conn, api_key_for("bob"))
    queued_before = Queue(QUEUE_NAME, connection=redis_conn).count

    response = bob.post(
        "/v1/analyses", json={"dataset_id": alices_analysis["dataset_id"], "schema": schema}
    )

    assert response.status_code == 404
    assert Queue(QUEUE_NAME, connection=redis_conn).count == queued_before


@pytest.mark.parametrize(
    "header",
    [
        "Bearer",
        "Basic dXNlcjpwYXNz",
        "Bearer not-a-key",
        "Bearer cpk_AAAAAAAAAAAA_wrongsecret",  # an unknown key id
    ],
)
def test_malformed_or_unknown_credentials_are_401(sessions, redis_conn, header):
    client = TestClient(create_app(sessions=sessions, redis=redis_conn))

    response = client.get(
        "/v1/analyses/00000000-0000-0000-0000-000000000000", headers={"Authorization": header}
    )

    assert response.status_code == 401
    assert response.json() == {"detail": UNAUTHORIZED}


def test_the_right_key_id_with_the_wrong_secret_is_401(sessions, redis_conn, api_key_for):
    key_id, _ = parse_key(api_key_for("alice"))

    response = _client(sessions, redis_conn, f"cpk_{key_id}_wrongsecret").get(
        "/v1/analyses/00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 401


def test_a_revoked_key_is_401(sessions, redis_conn, api_key_for):
    key = api_key_for("alice")
    client = _client(sessions, redis_conn, key)
    assert client.get("/v1/analyses/00000000-0000-0000-0000-000000000000").status_code == 404
    with sessions() as session:
        revoke_key(session, parse_key(key)[0])
        session.commit()

    assert client.get("/v1/analyses/00000000-0000-0000-0000-000000000000").status_code == 401
