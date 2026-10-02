"""The routes the dashboard reads that phase 2 did not have: who am I, the
estimator catalogue, the two lists, and one dataset with its columns."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from cp_api.app import WORKER_LOST, create_app
from cp_api.db import Analysis, Dataset, User
from cp_core.catalogue import ESTIMATOR_FAMILIES
from fastapi.testclient import TestClient
from rq.job import Job
from sqlalchemy import event, select
from worker.data import scenario_d_upload

PAIR = ["difference_in_means", "lin_regression"]
BASE = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


def _client(sessions, redis_conn, key):
    return TestClient(
        create_app(sessions=sessions, redis=redis_conn),
        headers={"Authorization": f"Bearer {key}"},
    )


@pytest.fixture
def alice(sessions, redis_conn, worker_env, api_key_for):
    return _client(sessions, redis_conn, api_key_for("alice"))


@pytest.fixture
def bob(sessions, redis_conn, worker_env, api_key_for):
    return _client(sessions, redis_conn, api_key_for("bob"))


def _upload(client, content, name="d.csv"):
    return client.post("/v1/datasets", files={"file": (name, content, "text/csv")}).json()


def _submit(client, dataset_id, estimators=PAIR):
    _, schema = scenario_d_upload()
    body = {"dataset_id": dataset_id, "schema": schema, "estimators": estimators}
    return client.post("/v1/analyses", json=body).json()["id"]


def _walk(client, path, limit):
    """Every item of a list, following `next_cursor` to the end."""
    items, cursor, pages = [], None, 0
    while True:
        params = {"limit": limit} if cursor is None else {"limit": limit, "cursor": cursor}
        page = client.get(path, params=params).json()
        items.extend(page["items"])
        cursor, pages = page["next_cursor"], pages + 1
        if cursor is None:
            return items, pages


def _seed_rows(sessions, owner_name, moments):
    """One dataset and one finished analysis per moment, created at that moment."""
    with sessions() as session:
        owner_id = session.scalars(select(User.id).where(User.name == owner_name)).one()
        for index, moment in enumerate(moments):
            dataset = Dataset(
                owner_id=owner_id,
                filename=f"seed-{index}.csv",
                sha256=f"{index:064d}",
                n_bytes=8,
                csv=b"a,b\n1,2\n",
                created_at=moment,
            )
            session.add(dataset)
            session.flush()
            session.add(
                Analysis(
                    owner_id=owner_id,
                    dataset_id=dataset.id,
                    causal_schema={},
                    requested_estimators=None,
                    random_seed=0,
                    status="succeeded",
                    created_at=moment,
                )
            )
        session.commit()


def test_me_names_the_keys_owner(alice, bob):
    assert alice.get("/v1/me").json() == {"name": "alice"}
    assert bob.get("/v1/me").json() == {"name": "bob"}


def test_the_catalogue_lists_every_estimator_with_its_family(alice):
    listed = alice.get("/v1/estimators").json()["estimators"]

    assert {entry["name"]: entry["family"] for entry in listed} == ESTIMATOR_FAMILIES
    assert [entry["family"] for entry in listed][:1] == ["cross_sectional"]
    assert [entry["family"] for entry in listed][-1:] == ["donor_pool"]


def test_a_dataset_reads_back_with_the_columns_its_upload_reported(alice):
    content, _ = scenario_d_upload()
    uploaded = _upload(alice, content, name="trial.csv")

    read = alice.get(f"/v1/datasets/{uploaded['id']}")

    assert read.status_code == 200
    body = read.json()
    assert body["columns"] == uploaded["columns"] and len(body["columns"]) > 3
    assert {k: body[k] for k in ("id", "filename", "sha256", "n_bytes")} == {
        k: uploaded[k] for k in ("id", "filename", "sha256", "n_bytes")
    }
    assert datetime.fromisoformat(body["created_at"]).tzinfo is not None


def test_another_users_dataset_reads_as_missing(alice, bob):
    content, _ = scenario_d_upload()
    theirs = bob.get(f"/v1/datasets/{_upload(alice, content)['id']}")
    missing = bob.get(f"/v1/datasets/{uuid.UUID(int=0)}")

    assert theirs.status_code == missing.status_code == 404


def test_the_dataset_list_never_selects_the_csv(alice, sessions):
    content, _ = scenario_d_upload()
    uploaded = _upload(alice, content)
    statements: list[str] = []
    engine = sessions.kw["bind"]

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        listed = alice.get("/v1/datasets").json()
        on_the_list = [s for s in statements if "FROM datasets" in s]
        statements.clear()
        alice.get(f"/v1/datasets/{uploaded['id']}")
        on_the_read = [s for s in statements if "FROM datasets" in s]
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert [item["id"] for item in listed["items"]] == [uploaded["id"]]
    assert on_the_list and not any("datasets.csv" in s for s in on_the_list)
    # The control: the single read does load the bytes, and this capture sees it.
    assert any("datasets.csv" in s for s in on_the_read)


@pytest.mark.parametrize("path", ["/v1/datasets", "/v1/analyses"])
def test_paging_returns_every_row_once_even_when_rows_share_a_created_at(
    alice, sessions, path
):
    # Three rows share one instant, and the page boundary falls inside the tie.
    moments = [BASE, BASE + timedelta(seconds=1), *[BASE + timedelta(seconds=2)] * 3,
               BASE + timedelta(seconds=3), BASE + timedelta(seconds=4)]
    _seed_rows(sessions, "alice", moments)

    in_one_page = alice.get(path, params={"limit": 200}).json()
    paged, pages = _walk(alice, path, limit=2)

    assert in_one_page["next_cursor"] is None and len(in_one_page["items"]) == 7
    assert [item["id"] for item in paged] == [item["id"] for item in in_one_page["items"]]
    assert pages == 4
    stamps = [(item["created_at"], item["id"]) for item in in_one_page["items"]]
    keyed = [(datetime.fromisoformat(at), uuid.UUID(row)) for at, row in stamps]
    assert keyed == sorted(keyed, reverse=True)


@pytest.mark.parametrize("path", ["/v1/datasets", "/v1/analyses"])
def test_a_page_that_ends_exactly_at_the_last_row_has_no_cursor(alice, sessions, path):
    _seed_rows(sessions, "alice", [BASE, BASE + timedelta(seconds=1)])

    page = alice.get(path, params={"limit": 2}).json()

    assert len(page["items"]) == 2 and page["next_cursor"] is None


@pytest.mark.parametrize("path", ["/v1/datasets", "/v1/analyses"])
@pytest.mark.parametrize("params", [{"cursor": "not-a-cursor"}, {"limit": 0}, {"limit": 201}])
def test_a_bad_cursor_or_limit_is_422(alice, path, params):
    assert alice.get(path, params=params).status_code == 422


@pytest.mark.parametrize("path", ["/v1/datasets", "/v1/analyses"])
def test_a_list_holds_only_the_callers_rows(alice, bob, sessions, path):
    _seed_rows(sessions, "alice", [BASE, BASE + timedelta(seconds=1)])
    _seed_rows(sessions, "bob", [BASE + timedelta(seconds=2)])

    hers = alice.get(path).json()["items"]
    his = bob.get(path).json()["items"]

    assert (len(hers), len(his)) == (2, 1)
    assert not {item["id"] for item in hers} & {item["id"] for item in his}


def test_the_analysis_list_carries_what_the_page_shows(alice, redis_conn):
    content, _ = scenario_d_upload()
    dataset = _upload(alice, content, name="trial.csv")
    analysis_id = _submit(alice, dataset["id"])

    (item,) = alice.get("/v1/analyses").json()["items"]

    assert item["id"] == analysis_id
    assert item["dataset_id"] == dataset["id"]
    assert item["dataset_filename"] == "trial.csv"
    assert item["requested_estimators"] == PAIR
    assert (item["status"], item["partial"], item["error"]) == ("queued", False, None)
    assert item["started_at"] is None and item["finished_at"] is None


def test_the_list_marks_a_lost_job_failed_and_leaves_a_live_one_queued(alice, redis_conn):
    content, _ = scenario_d_upload()
    dataset = _upload(alice, content)
    live = _submit(alice, dataset["id"])
    lost = _submit(alice, dataset["id"])
    Job.fetch(lost, connection=redis_conn).delete()

    by_id = {item["id"]: item for item in alice.get("/v1/analyses").json()["items"]}

    assert (by_id[lost]["status"], by_id[lost]["error"]) == ("failed", WORKER_LOST)
    assert by_id[lost]["finished_at"] is not None
    assert (by_id[live]["status"], by_id[live]["error"]) == ("queued", None)
    # The single read agrees with the list.
    assert alice.get(f"/v1/analyses/{lost}").json()["status"] == "failed"
