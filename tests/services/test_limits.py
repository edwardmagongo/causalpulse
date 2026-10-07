"""What the limits do to real requests, against a real PostgreSQL and Redis.

Each test builds its app with the limits it needs, passed to `create_app`
rather than read from the process environment; one test at the end checks the
environment is what an app given no limits reads. That no limit refuses
anything when none is configured is shown by every other service test, which
builds its app without any.
"""

import re
from concurrent.futures import ThreadPoolExecutor

import pytest
from cp_api.accounts import revoke_key
from cp_api.app import create_app
from cp_api.auth import parse_key
from cp_api.db import Analysis, ApiKey
from cp_api.limits import FixedWindow, Limit, Limits
from fastapi.testclient import TestClient
from redis import Redis
from sqlalchemy import select, update
from worker.data import scenario_d_upload

pytestmark = pytest.mark.services
DAY = 86_400
TESTCLIENT = "testclient"  # the peer address Starlette's test client reports
ELSEWHERE = ("198.51.100.4", 50_000)
WAIT = r"Try again in \d+ (second|minute|hour)s?\."


def uploads(count):
    return Limits(uploads_per_day=Limit("uploads_per_day", count, DAY))


def analyses(count):
    return Limits(analyses_per_day=Limit("analyses_per_day", count, DAY))


def active(count, per_day=None):
    return Limits(
        analyses_active=Limit("analyses_active", count, 60),
        analyses_per_day=None if per_day is None else Limit("analyses_per_day", per_day, DAY),
    )


def signins(count=10):
    return Limits(signin_failures=Limit("signin_failures", count, 900))


def _client(app, key=None, peer=None, headers=None):
    sent = dict(headers or {})
    if key is not None:
        sent["Authorization"] = f"Bearer {key}"
    if peer is None:
        return TestClient(app, headers=sent)
    return TestClient(app, headers=sent, client=peer)


def _csv(n):
    return f"a,b\n{n},2\n".encode()


def _upload(client, content):
    return client.post("/v1/datasets", files={"file": ("d.csv", content, "text/csv")})


def _refused(response, detail, longest):
    assert response.status_code == 429, response.text
    assert re.fullmatch(detail, response.json()["detail"]), response.json()
    assert set(response.json()) == {"detail"}
    assert re.fullmatch(r"[1-9]\d*", response.headers["retry-after"])
    assert 1 <= int(response.headers["retry-after"]) <= longest
    return True


def _upload_refused(response):
    return _refused(response, rf"Too many uploads today for this account\. {WAIT}", DAY)


def _analyses_refused(response):
    return _refused(response, rf"Too many analyses today for this account\. {WAIT}", DAY)


def _signin_refused(response):
    return _refused(response, rf"Too many failed sign-ins from this address\. {WAIT}", 900)


@pytest.fixture
def scenario():
    return scenario_d_upload()


def _submitter(client, scenario):
    content, schema = scenario
    dataset = _upload(client, content).json()

    def submit(**changed):
        body = {"dataset_id": dataset["id"], "schema": schema, "estimators": ["lin_regression"]}
        return client.post("/v1/analyses", json={**body, **changed})

    return submit


def test_the_upload_one_past_the_days_limit_is_refused(sessions, redis_conn, api_key_for):
    alice = _client(create_app(sessions, redis_conn, uploads(2)), api_key_for("alice"))

    codes = [_upload(alice, _csv(n)).status_code for n in range(2)]
    third = _upload(alice, _csv(2))

    assert codes == [201, 201]
    assert _upload_refused(third)
    # Refused, not stored.
    assert len(alice.get("/v1/datasets").json()["items"]) == 2


def test_an_upload_already_held_still_counts(sessions, redis_conn, api_key_for):
    alice = _client(create_app(sessions, redis_conn, uploads(2)), api_key_for("alice"))

    codes = [_upload(alice, _csv(0)).status_code for _ in range(2)]

    assert codes == [201, 200]
    assert _upload_refused(_upload(alice, _csv(0)))


def test_an_upload_that_fails_validation_keeps_its_place_in_the_count(
    sessions, redis_conn, api_key_for
):
    alice = _client(create_app(sessions, redis_conn, uploads(2)), api_key_for("alice"))

    codes = [_upload(alice, b"a,b\n").status_code, _upload(alice, _csv(0)).status_code]

    assert codes == [422, 201]
    assert _upload_refused(_upload(alice, _csv(1)))


def test_two_users_have_an_allowance_each_and_two_keys_of_one_user_share_one(
    sessions, redis_conn, api_key_for
):
    app = create_app(sessions, redis_conn, uploads(2))
    first, second = (_client(app, api_key_for("alice")) for _ in range(2))
    bob = _client(app, api_key_for("bob"))

    assert [_upload(first, _csv(0)).status_code, _upload(second, _csv(1)).status_code] == [
        201, 201,
    ]  # fmt: skip
    assert _upload_refused(_upload(first, _csv(2)))
    assert _upload_refused(_upload(second, _csv(2)))
    assert _upload(bob, _csv(2)).status_code == 201


def test_only_uploads_are_counted_as_uploads(sessions, redis_conn, api_key_for):
    alice = _client(create_app(sessions, redis_conn, uploads(1)), api_key_for("alice"))

    assert _upload(alice, _csv(0)).status_code == 201
    assert [alice.get("/v1/datasets").status_code for _ in range(3)] == [200, 200, 200]


def test_the_analysis_one_past_the_days_limit_is_refused(
    sessions, redis_conn, api_key_for, scenario
):
    app = create_app(sessions, redis_conn, analyses(2))
    submit = _submitter(_client(app, api_key_for("alice")), scenario)
    submit_as_bob = _submitter(_client(app, api_key_for("bob")), scenario)

    codes = [submit().status_code for _ in range(2)]
    third = submit()

    assert codes == [202, 202]
    assert _analyses_refused(third)
    with sessions() as session:
        assert len(session.scalars(select(Analysis.id)).all()) == 2
    assert submit_as_bob().status_code == 202


def test_an_analysis_that_fails_validation_keeps_its_place_in_the_count(
    sessions, redis_conn, api_key_for, scenario
):
    alice = _client(create_app(sessions, redis_conn, analyses(2)), api_key_for("alice"))
    submit = _submitter(alice, scenario)

    codes = [submit(estimators=[]).status_code, submit().status_code]

    assert codes == [422, 202]
    assert _analyses_refused(submit())


def _finish(sessions, analysis_id, status="succeeded"):
    with sessions() as session:
        session.execute(update(Analysis).where(Analysis.id == analysis_id).values(status=status))
        session.commit()


def test_the_active_limit_follows_the_analyses_status(
    sessions, redis_conn, api_key_for, scenario
):
    alice = _client(create_app(sessions, redis_conn, active(2)), api_key_for("alice"))
    submit = _submitter(alice, scenario)

    first, second, third = submit(), submit(), submit()

    assert (first.status_code, second.status_code) == (202, 202)
    assert third.status_code == 429
    assert third.headers["retry-after"] == "60"
    assert third.json() == {
        "detail": "This account already has 2 analyses waiting or running. "
        "Try again when one has finished."
    }
    # A running analysis is as active as a queued one.
    _finish(sessions, first.json()["id"], status="running")
    assert submit().status_code == 429
    _finish(sessions, first.json()["id"])
    assert submit().status_code == 202
    assert submit().status_code == 429
    _finish(sessions, second.json()["id"], status="failed")
    assert submit().status_code == 202


def test_the_active_limit_is_each_users_own(sessions, redis_conn, api_key_for, scenario):
    app = create_app(sessions, redis_conn, active(1))
    submit = _submitter(_client(app, api_key_for("alice")), scenario)
    submit_as_bob = _submitter(_client(app, api_key_for("bob")), scenario)
    submit_with_another_key = _submitter(_client(app, api_key_for("alice")), scenario)

    assert submit().status_code == 202
    assert submit_with_another_key().status_code == 429
    assert submit_as_bob().status_code == 202


def test_a_submission_refused_as_active_is_not_one_of_the_days(
    sessions, redis_conn, api_key_for, scenario
):
    alice = _client(create_app(sessions, redis_conn, active(1, per_day=2)), api_key_for("alice"))
    submit = _submitter(alice, scenario)

    first = submit()
    refusals = [submit().status_code for _ in range(3)]
    _finish(sessions, first.json()["id"])
    second = submit()
    _finish(sessions, second.json()["id"])

    assert refusals == [429, 429, 429]
    assert (first.status_code, second.status_code) == (202, 202)
    assert _analyses_refused(submit())


def test_the_eleventh_bad_key_from_one_address_is_refused(sessions, redis_conn, api_key_for):
    app = create_app(sessions, redis_conn, signins())
    key = api_key_for("alice")
    stranger = _client(app, "cpk_AAAAAAAAAAAA_wrong")

    codes = [stranger.get("/v1/me").status_code for _ in range(10)]
    eleventh = stranger.get("/v1/me")

    assert codes == [401] * 10
    assert _signin_refused(eleventh)
    # Another address is not held to this one's count, with a bad key or a good one.
    assert _client(app, "cpk_AAAAAAAAAAAA_wrong", peer=ELSEWHERE).get("/v1/me").status_code == 401
    assert _client(app, key, peer=ELSEWHERE).get("/v1/me").json() == {"name": "alice"}


def test_a_refused_address_is_refused_whatever_it_sends(sessions, redis_conn, api_key_for):
    app = create_app(sessions, redis_conn, signins(3))
    key = api_key_for("alice")
    stranger = _client(app)
    for _ in range(3):
        assert stranger.get("/v1/me").status_code == 401

    # Its key is not looked at: a good one is answered like a bad one...
    assert _signin_refused(_client(app, key).get("/v1/me"))
    assert _signin_refused(_upload(_client(app, key), _csv(0)))
    assert _signin_refused(stranger.get("/v1/no-such-route"))
    # ...and its use is not recorded.
    key_id, _ = parse_key(key)
    with sessions() as session:
        assert session.get(ApiKey, key_id).last_used_at is None
    # The health check takes no key and is never refused.
    assert stranger.get("/v1/healthz").json() == {"postgres": "ok", "redis": "ok"}


def test_being_refused_is_not_counted_and_does_not_lengthen_the_wait(
    sessions, redis_conn, api_key_for
):
    app = create_app(sessions, redis_conn, signins(3))
    stranger = _client(app)
    for _ in range(3):
        stranger.get("/v1/me")
    name = f"cp:limit:signin:{TESTCLIENT}"
    redis_conn.expire(name, 100)

    refusals = [stranger.get("/v1/me") for _ in range(5)]

    assert all(_signin_refused(response) for response in refusals)
    assert redis_conn.get(name) == b"3"
    assert 1 <= redis_conn.ttl(name) <= 100
    assert abs(int(refusals[-1].headers["retry-after"]) - redis_conn.ttl(name)) <= 1
    # The window ends, and with it the refusal.
    redis_conn.delete(name)
    assert stranger.get("/v1/me").status_code == 401


def test_every_kind_of_401_counts_and_a_success_does_not_reset_the_count(
    sessions, redis_conn, api_key_for
):
    app = create_app(sessions, redis_conn, signins(4))
    good, revoked = api_key_for("alice"), api_key_for("alice")
    with sessions() as session:
        revoke_key(session, parse_key(revoked)[0])
        session.commit()

    codes = [
        _client(app).get("/v1/me").status_code,  # no key
        _client(app, "nonsense").get("/v1/estimators").status_code,  # malformed
        _client(app, good).get("/v1/me").status_code,
        _client(app, revoked).get("/v1/datasets").status_code,
        _client(app, good).get("/v1/me").status_code,
        _client(app, "cpk_AAAAAAAAAAAA_wrong").post("/v1/analyses", json={}).status_code,
    ]

    assert codes == [401, 401, 200, 401, 200, 401]
    assert _signin_refused(_client(app, good).get("/v1/me"))


def test_answers_other_than_401_are_not_failed_sign_ins(sessions, redis_conn, api_key_for):
    app = create_app(sessions, redis_conn, signins(2))
    alice = _client(app, api_key_for("alice"))

    codes = [alice.get("/v1/datasets/not-a-uuid").status_code for _ in range(3)]
    codes += [alice.get("/v1/no-such-route").status_code for _ in range(3)]

    assert codes == [422] * 3 + [404] * 3
    assert alice.get("/v1/me").status_code == 200
    assert redis_conn.keys("cp:limit:*") == []


def test_the_client_address_header_is_ignored_outside_production(
    sessions, redis_conn, monkeypatch
):
    monkeypatch.delenv("CP_ENV", raising=False)
    app = create_app(sessions, redis_conn, signins(3))

    codes = [
        _client(app, headers={"X-CP-Client-Address": f"203.0.113.{n}"}).get("/v1/me").status_code
        for n in range(4)
    ]

    # Four claimed addresses, one connection, one count.
    assert codes == [401, 401, 401, 429]
    assert redis_conn.keys("cp:limit:*") == [f"cp:limit:signin:{TESTCLIENT}".encode()]


def test_in_production_the_count_is_kept_by_the_address_the_dashboard_passes_on(
    sessions, redis_conn, monkeypatch
):
    monkeypatch.setenv("CP_ENV", "production")
    app = create_app(sessions, redis_conn, signins(3))
    one = _client(app, headers={"X-CP-Client-Address": "203.0.113.1"})
    another = _client(app, headers={"X-CP-Client-Address": "203.0.113.2"})

    codes = [one.get("/v1/me").status_code for _ in range(4)]

    assert codes == [401, 401, 401, 429]
    assert another.get("/v1/me").status_code == 401
    assert sorted(redis_conn.keys("cp:limit:*")) == [
        b"cp:limit:signin:203.0.113.1", b"cp:limit:signin:203.0.113.2",
    ]  # fmt: skip


def test_the_limiters_keys_expire_and_sit_apart_from_the_queues(
    sessions, redis_conn, api_key_for, scenario
):
    app = create_app(
        sessions,
        redis_conn,
        Limits(
            signin_failures=Limit("signin_failures", 10, 900),
            uploads_per_day=Limit("uploads_per_day", 5, DAY),
            analyses_per_day=Limit("analyses_per_day", 5, DAY),
        ),
    )
    _client(app).get("/v1/me")
    assert _submitter(_client(app, api_key_for("alice")), scenario)().status_code == 202

    ours = sorted(name.decode() for name in redis_conn.keys("cp:limit:*"))

    assert [name.rsplit(":", 1)[0] for name in ours] == [
        "cp:limit:analyses", "cp:limit:signin", "cp:limit:uploads",
    ]  # fmt: skip
    ttls = {name.split(":")[2]: redis_conn.ttl(name) for name in ours}
    assert 890 <= ttls["signin"] <= 900
    assert DAY - 10 <= ttls["uploads"] <= DAY
    assert DAY - 10 <= ttls["analyses"] <= DAY
    assert not any(name.startswith(b"cp:limit:") for name in redis_conn.keys("rq:*"))


def test_two_requests_at_once_cannot_both_take_the_last_place(redis_conn):
    window = FixedWindow(redis_conn)
    limit = Limit("uploads_per_day", 5, DAY)

    with ThreadPoolExecutor(max_workers=16) as pool:
        decisions = list(pool.map(lambda _: window.hit("uploads:race", limit), range(64)))

    assert sum(decision.allowed for decision in decisions) == 5


def test_with_redis_unreachable_the_limits_allow(sessions, api_key_for):
    # A port nothing listens on: every command fails at once with a connection error.
    down = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=1)
    every = Limits(
        signin_failures=Limit("signin_failures", 1, 900),
        uploads_per_day=Limit("uploads_per_day", 1, DAY),
    )
    app = create_app(sessions, down, every)
    alice = _client(app, api_key_for("alice"))

    assert [_client(app).get("/v1/me").status_code for _ in range(3)] == [401, 401, 401]
    assert [_upload(alice, _csv(n)).status_code for n in range(3)] == [201, 201, 201]


def test_an_app_given_no_limits_reads_them_from_the_environment(
    sessions, redis_conn, api_key_for, monkeypatch
):
    monkeypatch.setenv("CP_LIMIT_UPLOADS_PER_DAY", "1")
    alice = _client(create_app(sessions, redis_conn), api_key_for("alice"))

    assert _upload(alice, _csv(0)).status_code == 201
    assert _upload_refused(_upload(alice, _csv(1)))


def test_with_no_limit_configured_nothing_is_counted(sessions, redis_conn, api_key_for):
    app = create_app(sessions, redis_conn, Limits())
    alice = _client(app, api_key_for("alice"))

    assert [_client(app).get("/v1/me").status_code for _ in range(12)] == [401] * 12
    assert [_upload(alice, _csv(n)).status_code for n in range(3)] == [201] * 3
    assert redis_conn.keys("cp:limit:*") == []
