"""`POST /v1/datasets` checks the key and the limits before it reads the upload.

The first half runs the app under `uvicorn` on a real socket and sends a
request's headers with none of its body. A test client hands the app a whole
body at once, so it cannot tell a route that refuses first from one that reads
first; a server that is sent no body can only answer if it did not wait for
one. The control case is the same request with a valid key, which must get no
answer until its body arrives: without it these tests would pass against a
server that answers everything early.

The second half pins what the route answers for a request that is not a
well-formed upload, which FastAPI wrote before the route read the form itself.
"""

import http.client
import json
import socket
import threading
import time

import pytest
import uvicorn
from cp_api import app as app_module
from cp_api import validation
from cp_api.accounts import revoke_key
from cp_api.app import create_app
from cp_api.auth import UNAUTHORIZED, parse_key
from cp_api.db import Dataset, make_sessionmaker
from cp_api.limits import Limit, Limits
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from starlette import formparsers

pytestmark = pytest.mark.services

BOUNDARY = "cp-upload-order"
CONTENT_TYPE = f"multipart/form-data; boundary={BOUNDARY}"
# The body the headers promise and the refused requests never send.
PROMISED_BYTES = 12 * 1024 * 1024
# Long enough for a database and a Redis round trip on a loaded machine.
ANSWER_SECONDS = 5.0
# How long the control case is watched for an answer it must not get.
SILENCE_SECONDS = 2.0
MISSING_FILE = {
    "detail": [{"type": "missing", "loc": ["body", "file"], "msg": "Field required", "input": None}]
}


def _multipart(content, name="d.csv"):
    return (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode() + content + f"\r\n--{BOUNDARY}--\r\n".encode()


class Served:
    """The app under `uvicorn` on a port of its own, and raw connections to it."""

    def __init__(self, app):
        self._listening = socket.socket()
        self._listening.bind(("127.0.0.1", 0))
        self.port = self._listening.getsockname()[1]
        self._server = uvicorn.Server(
            uvicorn.Config(
                app, log_config=None, lifespan="off", ws="none", timeout_graceful_shutdown=1
            )
        )
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [self._listening]}, daemon=True
        )
        self._open = []

    def start(self):
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.01)

    def stop(self):
        for connection in self._open:
            connection.close()
        self._server.should_exit = True
        self._thread.join(timeout=10)
        self._listening.close()
        assert not self._thread.is_alive(), "uvicorn did not stop"

    def send_headers(self, key, length=PROMISED_BYTES):
        """Open a connection and send an upload's request line and headers, and no body."""
        connection = socket.create_connection(("127.0.0.1", self.port), timeout=ANSWER_SECONDS)
        self._open.append(connection)
        lines = [
            "POST /v1/datasets HTTP/1.1",
            f"Host: 127.0.0.1:{self.port}",
            f"Content-Type: {CONTENT_TYPE}",
            f"Content-Length: {length}",
        ]
        if key is not None:
            lines.append(f"Authorization: Bearer {key}")
        connection.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        return connection

    def upload(self, key, content):
        body = _multipart(content)
        connection = self.send_headers(key, len(body))
        connection.sendall(body)
        return _answer(connection)

    def get(self, key, path):
        connection = socket.create_connection(("127.0.0.1", self.port), timeout=ANSWER_SECONDS)
        self._open.append(connection)
        connection.sendall(
            f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n"
            f"Authorization: Bearer {key}\r\n\r\n".encode()
        )
        return _answer(connection)


def _answer(connection, within=ANSWER_SECONDS):
    """The response's status, headers and JSON body, or `TimeoutError` if none comes."""
    connection.settimeout(within)
    response = http.client.HTTPResponse(connection, method="POST")
    try:
        response.begin()
        headers = {name.lower(): value for name, value in response.getheaders()}
        return response.status, headers, json.loads(response.read())
    finally:
        response.close()


def _is_silent(connection):
    """Whether nothing at all, not even a close, arrives while the connection is watched."""
    connection.settimeout(SILENCE_SECONDS)
    try:
        connection.recv(1)
    except TimeoutError:
        return True
    return False


@pytest.fixture
def serve():
    running = []

    def start(app):
        served = Served(app)
        served.start()
        running.append(served)
        return served

    yield start
    for served in running:
        served.stop()


def _stored(sessions):
    with sessions() as session:
        return session.scalar(select(func.count()).select_from(Dataset))


def test_a_valid_key_gets_no_answer_until_the_body_arrives(
    sessions, redis_conn, api_key_for, serve
):
    # The control: the server does wait for a body it means to read.
    served = serve(create_app(sessions, redis_conn))
    body = _multipart(b"a,b\n1,2\n")

    connection = served.send_headers(api_key_for("alice"), len(body))

    assert _is_silent(connection)
    connection.sendall(body)
    status, _, payload = _answer(connection)
    assert (status, payload["columns"], payload["n_bytes"]) == (201, ["a", "b"], 8)


@pytest.mark.parametrize("credentials", [None, "not-a-key", "cpk_AAAAAAAAAAAA_wrongsecret"])
def test_a_missing_or_bad_key_is_401_with_the_body_unsent(
    sessions, redis_conn, serve, credentials
):
    served = serve(create_app(sessions, redis_conn))

    status, headers, payload = _answer(served.send_headers(credentials))

    assert (status, payload) == (401, {"detail": UNAUTHORIZED})
    assert headers["www-authenticate"] == "Bearer"


def test_a_revoked_key_is_401_with_the_body_unsent(sessions, redis_conn, api_key_for, serve):
    key = api_key_for("alice")
    served = serve(create_app(sessions, redis_conn))
    assert served.upload(key, b"a,b\n1,2\n")[0] == 201
    with sessions() as session:
        revoke_key(session, parse_key(key)[0])
        session.commit()

    status, _, payload = _answer(served.send_headers(key))

    assert (status, payload) == (401, {"detail": UNAUTHORIZED})


def test_a_user_over_the_upload_limit_is_429_with_the_body_unsent(
    sessions, redis_conn, api_key_for, serve
):
    key = api_key_for("alice")
    limits = Limits(uploads_per_day=Limit("uploads_per_day", 1, 86_400))
    served = serve(create_app(sessions, redis_conn, limits))
    assert served.upload(key, b"a,b\n1,2\n")[0] == 201

    status, headers, payload = _answer(served.send_headers(key))

    assert status == 429
    assert payload["detail"].startswith("Too many uploads today for this account.")
    assert 1 <= int(headers["retry-after"]) <= 86_400
    assert _stored(sessions) == 1


def test_an_address_over_the_failed_sign_in_limit_is_429_with_the_body_unsent(
    sessions, redis_conn, api_key_for, serve
):
    limits = Limits(signin_failures=Limit("signin_failures", 2, 900))
    served = serve(create_app(sessions, redis_conn, limits))
    # Failed on a route with no body, so this test is about the refusal alone.
    assert [served.get("not-a-key", "/v1/me")[0] for _ in range(2)] == [401, 401]

    # A valid key, which is not looked at.
    status, headers, payload = _answer(served.send_headers(api_key_for("alice")))

    assert status == 429
    assert payload["detail"].startswith("Too many failed sign-ins from this address.")
    assert 1 <= int(headers["retry-after"]) <= 900


def test_an_upload_being_sent_holds_no_database_connection(
    migrated, api_key_for, redis_conn, serve
):
    # One connection in the pool and no waiting for another: if the upload kept
    # the one its key was checked on while its body arrived, the second request
    # would find none.
    key = api_key_for("alice")
    engine = create_engine(migrated, pool_size=1, max_overflow=0, pool_timeout=1)
    served = serve(create_app(make_sessionmaker(engine), redis_conn))
    body = _multipart(b"a,b\n1,2\n")
    try:
        uploading = served.send_headers(key, len(body))
        assert _is_silent(uploading)

        status, _, payload = served.get(key, "/v1/me")
        assert (status, payload) == (200, {"name": "alice"})
        uploading.sendall(body)
        assert _answer(uploading)[0] == 201
    finally:
        engine.dispose()


def test_a_body_far_over_the_limit_is_413_and_is_not_kept(
    sessions, redis_conn, api_key_for, serve, monkeypatch
):
    # What the form parser writes to its temporary files is counted, since
    # those are what an unbounded body would fill the disk with.
    monkeypatch.setattr(validation, "MAX_UPLOAD_BYTES", 100_000)
    spooled = []

    class Counted(formparsers.SpooledTemporaryFile):
        def write(self, data):
            spooled.append(len(data))
            return super().write(data)

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", Counted)
    served = serve(create_app(sessions, redis_conn))
    sent = 8 * 1024 * 1024

    status, _, payload = served.upload(api_key_for("alice"), b"a" * sent)

    assert (status, payload) == (413, {"detail": "the upload exceeds the 100000-byte limit"})
    assert sum(spooled) < sent
    assert sum(spooled) <= 100_000 + app_module.FORM_OVERHEAD_BYTES < sent
    assert _stored(sessions) == 0


def test_a_file_one_byte_over_the_limit_is_413_and_one_at_it_is_stored(
    sessions, redis_conn, api_key_for, serve, monkeypatch
):
    monkeypatch.setattr(validation, "MAX_UPLOAD_BYTES", 100_000)
    served = serve(create_app(sessions, redis_conn))
    key = api_key_for("alice")
    at_the_limit = b"a,b\n" + b"1,2\n" * 24_999

    over = served.upload(key, at_the_limit + b"3")
    at = served.upload(key, at_the_limit)

    assert (over[0], over[2]) == (413, {"detail": "the upload exceeds the 100000-byte limit"})
    assert (at[0], at[2]["n_bytes"]) == (201, 100_000)


@pytest.fixture
def alice(sessions, redis_conn, api_key_for):
    return TestClient(
        create_app(sessions=sessions, redis=redis_conn),
        headers={"Authorization": f"Bearer {api_key_for('alice')}"},
    )


def _not_a_file(value):
    return {
        "detail": [
            {
                "type": "value_error",
                "loc": ["body", "file"],
                "msg": "Value error, Expected UploadFile, received: <class 'str'>",
                "input": value,
                "ctx": {"error": {}},
            }
        ]
    }


def _answered(response):
    return response.status_code, response.json()


def test_a_request_with_no_file_field_is_422(alice):
    csv = ("d.csv", b"a,b\n1,2\n", "text/csv")

    assert _answered(alice.post("/v1/datasets")) == (422, MISSING_FILE)
    assert _answered(alice.post("/v1/datasets", files={"other": csv})) == (422, MISSING_FILE)
    assert _answered(alice.post("/v1/datasets", data={"file": ""})) == (422, MISSING_FILE)


def test_a_request_that_is_not_a_form_is_422(alice):
    as_json = alice.post("/v1/datasets", json={"file": "a,b\n1,2\n"})
    as_csv = alice.post(
        "/v1/datasets", content=b"a,b\n1,2\n", headers={"Content-Type": "text/csv"}
    )

    assert _answered(as_json) == (422, MISSING_FILE)
    assert _answered(as_csv) == (422, MISSING_FILE)


def test_a_file_field_that_is_text_is_422(alice):
    in_multipart = alice.post(
        "/v1/datasets", data={"file": "a,b"}, files={"other": ("d.csv", b"a,b\n1,2\n")}
    )
    urlencoded = alice.post("/v1/datasets", data={"file": "a,b"})

    assert _answered(in_multipart) == (422, _not_a_file("a,b"))
    assert _answered(urlencoded) == (422, _not_a_file("a,b"))


def test_a_form_that_cannot_be_parsed_is_400(alice):
    def post(content, content_type):
        return alice.post("/v1/datasets", content=content, headers={"Content-Type": content_type})

    assert _answered(post(b"xx", "multipart/form-data")) == (
        400,
        {"detail": "Missing boundary in multipart."},
    )
    assert _answered(post(b"xx", CONTENT_TYPE)) == (400, {"detail": "Invalid multipart data."})
    # A part that is not a file may be 1 MB and no more.
    oversized_field = alice.post(
        "/v1/datasets",
        data={"note": "x" * (1024 * 1024 + 1)},
        files={"file": ("d.csv", b"a,b\n1,2\n")},
    )
    assert _answered(oversized_field) == (400, {"detail": "Part exceeded maximum size of 1024KB."})


def test_the_last_of_two_file_fields_is_the_upload(alice):
    response = alice.post(
        "/v1/datasets",
        files=[("file", ("first.csv", b"a,b\n1,2\n")), ("file", ("second.csv", b"c,d\n3,4\n"))],
    )

    assert response.status_code == 201
    assert (response.json()["filename"], response.json()["columns"]) == ("second.csv", ["c", "d"])

