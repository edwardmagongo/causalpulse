"""`web/openapi.json` is the API's current OpenAPI document, byte for byte.

The dashboard's types are generated from the committed file. If a route or a
response model changes and the file does not, this fails; regenerate it with
`make openapi`.
"""

from pathlib import Path
from typing import Annotated

from cp_api.app import create_app, upload_allowance
from cp_api.auth import current_user
from cp_api.db import User
from cp_api.openapi import document, render
from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient
from redis import Redis
from sqlalchemy.orm import sessionmaker

COMMITTED = Path(__file__).resolve().parents[2] / "web" / "openapi.json"


def test_the_committed_document_is_current():
    assert COMMITTED.read_text() == render(), "run `make openapi` and commit web/openapi.json"


def test_every_json_route_the_dashboard_reads_declares_its_response():
    # Without a response model a route's 200 is an empty schema, and the
    # generated type for it is `unknown`.
    paths = document()["paths"]
    typed = {
        ("get", "/v1/me"),
        ("get", "/v1/estimators"),
        ("get", "/v1/datasets"),
        ("get", "/v1/datasets/{dataset_id}"),
        ("post", "/v1/datasets"),
        ("get", "/v1/analyses"),
        ("post", "/v1/analyses"),
        ("get", "/v1/analyses/{analysis_id}"),
        ("get", "/v1/analyses/{analysis_id}/summary"),
    }
    for method, path in sorted(typed):
        responses = paths[path][method]["responses"]
        success = next(responses[code] for code in ("200", "201", "202") if code in responses)
        assert "$ref" in success["content"]["application/json"]["schema"], (method, path)


# --- the upload's body is still what FastAPI would have written ----------------
#
# `POST /v1/datasets` reads its own form, so two things FastAPI wrote while the
# route declared `file: UploadFile` are now copies kept in `cp_api.app`: the
# request body in the OpenAPI document, and the 422 for a form without the
# file. `declared` is a route that still declares the field, on the same path
# and under the same name; the copies are compared with what FastAPI makes of
# it, so an upgrade that rewords either fails here.


def declared() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/datasets", status_code=201)
    def upload_dataset(file: Annotated[UploadFile, File()]) -> None:
        raise AssertionError("no request here carries a file")

    return app


def _request_body(document_: dict) -> dict:
    """The upload's request body with its schema in place of the reference to it.

    Without the schema's title, which is the component's name: FastAPI makes
    the name up from the route's, and nothing reads it but the reference.
    """
    body = document_["paths"]["/v1/datasets"]["post"]["requestBody"]
    ((media_type, media),) = body["content"].items()
    name = media["schema"]["$ref"].removeprefix("#/components/schemas/")
    schema = dict(document_["components"]["schemas"][name])
    assert schema.pop("title") == name
    return {**body, "content": {media_type: {**media, "schema": schema}}}


def test_the_uploads_body_and_its_422_are_what_fastapi_writes_for_a_declared_file():
    assert _request_body(document()) == _request_body(declared().openapi())

    app = create_app(sessions=sessionmaker(), redis=Redis())
    # The key and the day's limit are a database's and a Redis's to check, and
    # neither is what is compared here.
    app.dependency_overrides[current_user] = lambda: User(name="alice")
    app.dependency_overrides[upload_allowance] = lambda: User(name="alice")
    ours, theirs = TestClient(app), TestClient(declared())
    csv = ("d.csv", b"a,b\n1,2\n", "text/csv")
    without_a_file = [
        {},
        {"files": {"other": csv}},
        {"data": {"file": ""}},
        {"json": {"file": "a,b"}},
        # A `file` that is text and not a file: the other 422 the route writes itself.
        {"data": {"file": "a,b"}},
        {"data": {"file": "a,b"}, "files": {"other": csv}},
    ]
    for sent in without_a_file:
        expected = theirs.post("/v1/datasets", **sent)
        answered = ours.post("/v1/datasets", **sent)
        assert expected.status_code == 422, sent
        assert (answered.status_code, answered.json()) == (422, expected.json()), sent
