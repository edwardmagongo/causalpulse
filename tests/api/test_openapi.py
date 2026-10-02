"""`web/openapi.json` is the API's current OpenAPI document, byte for byte.

The dashboard's types are generated from the committed file. If a route or a
response model changes and the file does not, this fails; regenerate it with
`make openapi`.
"""

from pathlib import Path

from cp_api.openapi import document, render

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
