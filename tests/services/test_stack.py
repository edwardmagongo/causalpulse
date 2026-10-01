"""The whole compose stack, over HTTP, on the real LaLonde fixture."""

import os
import time
from pathlib import Path

import httpx
import pytest

pytestmark = [pytest.mark.services, pytest.mark.slow]

LALONDE = Path(__file__).parents[1] / "fixtures" / "real_data" / "lalonde.csv"
SCHEMA = {
    "unit": "unit_id",
    "treatment": "treat",
    "outcome": "re78",
    "covariates": ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"],
    "outcome_type": "continuous",
}


def _wait(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(1)
    raise AssertionError(f"timed out after {timeout}s")


def test_upload_lalonde_and_fetch_a_report():
    base = os.environ.get("CP_STACK_URL")
    if not base:
        pytest.skip("CP_STACK_URL is not set; run `make smoke`")
    key = os.environ.get("CP_STACK_KEY")
    if not key:
        pytest.skip("CP_STACK_KEY is not set; run `make smoke`")
    with httpx.Client(
        base_url=base, timeout=30, headers={"Authorization": f"Bearer {key}"}
    ) as client:

        def healthy():
            try:
                return client.get("/v1/healthz").status_code == 200
            except httpx.TransportError:
                return False

        _wait(healthy, 180)
        dataset = client.post(
            "/v1/datasets", files={"file": ("lalonde.csv", LALONDE.read_bytes(), "text/csv")}
        ).json()
        analysis = client.post(
            "/v1/analyses",
            json={
                "dataset_id": dataset["id"],
                "schema": SCHEMA,
                "estimators": ["difference_in_means", "lin_regression"],
            },
        ).json()

        def finished():
            body = client.get(f"/v1/analyses/{analysis['id']}").json()
            return body if body["status"] in ("succeeded", "failed") else None

        body = _wait(finished, 300)
        assert body["status"] == "succeeded", body["error"]
        report = client.get(
            f"/v1/analyses/{analysis['id']}/estimates/lin_regression/report.html"
        )
        assert report.status_code == 200
        anonymous = httpx.get(f"{base}/v1/analyses/{analysis['id']}", timeout=30)
        assert anonymous.status_code == 401
