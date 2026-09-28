import hashlib

import pytest
from cp_worker.execute import execute

from worker.data import scenario_d_upload


def test_execute_names_the_upload_by_its_own_hash():
    content, schema = scenario_d_upload()

    outcome = execute(content, "d.csv", schema, ["difference_in_means", "lin_regression"], 0)

    assert outcome.manifest.dataset.content_sha256 == hashlib.sha256(content).hexdigest()
    assert outcome.manifest.dataset.path == "d.csv"
    assert [r.estimate.method for r in outcome.reports] == [
        "difference_in_means", "lin_regression"
    ]


def test_an_ineligible_request_raises_the_routers_reasons_verbatim():
    content, schema = scenario_d_upload()

    with pytest.raises(ValueError, match="synthetic_control: requires panel data"):
        execute(content, "d.csv", schema, ["synthetic_control"], 0)
