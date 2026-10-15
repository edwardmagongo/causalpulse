"""The response models describe the payloads; they must not reshape them.

A response model drops any key it does not declare, without an error. So a
key the core adds to a stored payload would vanish from the API's answer the
day it was added. These tests pin each model to what the core writes.
"""

import pytest
from cp_api.app import create_app
from cp_api.models import ComparisonGroup, EstimateSummary
from cp_core.results import estimate_summary
from cp_worker.execute import execute
from worker.data import scenario_d_upload

PAIR = ["difference_in_means", "lin_regression"]


@pytest.fixture(scope="module")
def compared():
    content, schema = scenario_d_upload()
    return execute(content, "d.csv", schema, PAIR, 0)


@pytest.fixture(scope="module")
def alone():
    content, schema = scenario_d_upload()
    return execute(content, "d.csv", schema, PAIR[:1], 0)


def test_the_summary_model_declares_exactly_the_keys_the_core_writes(compared):
    summary = estimate_summary(compared.reports[0].estimate)

    assert set(EstimateSummary.model_fields) == set(summary)


def test_a_summary_passes_through_the_model_unchanged(compared):
    for report in compared.reports:
        summary = estimate_summary(report.estimate)

        # Exact equality: replay parity compares these floats bit for bit.
        assert EstimateSummary.model_validate(summary).model_dump() == summary


def test_a_summary_stored_before_the_interval_level_was_kept_still_validates(compared):
    # Analyses stored before `bootstrap_ci_level` existed have summaries without it.
    summary = estimate_summary(compared.reports[0].estimate)
    del summary["bootstrap_ci_level"]

    assert EstimateSummary.model_validate(summary).bootstrap_ci_level is None


def test_a_group_answers_with_exactly_the_keys_it_was_stored_with(compared, alone):
    stored = [group.to_dict() for group in (*compared.comparisons, *alone.comparisons)]
    # Both shapes, or the loop below would prove only one of them.
    assert ["verdict" in group for group in stored] == [True, False]

    for group in stored:
        assert set(group) <= set(ComparisonGroup.model_fields)
        assert ComparisonGroup.model_validate(group).model_dump(exclude_unset=True) == group


def test_the_json_routes_that_predate_the_models_declare_their_responses():
    paths = create_app(sessions=object(), redis=object()).openapi()["paths"]
    declared = {
        ("post", "/v1/datasets"): "201",
        ("post", "/v1/analyses"): "202",
        ("get", "/v1/analyses/{analysis_id}"): "200",
        ("get", "/v1/analyses/{analysis_id}/summary"): "200",
    }

    for (method, path), status in declared.items():
        schema = paths[path][method]["responses"][status]["content"]["application/json"]["schema"]
        assert "$ref" in schema, (method, path)


def test_the_database_and_the_api_list_the_same_interpretation_statuses():
    import typing

    from cp_api.db import INTERPRETATION_STATUSES
    from cp_api.models import InterpretationStatus

    assert set(INTERPRETATION_STATUSES) == set(typing.get_args(InterpretationStatus))


def _row(status, version):
    from cp_api.db import Interpretation

    return Interpretation(
        status=status,
        job_id="interpretation-x",
        model="claude-opus-5-5",
        effort="medium",
        interpreter_version=version,
        rendered={
            "sections": [{"name": "summary", "heading": "Summary", "statements": ["A."]}],
            "limits": "L.",
        },
        notice="N.",
        attempts=[{"violations": [{"rule": "bare_value"}]}],
        error="E.",
    )


@pytest.mark.parametrize("status", ["queued", "running", "verified", "refused", "failed"])
def test_an_interpretation_passes_through_its_model_unchanged(status):
    from cp_api.app import CURRENT_INTERPRETER_VERSION, _interpretation_out
    from cp_api.models import InterpretationDetail

    for version in (CURRENT_INTERPRETER_VERSION, "0123456789ab", None):
        served = _interpretation_out(_row(status, version))

        assert set(served) == set(InterpretationDetail.model_fields)
        assert InterpretationDetail.model_validate(served).model_dump() == served
        assert (served["interpreter_version"], served["out_of_date"]) == (
            version, version != CURRENT_INTERPRETER_VERSION,
        )


def test_the_analysis_declares_its_earlier_interpretations_as_a_list_never_null():
    from cp_api.models import AnalysisDetail, InterpretationDetail

    field = AnalysisDetail.model_fields["earlier_interpretations"]
    assert field.annotation == list[InterpretationDetail]
    assert field.is_required()
