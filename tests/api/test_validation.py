import pytest
from cp_api import validation
from cp_api.validation import (
    RequestError,
    check_analysis_request,
    check_upload,
    csv_header,
    safe_filename,
)

CSV = b"unit_id,treated,outcome,x1,x2\n1,1,0.5,0.1,0.2\n2,0,0.4,0.3,0.1\n"
SCHEMA = {
    "unit": "unit_id", "treatment": "treated", "outcome": "outcome",
    "covariates": ["x1", "x2"], "outcome_type": "continuous",
}
HEADER = ("unit_id", "treated", "outcome", "x1", "x2")


def test_a_valid_upload_returns_its_columns():
    assert check_upload(CSV) == HEADER
    assert csv_header(CSV) == HEADER


def test_an_upload_over_the_limit_is_413(monkeypatch):
    monkeypatch.setattr(validation, "MAX_UPLOAD_BYTES", 10)

    with pytest.raises(RequestError) as caught:
        check_upload(CSV)

    assert caught.value.status_code == 413


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        (b"", "empty"),
        (b'a,b\n"1,2\n', "could not be read as CSV"),
        (b"a,b\n", "no rows"),
    ],
)
def test_unreadable_uploads_are_422(content, fragment):
    with pytest.raises(RequestError, match=fragment) as caught:
        check_upload(content)

    assert caught.value.status_code == 422


def test_a_valid_request_returns_the_parsed_schema():
    schema = check_analysis_request(SCHEMA, HEADER, None)

    assert schema.covariates == ("x1", "x2")


def test_a_column_the_dataset_lacks_is_named():
    with pytest.raises(RequestError, match=r"\['x3'\]") as caught:
        check_analysis_request({**SCHEMA, "covariates": ["x1", "x3"]}, HEADER, None)

    assert caught.value.status_code == 422


def test_an_auxiliary_column_the_dataset_lacks_is_named():
    with pytest.raises(RequestError, match=r"\['z'\]"):
        check_analysis_request({**SCHEMA, "auxiliary_columns": ["z"]}, HEADER, None)


def test_a_malformed_schema_is_422_with_the_parsers_message():
    with pytest.raises(RequestError, match="unknown schema field") as caught:
        check_analysis_request({**SCHEMA, "colour": "red"}, HEADER, None)

    assert caught.value.status_code == 422


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"outcome": "treated"}, "treatment and outcome both name column 'treated'"),
        ({"covariates": ["x1", "outcome"]}, "covariates contains 'outcome'"),
    ],
)
def test_a_schema_whose_roles_collide_is_422_with_the_schemas_reason(overrides, fragment):
    # Every column named is in the upload, so the missing-column check cannot
    # be what refuses this: without the schema's own refusal it is queued.
    with pytest.raises(RequestError, match=fragment) as caught:
        check_analysis_request({**SCHEMA, **overrides}, HEADER, None)

    assert caught.value.status_code == 422
    assert caught.value.message.startswith("invalid schema: ")


def test_an_empty_estimator_list_is_refused():
    with pytest.raises(RequestError, match="at least one"):
        check_analysis_request(SCHEMA, HEADER, [])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("lalonde.csv", "lalonde.csv"),
        ("dir/sub/data.csv", "data.csv"),
        ("../../etc/passwd", "passwd"),
        ("C:\\x\\y.csv", "y.csv"),
        ("a/b\\c.csv", "c.csv"),
        ("..", "upload.csv"),
        (".", "upload.csv"),
        ("/", "upload.csv"),
        ("dir/..", "upload.csv"),
        ("", "upload.csv"),
        (None, "upload.csv"),
    ],
)
def test_an_uploaded_filename_is_reduced_to_a_bare_name(raw, expected):
    assert safe_filename(raw) == expected


def test_a_long_uploaded_filename_is_cut_to_the_column_width():
    assert safe_filename("x" * 300 + ".csv") == "x" * 255


@pytest.mark.parametrize(
    "content",
    [
        b"unit_id,treated,outcome\n1,1,0.5,9\n2,0,0.4,9\n",
        b"unit_id,treated,outcome\n1,1,0.5,\n2,0,0.4,\n",
    ],
    ids=["an extra field holding data", "a comma ending every line"],
)
def test_an_upload_with_a_row_wider_than_its_header_is_422(content):
    # Before the shared reader both were accepted, with every column shifted.
    with pytest.raises(RequestError) as caught:
        check_upload(content)

    assert caught.value.status_code == 422
    assert "line 2 of the CSV has 4 fields and its header names 3" in caught.value.message
