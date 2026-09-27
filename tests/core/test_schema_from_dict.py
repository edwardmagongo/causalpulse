import pytest
from cp_core.dataset import CausalSchema, schema_from_dict

MINIMAL = {
    "unit": "unit_id",
    "treatment": "treated",
    "outcome": "outcome",
    "covariates": ["x1", "x2"],
    "outcome_type": "binary",
}


def test_a_minimal_json_schema_parses_with_tuples_where_the_dataclass_has_them():
    schema = schema_from_dict(MINIMAL)

    assert schema == CausalSchema(
        unit="unit_id", treatment="treated", outcome="outcome",
        covariates=("x1", "x2"), outcome_type="binary",
    )
    assert schema.auxiliary_columns == ()


def test_an_unknown_field_is_named():
    with pytest.raises(ValueError, match=r"unknown schema field\(s\): \['colour'\]"):
        schema_from_dict({**MINIMAL, "colour": "red"})


def test_a_missing_required_field_is_named_rather_than_defaulted():
    data = {k: v for k, v in MINIMAL.items() if k != "covariates"}

    with pytest.raises(ValueError, match=r"missing required field\(s\): \['covariates'\]"):
        schema_from_dict(data)


def test_covariates_given_as_one_string_is_refused_not_split_into_characters():
    with pytest.raises(ValueError, match="list of column names"):
        schema_from_dict({**MINIMAL, "covariates": "x1"})


def test_optional_fields_pass_through():
    schema = schema_from_dict(
        {**MINIMAL, "assignment_provenance": "randomised", "auxiliary_columns": ["z"]}
    )

    assert schema.assignment_provenance == "randomised"
    assert schema.auxiliary_columns == ("z",)
