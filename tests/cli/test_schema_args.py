import argparse
import json

import pytest
from cp_cli.schema_args import add_schema_arguments, schema_from_args
from cp_core.dataset import CausalSchema


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_schema_arguments(parser)
    return parser.parse_args(argv)


def test_the_five_required_fields_build_a_schema():
    args = _parse(
        [
            "--unit", "unit_id",
            "--treatment", "treat",
            "--outcome", "y",
            "--covariates", "x1,x2",
            "--outcome-type", "continuous",
        ]
    )

    schema = schema_from_args(args)

    assert schema == CausalSchema(
        unit="unit_id",
        treatment="treat",
        outcome="y",
        covariates=("x1", "x2"),
        outcome_type="continuous",
    )


def test_covariates_may_be_empty():
    args = _parse(
        [
            "--unit", "unit_id",
            "--treatment", "treat",
            "--outcome", "y",
            "--covariates", "",
            "--outcome-type", "continuous",
        ]
    )

    assert schema_from_args(args).covariates == ()


def test_a_schema_file_carries_the_optional_fields(tmp_path):
    # `time` has no flag: the optional CausalSchema fields go through a file
    # rather than growing ten more flags nobody would discover.
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps(
            {
                "unit": "state",
                "treatment": "treatment",
                "outcome": "cigsale",
                "covariates": [],
                "outcome_type": "continuous",
                "time": "year",
            }
        )
    )
    args = _parse(["--schema", str(path)])

    schema = schema_from_args(args)

    assert schema.time == "year"
    assert schema.unit == "state"


def test_giving_both_a_schema_file_and_flags_is_refused():
    # Silently preferring one would mean a user's flags are ignored without
    # a word, and they would read the resulting estimates as answering the
    # question they thought they asked.
    args = _parse(["--schema", "s.json", "--unit", "unit_id"])

    with pytest.raises(ValueError, match="both"):
        schema_from_args(args)


def test_giving_neither_is_refused():
    args = _parse([])

    with pytest.raises(ValueError, match="--schema"):
        schema_from_args(args)


def test_a_partial_flag_set_is_refused_naming_what_is_missing():
    args = _parse(["--unit", "unit_id", "--treatment", "treat"])

    with pytest.raises(ValueError, match="outcome"):
        schema_from_args(args)


def test_an_unknown_outcome_type_is_refused_by_causal_schema():
    args = _parse(
        [
            "--unit", "unit_id",
            "--treatment", "treat",
            "--outcome", "y",
            "--covariates", "",
            "--outcome-type", "ordinal",
        ]
    )

    with pytest.raises(ValueError, match="outcome_type"):
        schema_from_args(args)
