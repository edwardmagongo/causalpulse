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


def test_a_schema_file_with_non_empty_covariates_loads_as_a_tuple(tmp_path):
    # JSON has no tuple type, so `covariates` arrives as a list; `CausalSchema`
    # is a frozen, eq=True dataclass declaring `covariates: tuple[str, ...]`,
    # so a list-typed value compares unequal to (and hashes differently from)
    # the tuple-typed schema an equivalent flag invocation builds. What
    # matters is that a file-loaded schema is interchangeable with a
    # flag-built one, so this asserts equality, not just field contents.
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps(
            {
                "unit": "state",
                "treatment": "treatment",
                "outcome": "cigsale",
                "covariates": ["x1", "x2"],
                "outcome_type": "continuous",
            }
        )
    )
    args = _parse(["--schema", str(path)])

    schema = schema_from_args(args)

    assert isinstance(schema.covariates, tuple)
    assert schema.covariates == ("x1", "x2")
    assert schema == CausalSchema(
        unit="state",
        treatment="treatment",
        outcome="cigsale",
        covariates=("x1", "x2"),
        outcome_type="continuous",
    )


def test_a_schema_file_with_non_empty_auxiliary_columns_loads_as_a_tuple(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps(
            {
                "unit": "state",
                "treatment": "treatment",
                "outcome": "cigsale",
                "covariates": [],
                "outcome_type": "continuous",
                "auxiliary_columns": ["a1", "a2"],
            }
        )
    )
    args = _parse(["--schema", str(path)])

    schema = schema_from_args(args)

    assert isinstance(schema.auxiliary_columns, tuple)
    assert schema.auxiliary_columns == ("a1", "a2")
    assert schema == CausalSchema(
        unit="state",
        treatment="treatment",
        outcome="cigsale",
        covariates=(),
        outcome_type="continuous",
        auxiliary_columns=("a1", "a2"),
    )


def test_a_schema_file_omitting_covariates_is_refused_naming_it(tmp_path):
    # `covariates` has no default on `CausalSchema`: silently defaulting it
    # would mean a schema file that omits it produces a schema with no
    # confounders at all, and the resulting estimates would answer a
    # question the user never asked.
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps(
            {
                "unit": "state",
                "treatment": "treatment",
                "outcome": "cigsale",
                "outcome_type": "continuous",
            }
        )
    )
    args = _parse(["--schema", str(path)])

    with pytest.raises(ValueError, match="covariates"):
        schema_from_args(args)


def test_a_schema_file_omitting_several_required_fields_names_them(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps({"unit": "state"}))
    args = _parse(["--schema", str(path)])

    with pytest.raises(ValueError) as exc_info:
        schema_from_args(args)

    message = str(exc_info.value)
    for missing_field in ("treatment", "outcome", "covariates", "outcome_type"):
        assert missing_field in message


def test_a_schema_file_with_an_unknown_key_is_refused_naming_it(tmp_path):
    # A typo'd key (e.g. "covariate" for "covariates") would otherwise be
    # silently ignored by CausalSchema's constructor semantics and produce
    # the same silent no-adjustment outcome as an omitted field.
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps(
            {
                "unit": "state",
                "treatment": "treatment",
                "outcome": "cigsale",
                "covariate": ["x1"],
                "outcome_type": "continuous",
            }
        )
    )
    args = _parse(["--schema", str(path)])

    with pytest.raises(ValueError, match="covariate"):
        schema_from_args(args)


def test_a_schema_file_omitting_only_auxiliary_columns_still_succeeds(tmp_path):
    # Unlike `covariates`, `auxiliary_columns` genuinely defaults to `()` on
    # `CausalSchema`, so omitting it from a schema file is not an error.
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps(
            {
                "unit": "state",
                "treatment": "treatment",
                "outcome": "cigsale",
                "covariates": ["x1"],
                "outcome_type": "continuous",
            }
        )
    )
    args = _parse(["--schema", str(path)])

    schema = schema_from_args(args)

    assert schema.auxiliary_columns == ()
