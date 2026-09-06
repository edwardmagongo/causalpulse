"""Constructing a `CausalSchema` from command-line arguments.

Two ways in, and they may not be mixed. The five required fields have flags,
because naming a unit and an outcome column is the common case and should not
need a file. Everything else -- `time`, `cohort`, `treatment_start`,
`assignment_provenance`, `negative_control_outcome`, `auxiliary_columns` --
goes through `--schema`, reusing the JSON shape `AnalysisManifest` already
serialises rather than inventing a second format or growing ten more flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cp_core.dataset import CausalSchema

_FLAG_FIELDS = ("unit", "treatment", "outcome", "covariates", "outcome_type")


def add_schema_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--schema", help="Path to a JSON schema, for the optional fields.")
    parser.add_argument("--unit", help="Unit identifier column.")
    parser.add_argument("--treatment", help="Treatment column.")
    parser.add_argument("--outcome", help="Outcome column.")
    parser.add_argument("--covariates", help="Comma-separated covariate columns.")
    parser.add_argument("--outcome-type", help="One of: binary, continuous.")


def _tuple_fields(data: dict[str, Any]) -> dict[str, Any]:
    """JSON has no tuple type, so every tuple-typed field arrives as a list.

    `covariates` and `auxiliary_columns` are `CausalSchema`'s only two such
    fields as of this phase; a third would need the same treatment.
    """
    data["covariates"] = tuple(data.get("covariates", ()))
    data["auxiliary_columns"] = tuple(data.get("auxiliary_columns", ()))
    return data


def schema_from_args(args: argparse.Namespace) -> CausalSchema:
    given_flags = [f for f in _FLAG_FIELDS if getattr(args, f, None) is not None]
    schema_file = getattr(args, "schema", None)

    if schema_file is not None and given_flags:
        flag_names = sorted("--" + f.replace("_", "-") for f in given_flags)
        raise ValueError(
            f"both --schema and {flag_names} were given; pass one or the "
            "other. Preferring either silently would ignore what you asked "
            "for without saying so."
        )
    if schema_file is not None:
        data = json.loads(Path(schema_file).read_text())
        return CausalSchema(**_tuple_fields(dict(data)))
    if not given_flags:
        raise ValueError(
            "a schema is required for a CSV dataset: pass --schema, or all of "
            "--unit --treatment --outcome --covariates --outcome-type"
        )

    missing = [f for f in _FLAG_FIELDS if getattr(args, f, None) is None]
    if missing:
        missing_names = sorted("--" + f.replace("_", "-") for f in missing)
        raise ValueError(f"schema flags are incomplete; missing {missing_names}")

    covariates = tuple(c for c in str(args.covariates).split(",") if c)
    return CausalSchema(
        unit=str(args.unit),
        treatment=str(args.treatment),
        outcome=str(args.outcome),
        covariates=covariates,
        outcome_type=args.outcome_type,
    )
