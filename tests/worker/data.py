"""A small real upload -- CSV bytes and the JSON schema a client would send."""

import dataclasses
from typing import Any

import numpy as np
from cp_simulation.params import ScenarioParams
from cp_simulation.scenarios import build_scenario


def scenario_d_upload() -> tuple[bytes, dict[str, Any]]:
    dataset = build_scenario("D", ScenarioParams()).generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset
    schema = dataclasses.asdict(dataset.schema)
    schema["covariates"] = list(schema["covariates"])
    schema["auxiliary_columns"] = list(schema["auxiliary_columns"])
    return dataset.frame.to_csv(index=False).encode(), schema
