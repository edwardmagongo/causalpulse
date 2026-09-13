"""`ScenarioParams` and the widened `ScenarioSpec.build`.

`build_scenario` keeps its pre-Phase-18 signature -- a positional/keyword
`confounding_strength: float` -- because `cp_evaluation.resolve.resolve_cell`
and `cp_cli`'s `simulate`/`analyze` commands call it that way, and this task
does not touch those packages. Internally it now builds a `ScenarioParams`
and hands it to the widened registry callable.

The equivalence test below is Phase 18's baseline: every later task threading
an axis into a builder must leave a scenario's *default* output untouched, and
this is the test that would catch a builder quietly reading the wrong field or
applying an axis that should default to a no-op.
"""

import dataclasses
import hashlib
import json

import numpy as np
import pytest
from cp_simulation.params import AXIS_NAMES, ScenarioParams
from cp_simulation.scenarios import SCENARIOS, build_scenario


def test_params_default_to_todays_behaviour():
    params = ScenarioParams()

    assert params.confounding_strength == 1.0
    assert params.treatment_intercept == 0.0
    assert params.noise_scale == 0.0
    assert params.effect_scale == 1.0
    assert params.heterogeneity_scale == 1.0


def test_params_are_frozen():
    params = ScenarioParams()

    with pytest.raises(dataclasses.FrozenInstanceError):
        params.confounding_strength = 2.0


def test_axis_names_matches_the_dataclass_fields():
    # The only way this module can lie is AXIS_NAMES drifting from the
    # dataclass it describes -- close that off directly rather than trusting
    # the two lists to stay in sync by hand.
    assert tuple(f.name for f in dataclasses.fields(ScenarioParams)) == AXIS_NAMES


@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_every_scenario_builds_from_params(scenario_id):
    # This is the widened contract itself -- ScenarioSpec.build:
    # Callable[[ScenarioParams], CrossSectionDGP] -- exercised directly rather
    # than through build_scenario's float-compatible facade, which is tested
    # separately below.
    dgp = SCENARIOS[scenario_id].build(ScenarioParams())

    assert dgp.scenario_id == scenario_id


def _digest_simulated(simulated):
    """A stable fingerprint of a generated frame and its ground truth.

    Content, not identity: two independently generated but numerically equal
    runs digest the same. Anything that changes a single float anywhere in the
    frame or the truth arrays changes this digest.
    """
    frame_bytes = simulated.dataset.frame.to_csv(index=True).encode()
    truth = simulated.truth
    payload = {
        "true_ate": truth.true_ate,
        "true_att": truth.true_att,
        "true_cate": truth.true_cate.tolist(),
        "true_propensity": truth.true_propensity.tolist(),
        "true_confounders": list(truth.true_confounders),
        "scenario_id": truth.scenario_id,
        "config_hash": truth.config_hash,
        "true_treatment": (
            None if truth.true_treatment is None else truth.true_treatment.tolist()
        ),
        "complete_outcome": (
            None if truth.complete_outcome is None else truth.complete_outcome.tolist()
        ),
        "schema": repr(simulated.dataset.schema),
    }
    digest = hashlib.sha256()
    digest.update(frame_bytes)
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode())
    return digest.hexdigest()


# Captured from the pre-Phase-18 code -- `build_scenario`'s builders taking a
# bare positional confounding strength, before `ScenarioParams` existed -- at
# n_units=2000, seed=0, for every registered scenario. Not derived from the
# code this task ships: if this task's refactor changed what a single one of
# these scenarios generates, every digest below would already have been wrong
# before this test was ever run against the new code.
_PRE_PHASE18_DIGESTS = {
    "A": "96efbf0148361e7776e3e0222432a79e3b37197c38490c68b95e142196ae3b91",
    "B": "e9b86aea995effb4506f7d48f20d1855d493b9ea3972c9ded4ecafa4b5f348d6",
    "C": "d3ec44fd852688c8a330f15d9c1b3371c821688c95e35d350b9baf9cbe7be61d",
    "D": "4fcde1cbb65c3bfac4fd8b74f8ac2c035d66921610af31f39c511ea65cf0a54c",
    "F": "502998f5ddcc2ce437a11092b96f83c0e8b98b5e61519a74eb800a5e96bb9f8d",
    "G": "ff2bd6376127f7f4a353d54c5d257655a9923930c672c688e1096d3bd36dabe9",
    "H": "3044a00127222f806244a70ab2f7faaf7dce163fa878d3e7934339c08833fb19",
}


@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_scenario_output_is_byte_identical_to_pre_phase18(scenario_id):
    # Through the real public entry point every existing caller uses, at its
    # default confounding strength -- not the registry callable directly --
    # so this pins the thing production code and the rest of the suite
    # actually depend on.
    simulated = build_scenario(scenario_id).generate(n_units=2000, rng=np.random.default_rng(0))

    assert _digest_simulated(simulated) == _PRE_PHASE18_DIGESTS[scenario_id], (
        f"scenario {scenario_id} no longer generates the data it did before Phase 18. "
        "These digests were captured from pre-refactor code, so a mismatch means a "
        "change reached the DEFAULT axis values -- most likely an rng draw consumed "
        "unconditionally where it should be skipped. Updating the constant is almost "
        "never the right fix; find what moved first."
    )


def test_build_scenario_still_accepts_a_positional_confounding_strength():
    # build_scenario's own public signature keeps the pre-Phase-18 float form
    # -- see scenarios.py's docstring for why -- so this call shape, used all
    # over the existing suite and by cp_evaluation/cp_cli, must keep working
    # unchanged.
    strong = build_scenario("D", 4.0)
    weak = build_scenario("D", 0.5)

    assert strong.assignment.strength == 4.0  # type: ignore[attr-defined]
    assert weak.assignment.strength == 0.5  # type: ignore[attr-defined]


def test_build_scenario_float_facade_matches_the_widened_params_path():
    # The one thing that could silently break in translating the compat
    # facade to ScenarioParams: threading the float into the wrong field, or
    # applying it with the wrong operation. Same seed, same scenario, two call
    # paths -- their generated frames must agree exactly.
    via_float = build_scenario("D", 3.0).generate(n_units=500, rng=np.random.default_rng(7))
    via_params = SCENARIOS["D"].build(ScenarioParams(confounding_strength=3.0)).generate(
        n_units=500, rng=np.random.default_rng(7)
    )

    assert _digest_simulated(via_float) == _digest_simulated(via_params)
