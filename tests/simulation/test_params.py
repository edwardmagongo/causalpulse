"""`ScenarioParams` and the widened `ScenarioSpec.build`.

`build_scenario`'s second parameter, `params`, accepts either a bare
`confounding_strength: float` -- the pre-Phase-18 call shape used all over
this project's existing suite, `cp_evaluation.resolve.resolve_cell` and
`cp_cli`'s `simulate`/`analyze` commands -- or a complete `ScenarioParams`,
so every caller keeps working unchanged while the union branch is the one
path a `ScenarioParams` reaches a scenario's builder through. A bare float is
internally converted to `ScenarioParams(confounding_strength=that_float)`
before being handed to the registry callable.

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


def _nine_digits(value):
    """A float, or every element of an array, as text at nine significant digits."""
    if value is None:
        return None
    if np.ndim(value) == 0:
        return f"{float(value):.9g}"
    return [f"{float(element):.9g}" for element in np.asarray(value).ravel()]


def _digest_simulated(simulated):
    """A stable fingerprint of a generated frame and its ground truth.

    Content, not identity: two independently generated but numerically equal
    runs digest the same. Anything that moves a float anywhere in the frame or
    the truth arrays by more than rounding at nine significant digits changes
    this digest.

    Nine digits, not every bit: `exp` and its relatives differ in the last
    place between CPU families, so the truth arrays are not bit-identical
    across machines. Measured on 2026-10-05 between Apple silicon and x86-64
    Linux (numpy 2.5.1): at n_units=2000 the frames were identical, and a
    handful of `true_cate` and `true_propensity` elements per scenario
    differed by 1 to 16 units in the last place (at most 1.9e-15 relative).
    A changed rng draw, which is what this guards against, moves values in
    the first digit.

    Deliberately excludes `truth.config_hash`: that field is provenance
    metadata describing the configuration, not generated data, and it can
    move (e.g. a component gaining a new field) without a single generated
    value changing. Mixing it into a digest meant to catch drift in *data*
    makes an unrelated metadata change look like a data regression.
    """
    frame_bytes = simulated.dataset.frame.to_csv(index=True, float_format="%.9g").encode()
    truth = simulated.truth
    payload = {
        "true_ate": _nine_digits(truth.true_ate),
        "true_att": _nine_digits(truth.true_att),
        "true_cate": _nine_digits(truth.true_cate),
        "true_propensity": _nine_digits(truth.true_propensity),
        "true_confounders": list(truth.true_confounders),
        "scenario_id": truth.scenario_id,
        "true_treatment": (
            None if truth.true_treatment is None else _nine_digits(truth.true_treatment)
        ),
        "complete_outcome": (
            None if truth.complete_outcome is None else _nine_digits(truth.complete_outcome)
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
#
# Recomputed after `config_hash` was removed from `_digest_simulated`'s
# payload above: every constant below changed as a result, because the
# payload being hashed changed shape for every scenario, not because any
# scenario's generated data changed.
#
# Recomputed again on 2026-10-05, when `_digest_simulated` began rounding to
# nine significant digits (see its docstring: the truth arrays differ in the
# last place between CPU families, and the suite first ran on x86-64 Linux
# that day). As before, the fingerprint changed and the data did not: with the
# rounding stashed, the previous constants still matched on the machine they
# were captured on, and these were computed from that same run.
_PRE_PHASE18_DIGESTS = {
    "A": "617bba1ccae2f187767fb2319c12c6e6bb75a1e305c45e56678087cb40f8f01c",
    "B": "aba1e1a279a6848831bd0cd4ad242f42a445d798237b0bc7e2289832222c2b8b",
    "C": "e8a8960f0ca5bedb620ff680f4a2fc47bf512f5d2642773ae7bd9da514669dd7",
    "D": "aa5b90fb547507a4ac0f03c7bed63714474aa701b8213a0c6086e3f944a48a13",
    "F": "3922c6b0f73917e4bdc2a8ce4a618820a1f91764a1ad37a8ba204fb794634669",
    "G": "c13339a97be417f16bf1f32a4ae1faef507d0704191c8a003c8cf1ee48bd0744",
    "H": "20bc791b96ae00ed2c93ce6d8545f4c691bccd0bfb0deaec3d4b84aec92e69c9",
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
    # build_scenario's `params` argument still accepts a bare float
    # positionally -- see scenarios.py's docstring for why -- so this call
    # shape, used all over the existing suite and by cp_evaluation/cp_cli,
    # must keep working unchanged.
    strong = build_scenario("D", 4.0)
    weak = build_scenario("D", 0.5)

    assert strong.assignment.strength == 4.0  # type: ignore[attr-defined]
    assert weak.assignment.strength == 0.5  # type: ignore[attr-defined]


def test_build_scenario_float_facade_matches_the_widened_params_path():
    # The one thing that could silently break in translating the compat
    # facade to ScenarioParams: threading the float into the wrong field, or
    # applying it with the wrong operation. Same seed, same scenario, two call
    # paths -- their generated frames must agree exactly. Both go through
    # build_scenario itself, not SCENARIOS["D"].build directly, so this
    # exercises the isinstance branch in build_scenario's own body -- the
    # branch that is the public entry point's actual widening -- rather than
    # only the registry callable underneath it.
    via_float = build_scenario("D", 3.0).generate(n_units=500, rng=np.random.default_rng(7))
    via_params = build_scenario("D", ScenarioParams(confounding_strength=3.0)).generate(
        n_units=500, rng=np.random.default_rng(7)
    )

    assert _digest_simulated(via_float) == _digest_simulated(via_params)


def test_build_scenario_passes_a_full_scenario_params_through_unabridged():
    # The isinstance(params, ScenarioParams) branch is the headline deliverable
    # of widening build_scenario -- it is the one thing that makes the four
    # axes this phase adds reachable from the public entry point at all -- and
    # nothing else in this suite calls build_scenario with a full
    # ScenarioParams. A build_scenario that silently narrowed a passed-in
    # ScenarioParams down to just confounding_strength (e.g.
    # `ScenarioParams(confounding_strength=params.confounding_strength)`)
    # would still satisfy every other test in this repo, because every other
    # caller that reaches this branch only ever sets confounding_strength.
    #
    # Confirmed to fail under exactly that mutation and to pass again once
    # reverted.
    params = ScenarioParams(effect_scale=2.0, noise_scale=1.5, heterogeneity_scale=2.0)

    via_facade = build_scenario("F", params).generate(n_units=500, rng=np.random.default_rng(5))
    via_registry = SCENARIOS["F"].build(params).generate(
        n_units=500, rng=np.random.default_rng(5)
    )

    assert _digest_simulated(via_facade) == _digest_simulated(via_registry)
