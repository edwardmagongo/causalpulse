"""The checked axis declarations -- `ScenarioSpec.axes`.

Spec section 5.1: a scenario whose builder ignores a grid axis still produces
one cell per value of that axis, all identical -- the same absent-versus-zero
confusion Phase 16's benchmark heatmaps were built to avoid. `axes` is the
declaration that says which axes a scenario actually responds to, and this
module is what holds it honest: for every scenario x axis pair, a measured
property of the generated output must move when the axis is declared and must
not move at all when it is not.

Every scenario's builder reads `treatment_intercept`, `noise_scale` and
`effect_scale`, and `heterogeneity_scale` is wired into scenario F (the only
scenario with an `increment` for it to scale). Declaring only what is
actually wired keeps this suite meaningful: a scenario's declaration -- and
this test's "must move" branch for it -- reflects what its builder reads.
See `docs/specs/2026-09-13-causal-core-phase18-design.md` section 12 for the
full reasoning and the measurements behind every threshold below, and for
how each axis was wired into its scenarios.
"""

from dataclasses import replace

import numpy as np
import pytest
from cp_simulation.params import AXIS_NAMES, ScenarioParams
from cp_simulation.scenarios import SCENARIOS

# Every generated comparison in this module uses the same size and seed, so
# the measured thresholds below (captured at this exact configuration) stay
# valid. See docs/specs/2026-09-13-causal-core-phase18-design.md section 12
# for the sweep that produced them.
_N_UNITS = 10_000
_SEED = 0


def _generate(scenario_id: str, params: ScenarioParams):
    dgp = SCENARIOS[scenario_id].build(params)
    return dgp.generate(n_units=_N_UNITS, rng=np.random.default_rng(_SEED))


def _measured_true_ate(simulated) -> float:
    """Property for `effect_scale`: the measured mean effect.

    `effect_scale` is meant to scale tau, and `true_ate` is computed from the
    realised potential outcomes rather than copied from any input tau, so a
    scenario that actually scales its effect moves this and one that does not
    cannot.
    """
    return float(simulated.truth.true_ate)


def _treated_share(simulated) -> float:
    """Property for `treatment_intercept`: the realised treated fraction.

    `treatment_intercept` is meant to shift a propensity model's intercept,
    which moves the overall treated share without changing which covariates
    confound -- so the treated share is the property that isolates it.
    """
    frame = simulated.dataset.frame
    schema = simulated.dataset.schema
    return float(frame[schema.treatment].mean())


def _residual_outcome_variation(simulated) -> float:
    """Property for `noise_scale`: outcome variance the covariates don't explain.

    `noise_scale` adds independent noise inside the baseline's logit -- signal
    the covariates cannot predict -- so the residual variance of a linear fit
    of outcome on covariates is the property built to isolate it, rather than
    the outcome's raw variance, which also moves with the covariate
    coefficients and the treatment effect.

    Outcome values are dropped where missing (Scenario H) rather than
    treated as zero, matching how a real analyst would fit this regression on
    an outcome with gaps in it.
    """
    frame = simulated.dataset.frame
    schema = simulated.dataset.schema
    outcome = frame[schema.outcome].to_numpy(dtype=float)
    covariates = frame[list(schema.covariates)].to_numpy(dtype=float)
    observed = ~np.isnan(outcome)
    design = np.column_stack([np.ones(int(observed.sum())), covariates[observed]])
    coefficients, *_ = np.linalg.lstsq(design, outcome[observed], rcond=None)
    residuals = outcome[observed] - design @ coefficients
    return float(np.var(residuals))


def _cate_spread(simulated) -> float:
    """Property for `heterogeneity_scale`: the spread of the true CATE.

    `heterogeneity_scale` is meant to scale how much the effect varies across
    units, not its average level, so the standard deviation of the true CATE
    -- not its mean, which `effect_scale`'s property already covers -- is
    what isolates it. A homogeneous scenario's CATE is constant, so this is
    exactly zero for it regardless of `heterogeneity_scale`, which is a
    property of the scenario's effect function, not a flaw in the property.
    """
    return float(np.std(simulated.truth.true_cate))


def _bias_like_quantity(simulated) -> float:
    """Property for `confounding_strength`: naive diff-in-means bias.

    `confounding_strength` is meant to control how much treatment assignment
    depends on the covariates that also drive the outcome, which shows up as
    a gap between the naive (unadjusted) difference in means and the true
    ATE. Missing outcomes (Scenario H) are dropped via `nanmean` rather than
    treated as zero, matching what a naive analyst computing this quantity
    from real data with gaps in it would actually do.
    """
    frame = simulated.dataset.frame
    schema = simulated.dataset.schema
    outcome = frame[schema.outcome].to_numpy(dtype=float)
    treated = frame[schema.treatment].to_numpy()
    naive_diff = float(np.nanmean(outcome[treated == 1]) - np.nanmean(outcome[treated == 0]))
    return naive_diff - float(simulated.truth.true_ate)


_PROPERTY = {
    "confounding_strength": _bias_like_quantity,
    "treatment_intercept": _treated_share,
    "noise_scale": _residual_outcome_variation,
    "effect_scale": _measured_true_ate,
    "heterogeneity_scale": _cate_spread,
}

# How far each axis is perturbed from `ScenarioParams()`'s default, in the
# direction that gives the underlying mechanism the most room to respond.
_PERTURBED_VALUE = {
    "confounding_strength": 3.0,
    "treatment_intercept": 2.0,
    "noise_scale": 2.0,
    "effect_scale": 3.0,
    "heterogeneity_scale": 3.0,
}

# Measured floors for "the property moved" at n_units=10_000, seed=0. First
# calibrated against a direct component probe (bypassing the scenario
# registry), then re-measured through the real scenario builders once they
# read these axes: headroom is 2.4x-7.9x across all four new axes and all
# seven scenarios, tightest at the pre-existing confounding_strength axis
# against scenario H (2.41x). See
# docs/specs/2026-09-13-causal-core-phase18-design.md section 12 for the
# full per-axis table.
_MOVE_THRESHOLD = {
    "confounding_strength": 0.02,
    "treatment_intercept": 0.05,
    "noise_scale": 0.01,
    "effect_scale": 0.02,
    "heterogeneity_scale": 0.01,
}

assert set(_PROPERTY) == set(_PERTURBED_VALUE) == set(_MOVE_THRESHOLD) == set(AXIS_NAMES)


@pytest.mark.parametrize("axis_name", AXIS_NAMES)
@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_declared_axes_move_and_undeclared_axes_do_not(scenario_id, axis_name):
    base_params = ScenarioParams()
    perturbed_params = replace(base_params, **{axis_name: _PERTURBED_VALUE[axis_name]})
    measure = _PROPERTY[axis_name]

    base_value = measure(_generate(scenario_id, base_params))
    perturbed_value = measure(_generate(scenario_id, perturbed_params))

    if axis_name in SCENARIOS[scenario_id].axes:
        assert abs(perturbed_value - base_value) > _MOVE_THRESHOLD[axis_name], (
            f"scenario {scenario_id!r} declares axis {axis_name!r} but its measured "
            f"property barely moved ({base_value!r} -> {perturbed_value!r})"
        )
    else:
        # Exact, not a tolerance: nothing in the generation pipeline reads this
        # axis for this scenario, so the two runs consume the shared RNG
        # identically and must produce bit-identical output -- any difference
        # at all means the axis reached the builder without being declared.
        assert perturbed_value == base_value, (
            f"scenario {scenario_id!r} does not declare axis {axis_name!r} but its "
            f"measured property moved anyway ({base_value!r} -> {perturbed_value!r})"
        )


# Axes with a well-defined *direction*, not just a move. `confounding_strength`
# is left out (Task 5's original scope, unchanged here) and so is
# `heterogeneity_scale` (only F declares it, and the brief's Step 1 names only
# these three).
_DIRECTIONAL_AXES = ("effect_scale", "noise_scale", "treatment_intercept")


def _expected_direction(scenario_id: str, axis_name: str) -> int:
    """+1 if a scenario's declared property should rise under the perturbed
    value, -1 if it should fall.

    `noise_scale` and `treatment_intercept` are unconditionally +1: the
    perturbed value is larger than the default for both (more injected noise,
    a more positive propensity intercept), and both `_residual_outcome_variation`
    and `_treated_share` are monotonically increasing in their axis for every
    assignment mechanism this project wires (independent additive noise before
    clipping; a pointwise-monotonic sigmoid link before averaging).

    `effect_scale` cannot be a blanket +1: Scenario C's declared effect is
    negative, so scaling it up makes `true_ate` more negative, not less. The
    expected sign is read off `declared_effect` -- an independently-written
    and separately-tested label (Task 4), not the builder under test here --
    rather than hardcoded per scenario, so this stays correct even if a
    scenario's tau sign changes later.
    """
    if axis_name == "effect_scale":
        spec = SCENARIOS[scenario_id]
        base = ScenarioParams()
        perturbed = replace(base, effect_scale=_PERTURBED_VALUE["effect_scale"])
        delta = spec.declared_effect(perturbed) - spec.declared_effect(base)
        assert delta != 0.0, (
            f"scenario {scenario_id!r} declares effect_scale but its own label "
            "does not move under it -- this test cannot infer a direction"
        )
        return 1 if delta > 0 else -1
    return 1


@pytest.mark.parametrize("axis_name", _DIRECTIONAL_AXES)
@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_declared_axes_move_in_the_correct_direction(scenario_id, axis_name):
    # "Moves" alone would pass an axis wired backwards -- e.g. `tau / scale`
    # instead of `tau * scale`, which still moves `true_ate` but the wrong
    # way as `scale` grows. This checks the sign of the move, not just its
    # magnitude.
    if axis_name not in SCENARIOS[scenario_id].axes:
        pytest.skip(f"scenario {scenario_id!r} does not declare axis {axis_name!r}")

    base_params = ScenarioParams()
    perturbed_params = replace(base_params, **{axis_name: _PERTURBED_VALUE[axis_name]})
    measure = _PROPERTY[axis_name]

    base_value = measure(_generate(scenario_id, base_params))
    perturbed_value = measure(_generate(scenario_id, perturbed_params))
    # `1 if perturbed_value > base_value else -1` maps "did not move at all"
    # onto -1, which would let a dead axis pass here whenever the expected
    # direction also happens to be -1 (e.g. scenario C's effect_scale) --
    # `test_declared_axes_move_and_undeclared_axes_do_not` checks movement,
    # but this test must not silently rely on it never regressing.
    assert perturbed_value != base_value, (
        f"scenario {scenario_id!r} axis {axis_name!r} did not move the measured "
        f"property at all ({base_value!r}) -- this test cannot infer a direction "
        "from no movement"
    )
    expected = _expected_direction(scenario_id, axis_name)
    actual = 1 if perturbed_value > base_value else -1

    assert actual == expected, (
        f"scenario {scenario_id!r} axis {axis_name!r} moved the wrong way: "
        f"{base_value!r} -> {perturbed_value!r} (expected direction {expected:+d})"
    )


def test_heterogeneity_scale_increases_scenario_fs_cate_spread():
    # heterogeneity_scale is excluded from _DIRECTIONAL_AXES above -- only F
    # declares it, and the brief's Step 1 named only the other three -- but
    # its direction is still checkable and still worth pinning: a reciprocal
    # wiring (`increment / k` with base_tau compensated consistently) also
    # keeps the mean CATE invariant and clears _MOVE_THRESHOLD (measured
    # spread 0.035 -> 0.0117, |delta| = 0.023 > 0.01), so it would pass every
    # other check in this module while scaling the spread the wrong way.
    # Perturbing k upward (3.0, the same value used elsewhere in this module)
    # must make the CATE spread larger, not smaller.
    base_params = ScenarioParams()
    perturbed_params = replace(
        base_params, heterogeneity_scale=_PERTURBED_VALUE["heterogeneity_scale"]
    )

    base_spread = _cate_spread(_generate("F", base_params))
    perturbed_spread = _cate_spread(_generate("F", perturbed_params))

    assert perturbed_spread > base_spread, (
        f"scenario 'F' heterogeneity_scale did not increase the CATE spread: "
        f"{base_spread!r} -> {perturbed_spread!r}"
    )


# The explicit truth table Step 1 derived from reading each builder, locked in
# separately from the measured-behaviour test above: that test would also
# pass if every scenario declared nothing and every property happened to be
# insensitive, so this pins the actual, reasoned-through set.
#
# `effect_scale` is absent from B: its tau is exactly 0.0, and 0.0 times any
# scale is still exactly 0.0, so the axis moves nothing for that scenario even
# though the builder still reads it. `heterogeneity_scale` is present only for
# F: every other scenario's effect is `ConstantEffect`, which has no
# `increment` for that axis to scale, so its CATE spread is structurally zero
# regardless -- see docs/specs/2026-09-13-causal-core-phase18-design.md
# section 12.
_EXPECTED_AXES = {
    "A": frozenset({"effect_scale", "treatment_intercept", "noise_scale"}),
    "B": frozenset({"treatment_intercept", "noise_scale"}),
    "C": frozenset({"effect_scale", "treatment_intercept", "noise_scale"}),
    "D": frozenset(
        {"confounding_strength", "treatment_intercept", "noise_scale", "effect_scale"}
    ),
    "F": frozenset(
        {"effect_scale", "treatment_intercept", "noise_scale", "heterogeneity_scale"}
    ),
    "G": frozenset(
        {"confounding_strength", "treatment_intercept", "noise_scale", "effect_scale"}
    ),
    "H": frozenset(
        {"confounding_strength", "treatment_intercept", "noise_scale", "effect_scale"}
    ),
}


@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_declared_axes_match_what_each_builder_reads(scenario_id):
    assert SCENARIOS[scenario_id].axes == _EXPECTED_AXES[scenario_id]


def test_every_declared_axis_is_a_real_axis_name():
    # A typo in a declaration (e.g. "confounding_strenght") would otherwise
    # silently mean that axis is never exercised for that scenario -- caught
    # here as a failure instead.
    axis_names = set(AXIS_NAMES)
    for scenario_id, spec in SCENARIOS.items():
        unknown = spec.axes - axis_names
        assert not unknown, f"scenario {scenario_id!r} declares unknown axes {unknown}"
