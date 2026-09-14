"""Named scenario presets over the component grid.

Presets exist for legibility; the benchmark grid varies confounding strength
continuously through `build_scenario`, which is why each spec is a builder
rather than a fixed instance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from cp_core.dataset import CausalDataset, CausalSchema
from cp_core.manifest import ScenarioReference

from cp_simulation.components import (
    BINARY_COVARIATE_PROBABILITY,
    BaselineSelectionAssignment,
    ConstantEffect,
    HeterogeneousEffect,
    LogisticBaseline,
    MissingnessMechanism,
    NegativeControlOutcome,
    NonDifferentialMisclassification,
    OutcomeProxy,
    PropensityAssignment,
    RandomAssignment,
    UnitGenerator,
    sigmoid,
)
from cp_simulation.dgp import CrossSectionDGP
from cp_simulation.params import AXIS_NAMES, ScenarioParams

_COVARIATES = ("x1", "x2")
_BASELINE = LogisticBaseline(intercept=-1.0, coefficients=(0.8, 0.5))


def _baseline_for(params: ScenarioParams) -> LogisticBaseline:
    """The shared baseline, with this cell's noise added.

    `_BASELINE` is a single module-level instance reused by every scenario's
    registration; it cannot itself carry a per-cell `noise_scale` without
    every other scenario seeing the same value. `replace` builds a fresh
    instance at build time instead, leaving `_BASELINE`'s own `noise_scale`
    (its dataclass default, 0.0) untouched as the shared template. At the
    default `noise_scale = 0.0` this reproduces `_BASELINE` exactly -- an
    equal dataclass, not merely an equivalent one -- so it does not disturb
    `config_hash` or the pre-Phase-18 digest guard.
    """
    return replace(_BASELINE, noise_scale=params.noise_scale)


def _treatment_share(intercept: float) -> float:
    """Map a logit-scale intercept onto a Bernoulli share.

    `treatment_intercept` is a logit shift by construction -- it is exactly
    what `PropensityAssignment.intercept` already means -- so a confounded
    scenario reads it unconverted. A randomised or baseline-selection
    scenario instead exposes a `share` in [0, 1], so the same intercept has
    to pass through a link function to mean the same thing there: sigmoid is
    the natural choice, since it is the inverse of the logit `share` itself
    is defined on (`BaselineSelectionAssignment` solves for its own intercept
    via `logit(share)`). At the axis's default of 0.0 this returns exactly
    0.5, reproducing every scenario's pre-Phase-18 default share unchanged.
    Checked across the axis's perturbed value and a wider spread --
    sigmoid(-2)=0.12, sigmoid(-1)=0.27, sigmoid(0)=0.50, sigmoid(1)=0.73,
    sigmoid(2)=0.88, sigmoid(3)=0.95 -- every value lands well inside (0, 1)
    and moves monotonically, so the mapping stays sane across the intercept
    range the grid is expected to explore.
    """
    return float(sigmoid(np.asarray(intercept)))


# Homogeneous effect sizes, named once so a scenario's `build` and its
# `declared_effect` label read the same value rather than two copies of the
# same literal drifting apart.
_TAU_A = 0.05
_TAU_B = 0.0
_TAU_C = -0.03
_TAU_D = 0.05
_TAU_G = 0.05
_TAU_H = 0.05

# Scenario F's heterogeneous effect: +8pp for new users (x2 = 1), +1pp for
# returning users (x2 = 0).
_HETEROGENEOUS_BASE_TAU = 0.01
_HETEROGENEOUS_INCREMENT = 0.07

# One shared definition so the tests and the measured harness cannot drift
# into using different negative controls and then disagree about what the
# check measured. Its coefficients share the covariates every cross-sectional
# scenario declares, at a different intercept and slope from the outcome's own
# baseline so the column is not a copy of the outcome surface.
DEFAULT_NEGATIVE_CONTROL = NegativeControlOutcome(intercept=-0.6, coefficients=(0.9, 0.4))


@dataclass(frozen=True)
class ScenarioSpec:
    """A named preset over the component grid.

    `declared_effect` is a function of the parameters, not a fixed float:
    once effect size is a grid axis, a scenario no longer has one true effect,
    so its advertised label has to be checkable at every point rather than at
    one. See `tests/simulation/test_scenarios.py::test_measured_ate_matches_the_declared_label`
    for the contract this must satisfy.

    `axes` names the grid axes this scenario's `build` actually reads --
    derived from the builder, not from the scenario's name or description.
    Without it, a scenario that ignores an axis still produces one cell per
    value of that axis in the benchmark grid, all identical: the same
    absent-versus-zero confusion Phase 16's benchmark heatmaps were built to
    avoid (spec section 5.1). It is checked, not trusted: see
    `tests/simulation/test_axis_declarations.py`, which measures a property of
    each scenario's output at a perturbed value of every axis and fails if a
    declared axis's property does not move or an undeclared one's does.
    """

    scenario_id: str
    name: str
    declared_effect: Callable[[ScenarioParams], float]
    description: str
    build: Callable[[ScenarioParams], CrossSectionDGP]
    axes: frozenset[str]


def _homogeneous_declared_effect(tau: float) -> Callable[[ScenarioParams], float]:
    """Label for a scenario whose effect is the same for every unit.

    `effect_scale` is the only grid axis this label responds to today: the
    other axes this phase adds do not move a homogeneous scenario's mean
    effect.
    """

    def declared_effect(params: ScenarioParams) -> float:
        return tau * params.effect_scale

    return declared_effect


def _heterogeneous_declared_effect(
    base_tau: float, increment: float
) -> Callable[[ScenarioParams], float]:
    """Label for a heterogeneous scenario: the population mean CATE.

    Mean CATE is `base_tau + increment * P(x = 1)`, and `P(x = 1)` is a
    property of `UnitGenerator`'s own configuration
    (`BINARY_COVARIATE_PROBABILITY`) rather than a constant chosen here.
    Reading it structurally off the generator, instead of measuring it from a
    generated sample, keeps this label deterministic by construction and
    independent of whatever sample a caller -- including a test checking this
    very label against measured ground truth -- generates and measures.

    `heterogeneity_scale` does not appear in this formula, and that is by
    construction rather than an omission: spec section 4.4 compensates
    `base_tau` whenever `increment` is scaled by `heterogeneity_scale`
    (`base_tau' = base_tau - (k - 1) * increment * P(x = 1)`), specifically so
    the population mean is invariant to it --
    `base_tau' + increment' * P(x = 1) == base_tau + increment * P(x = 1)`
    for every `k`. `heterogeneity_scale` moves the spread of the effect, not
    its level, which is exactly what makes it a distinct axis from
    `effect_scale` rather than a second copy of it.
    """

    def declared_effect(params: ScenarioParams) -> float:
        scale = params.effect_scale
        return base_tau * scale + increment * scale * BINARY_COVARIATE_PROBABILITY

    return declared_effect


def _randomised(scenario_id: str, tau: float) -> Callable[[ScenarioParams], CrossSectionDGP]:
    def build(params: ScenarioParams) -> CrossSectionDGP:
        return CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_COVARIATES),
            baseline=_baseline_for(params),
            effect=ConstantEffect(tau=tau * params.effect_scale),
            assignment=RandomAssignment(share=_treatment_share(params.treatment_intercept)),
            scenario_id=scenario_id,
        )

    return build


def _confounded(scenario_id: str, tau: float) -> Callable[[ScenarioParams], CrossSectionDGP]:
    def build(params: ScenarioParams) -> CrossSectionDGP:
        return CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_COVARIATES),
            baseline=_baseline_for(params),
            effect=ConstantEffect(tau=tau * params.effect_scale),
            assignment=PropensityAssignment(
                coefficients=(0.9, 0.7),
                strength=params.confounding_strength,
                # Unconverted: `PropensityAssignment.intercept` is already a
                # logit shift, which is exactly what `treatment_intercept`
                # means -- no link function needed here, unlike the `share`
                # mechanisms below.
                intercept=params.treatment_intercept,
            ),
            scenario_id=scenario_id,
        )

    return build


def _missing_data(scenario_id: str, tau: float) -> Callable[[ScenarioParams], CrossSectionDGP]:
    """Scenario D's confounding, plus missingness that depends on the outcome.

    MNAR rather than MAR: the parent spec allows either, and MNAR is the case
    complete-case analysis cannot be adjusted out of, which is what makes it
    worth a scenario of its own.

    The rate/strength pair is a starting hypothesis until Task 8 measures
    whether it produces a complete-case bias large enough to test against.
    """

    def build(params: ScenarioParams) -> CrossSectionDGP:
        return replace(
            _confounded(scenario_id, tau)(params),
            missingness=MissingnessMechanism(
                mechanism="MNAR", rate=0.25, strength=2.0
            ),
        )

    return build


def _heterogeneous(
    scenario_id: str, base_tau: float, increment: float
) -> Callable[[ScenarioParams], CrossSectionDGP]:
    """Randomised assignment with an effect that varies by the binary covariate.

    x2 is the binary covariate from UnitGenerator, read here as a "new user"
    flag: new users gain 8pp, returning users 1pp.

    `heterogeneity_scale` scales `increment` and compensates `base_tau` in the
    same motion (spec section 4.4), so the population mean CATE this
    generates is unaffected by it -- only the spread across x2 moves. Only
    this scenario's effect has an `increment` to scale at all: every other
    scenario uses `ConstantEffect`, whose CATE is the same for every unit
    regardless of `heterogeneity_scale`, which is why this is the only
    scenario that declares that axis. `effect_scale` then scales the
    already-compensated `base_tau` and `increment` together, preserving shape,
    exactly as `_heterogeneous_declared_effect` computes it.
    """

    def build(params: ScenarioParams) -> CrossSectionDGP:
        k = params.heterogeneity_scale
        scaled_increment = k * increment
        compensated_base_tau = base_tau - (k - 1.0) * increment * BINARY_COVARIATE_PROBABILITY
        scale = params.effect_scale
        return CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_COVARIATES),
            baseline=_baseline_for(params),
            effect=HeterogeneousEffect(
                base_tau=compensated_base_tau * scale,
                covariate_index=1,
                increment=scaled_increment * scale,
            ),
            assignment=RandomAssignment(share=_treatment_share(params.treatment_intercept)),
            scenario_id=scenario_id,
        )

    return build


def _selected(scenario_id: str, tau: float) -> Callable[[ScenarioParams], CrossSectionDGP]:
    def build(params: ScenarioParams) -> CrossSectionDGP:
        return CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_COVARIATES),
            baseline=_baseline_for(params),
            effect=ConstantEffect(tau=tau * params.effect_scale),
            assignment=BaselineSelectionAssignment(
                strength=8.0 * params.confounding_strength,
                share=_treatment_share(params.treatment_intercept),
            ),
            scenario_id=scenario_id,
        )

    return build


SCENARIOS: dict[str, ScenarioSpec] = {
    "A": ScenarioSpec(
        scenario_id="A",
        name="Simple randomised experiment",
        declared_effect=_homogeneous_declared_effect(_TAU_A),
        description="Random assignment, homogeneous +5pp effect.",
        build=_randomised("A", _TAU_A),
        # `_randomised` reads `effect_scale`, `treatment_intercept` and
        # `noise_scale`. `confounding_strength` is meaningless under
        # RandomAssignment, and there is no `increment` here for
        # `heterogeneity_scale` to scale (ConstantEffect's CATE is the same
        # for every unit, so its spread is exactly 0 regardless).
        axes=frozenset({"effect_scale", "treatment_intercept", "noise_scale"}),
    ),
    "B": ScenarioSpec(
        scenario_id="B",
        name="No treatment effect",
        declared_effect=_homogeneous_declared_effect(_TAU_B),
        description="Random assignment, exactly zero effect. Any systematic "
        "positive finding here is a false positive.",
        build=_randomised("B", _TAU_B),
        # Same wiring as A, but `_TAU_B == 0.0`: `effect_scale` still
        # multiplies it, but zero times any scale is still exactly zero, so
        # the axis moves nothing and is not declared. `treatment_intercept`
        # and `noise_scale` are independent of tau and still move their
        # properties.
        axes=frozenset({"treatment_intercept", "noise_scale"}),
    ),
    "C": ScenarioSpec(
        scenario_id="C",
        name="Harmful intervention",
        declared_effect=_homogeneous_declared_effect(_TAU_C),
        description="Random assignment, homogeneous -3pp effect.",
        build=_randomised("C", _TAU_C),
        axes=frozenset({"effect_scale", "treatment_intercept", "noise_scale"}),
    ),
    "D": ScenarioSpec(
        scenario_id="D",
        name="Confounding",
        declared_effect=_homogeneous_declared_effect(_TAU_D),
        description="Covariates drive both assignment and outcome, so naive "
        "comparisons are biased upward.",
        build=_confounded("D", _TAU_D),
        # `_confounded` reads `confounding_strength` (PropensityAssignment's
        # strength), `treatment_intercept` (its intercept, unconverted),
        # `effect_scale` and `noise_scale`. No `increment` exists here for
        # `heterogeneity_scale` to scale.
        axes=frozenset(
            {"confounding_strength", "treatment_intercept", "noise_scale", "effect_scale"}
        ),
    ),
    "F": ScenarioSpec(
        scenario_id="F",
        name="Heterogeneous treatment effects",
        declared_effect=_heterogeneous_declared_effect(
            _HETEROGENEOUS_BASE_TAU, _HETEROGENEOUS_INCREMENT
        ),
        description="Random assignment; +8pp for new users (x2=1), +1pp for "
        "returning users (x2=0). The population average depends on the "
        "covariate mix, so it is measured rather than assumed.",
        build=_heterogeneous("F", _HETEROGENEOUS_BASE_TAU, _HETEROGENEOUS_INCREMENT),
        # `_heterogeneous` reads all four wired axes. It is the only scenario
        # with a real `increment` to scale, so it is the only one that can
        # ever respond to `heterogeneity_scale` -- every other scenario's
        # effect is `ConstantEffect`, whose CATE spread is structurally zero
        # regardless of scale.
        axes=frozenset(
            {"effect_scale", "treatment_intercept", "noise_scale", "heterogeneity_scale"}
        ),
    ),
    "G": ScenarioSpec(
        scenario_id="G",
        name="Selection bias",
        declared_effect=_homogeneous_declared_effect(_TAU_G),
        description="Treatment is taken up by units with higher untreated "
        "outcome probability, so naive comparison overstates the effect.",
        build=_selected("G", _TAU_G),
        # `_selected` reads `confounding_strength` (scaling
        # BaselineSelectionAssignment.strength), `treatment_intercept` (via
        # its `share`, since BaselineSelectionAssignment exposes a
        # probability rather than a raw logit shift), `effect_scale` and
        # `noise_scale`.
        axes=frozenset(
            {"confounding_strength", "treatment_intercept", "noise_scale", "effect_scale"}
        ),
    ),
    "H": ScenarioSpec(
        scenario_id="H",
        name="Missing data",
        declared_effect=_homogeneous_declared_effect(_TAU_H),
        description="Scenario D's confounding plus outcome missingness that "
        "depends on the outcome itself, so complete-case analysis is biased "
        "in a way adjusting for the covariates does not repair.",
        build=_missing_data("H", _TAU_H),
        # `_missing_data` builds on `_confounded`, so it inherits every axis
        # `_confounded` reads; the missingness mechanism it adds is fixed
        # (rate=0.25, strength=2.0), reading nothing from `params`.
        axes=frozenset(
            {"confounding_strength", "treatment_intercept", "noise_scale", "effect_scale"}
        ),
    ),
}


def build_scenario(
    scenario_id: str,
    params: float | ScenarioParams = 1.0,
    *,
    negative_control: NegativeControlOutcome | None = None,
    misclassification: NonDifferentialMisclassification | None = None,
    missingness: MissingnessMechanism | None = None,
    outcome_proxy: OutcomeProxy | None = None,
) -> CrossSectionDGP:
    """Instantiate a scenario's DGP, at a confounding strength or a full grid point.

    The second positional/keyword argument accepts either a bare
    `confounding_strength: float` -- the pre-Phase-18 call shape used all over
    this project's existing suite, `cp_evaluation.resolve.resolve_cell` and
    `cp_cli`'s `simulate`/`analyze` commands -- or a complete `ScenarioParams`,
    which is what lets the four axes Phase 18 adds
    (`treatment_intercept`, `noise_scale`, `effect_scale`,
    `heterogeneity_scale`) reach a scenario through this, the one public entry
    point every caller goes through. A bare float is equivalent to
    `ScenarioParams(confounding_strength=that_float)`: this is exactly what
    `test_build_scenario_float_facade_matches_the_widened_params_path` pins.
    The parameter is named `params`, not `confounding_strength`, precisely
    because it accepts either shape: under the old name,
    `build_scenario("D", confounding_strength=ScenarioParams(confounding_strength=3.0))`
    was a legal call that read as a category error -- a `ScenarioParams` handed
    to something named `confounding_strength`. `params` makes both accepted
    shapes equally at home under the keyword.

    The three optional components are attached after the scenario's own
    builder has run, rather than threaded through it, so all six builders stay
    as they were and a scenario that opts into none of them produces exactly
    the frame it produced before.
    """
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"unknown scenario {scenario_id!r}; available: {sorted(SCENARIOS)}"
        )
    resolved_params = (
        params
        if isinstance(params, ScenarioParams)
        else ScenarioParams(confounding_strength=params)
    )
    dgp = SCENARIOS[scenario_id].build(resolved_params)
    if (
        negative_control is None
        and misclassification is None
        and missingness is None
        and outcome_proxy is None
    ):
        return dgp
    return replace(
        dgp,
        negative_control=negative_control,
        misclassification=misclassification,
        # H bakes its own mechanism in, so an explicit argument must not
        # silently erase it when the caller passed none.
        missingness=missingness if missingness is not None else dgp.missingness,
        outcome_proxy=outcome_proxy,
    )


_SUPPORTED_REFERENCE_PARAMS = frozenset({"n_units", *AXIS_NAMES})


def resolve_scenario_reference(
    reference: ScenarioReference, schema: CausalSchema
) -> CausalDataset:
    """Regenerate the dataset a `ScenarioReference` names.

    Lives here rather than in `cp_core.manifest` beside the reference type
    itself because it needs `build_scenario`, and `.importlinter`'s
    `estimators-cannot-see-truth` contract forbids `cp_core` from importing
    this package.

    Supports `n_units` (required) plus every name in `ScenarioParams.AXIS_NAMES`
    (each optional). A reference built from a scenario with a
    `negative_control`, `misclassification`, `missingness`, or `outcome_proxy`
    override cannot be reconstructed from those params alone, and is refused
    rather than silently regenerating a different dataset -- see
    `ScenarioReference`'s own docstring.

    An axis absent from `reference.params` falls back to `ScenarioParams`'
    own default for it -- this is what lets a manifest written before Phase
    18 (recording only `n_units` and, optionally, `confounding_strength`)
    keep replaying unchanged: every new axis it never mentions is missing,
    not present-and-zero, and must resolve to whatever `ScenarioParams`
    already used for it. That makes each default in `ScenarioParams` a
    compatibility surface, not a convenience: changing one changes what
    every manifest already on disk that omits that axis replays as, silently
    and without the manifest itself changing at all.

    The schema comparison guards against the scenario registry changing under a
    stored manifest, and is weaker than it may look: several scenarios share a
    schema (A, B and C carry one; D, G and H carry another), so a reference
    naming a different scenario *within* one of those groups passes it. Nothing
    here can distinguish those; only the recorded scenario id can.
    """
    unknown = set(reference.params) - _SUPPORTED_REFERENCE_PARAMS
    if unknown:
        raise ValueError(
            f"ScenarioReference.params has unsupported keys {sorted(unknown)}; only "
            f"{sorted(_SUPPORTED_REFERENCE_PARAMS)} can be regenerated from a reference "
            "alone. A scenario built with a negative_control, misclassification, "
            "missingness, or outcome_proxy override cannot be reconstructed this way."
        )
    if "n_units" not in reference.params:
        raise KeyError(
            "ScenarioReference.params must include 'n_units' to regenerate a dataset"
        )
    n_units = int(reference.params["n_units"])  # type: ignore[call-overload]
    scenario_params = ScenarioParams(
        **{
            axis: float(reference.params[axis])  # type: ignore[arg-type]
            for axis in AXIS_NAMES
            if axis in reference.params
        }
    )

    dgp = build_scenario(reference.scenario, scenario_params)
    generated = dgp.generate(n_units=n_units, rng=np.random.default_rng(reference.seed))
    dataset = generated.dataset
    if dataset.schema != schema:
        raise ValueError(
            f"regenerating scenario {reference.scenario!r} produced a schema that does "
            "not match the manifest's recorded schema -- the scenario registry may have "
            "changed since this reference was built"
        )
    return dataset
