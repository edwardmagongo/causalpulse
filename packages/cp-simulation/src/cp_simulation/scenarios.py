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
)
from cp_simulation.dgp import CrossSectionDGP
from cp_simulation.params import ScenarioParams

_COVARIATES = ("x1", "x2")
_BASELINE = LogisticBaseline(intercept=-1.0, coefficients=(0.8, 0.5))

# One shared definition so the tests and the measured harness cannot drift
# into using different negative controls and then disagree about what the
# check measured. Its coefficients share the covariates every cross-sectional
# scenario declares, at a different intercept and slope from the outcome's own
# baseline so the column is not a copy of the outcome surface.
DEFAULT_NEGATIVE_CONTROL = NegativeControlOutcome(intercept=-0.6, coefficients=(0.9, 0.4))


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    name: str
    declared_effect: float
    description: str
    build: Callable[[ScenarioParams], CrossSectionDGP]


def _randomised(scenario_id: str, tau: float) -> Callable[[ScenarioParams], CrossSectionDGP]:
    def build(_params: ScenarioParams) -> CrossSectionDGP:
        return CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_COVARIATES),
            baseline=_BASELINE,
            effect=ConstantEffect(tau=tau),
            assignment=RandomAssignment(share=0.5),
            scenario_id=scenario_id,
        )

    return build


def _confounded(scenario_id: str, tau: float) -> Callable[[ScenarioParams], CrossSectionDGP]:
    def build(params: ScenarioParams) -> CrossSectionDGP:
        return CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_COVARIATES),
            baseline=_BASELINE,
            effect=ConstantEffect(tau=tau),
            assignment=PropensityAssignment(
                coefficients=(0.9, 0.7), strength=params.confounding_strength
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


def _heterogeneous(scenario_id: str) -> Callable[[ScenarioParams], CrossSectionDGP]:
    """Randomised assignment with an effect that varies by the binary covariate.

    x2 is the binary covariate from UnitGenerator, read here as a "new user"
    flag: new users gain 8pp, returning users 1pp.
    """

    def build(_params: ScenarioParams) -> CrossSectionDGP:
        return CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_COVARIATES),
            baseline=_BASELINE,
            effect=HeterogeneousEffect(base_tau=0.01, covariate_index=1, increment=0.07),
            assignment=RandomAssignment(share=0.5),
            scenario_id=scenario_id,
        )

    return build


def _selected(scenario_id: str, tau: float) -> Callable[[ScenarioParams], CrossSectionDGP]:
    def build(params: ScenarioParams) -> CrossSectionDGP:
        return CrossSectionDGP(
            unit_generator=UnitGenerator(covariate_names=_COVARIATES),
            baseline=_BASELINE,
            effect=ConstantEffect(tau=tau),
            assignment=BaselineSelectionAssignment(
                strength=8.0 * params.confounding_strength
            ),
            scenario_id=scenario_id,
        )

    return build


SCENARIOS: dict[str, ScenarioSpec] = {
    "A": ScenarioSpec(
        scenario_id="A",
        name="Simple randomised experiment",
        declared_effect=0.05,
        description="Random assignment, homogeneous +5pp effect.",
        build=_randomised("A", 0.05),
    ),
    "B": ScenarioSpec(
        scenario_id="B",
        name="No treatment effect",
        declared_effect=0.0,
        description="Random assignment, exactly zero effect. Any systematic "
        "positive finding here is a false positive.",
        build=_randomised("B", 0.0),
    ),
    "C": ScenarioSpec(
        scenario_id="C",
        name="Harmful intervention",
        declared_effect=-0.03,
        description="Random assignment, homogeneous -3pp effect.",
        build=_randomised("C", -0.03),
    ),
    "D": ScenarioSpec(
        scenario_id="D",
        name="Confounding",
        declared_effect=0.05,
        description="Covariates drive both assignment and outcome, so naive "
        "comparisons are biased upward.",
        build=_confounded("D", 0.05),
    ),
    "F": ScenarioSpec(
        scenario_id="F",
        name="Heterogeneous treatment effects",
        declared_effect=0.038,
        description="Random assignment; +8pp for new users (x2=1), +1pp for "
        "returning users (x2=0). The population average depends on the "
        "covariate mix, so it is measured rather than assumed.",
        build=_heterogeneous("F"),
    ),
    "G": ScenarioSpec(
        scenario_id="G",
        name="Selection bias",
        declared_effect=0.05,
        description="Treatment is taken up by units with higher untreated "
        "outcome probability, so naive comparison overstates the effect.",
        build=_selected("G", 0.05),
    ),
    "H": ScenarioSpec(
        scenario_id="H",
        name="Missing data",
        declared_effect=0.05,
        description="Scenario D's confounding plus outcome missingness that "
        "depends on the outcome itself, so complete-case analysis is biased "
        "in a way adjusting for the covariates does not repair.",
        build=_missing_data("H", 0.05),
    ),
}


def build_scenario(
    scenario_id: str,
    confounding_strength: float = 1.0,
    *,
    negative_control: NegativeControlOutcome | None = None,
    misclassification: NonDifferentialMisclassification | None = None,
    missingness: MissingnessMechanism | None = None,
    outcome_proxy: OutcomeProxy | None = None,
) -> CrossSectionDGP:
    """Instantiate a scenario's DGP at a given confounding strength.

    Kept as a positional/keyword `float` rather than accepting `ScenarioParams`
    directly, because `cp_evaluation.resolve.resolve_cell` and `cp_cli`'s
    `simulate`/`analyze` commands call it this way today, and this task widens
    only `cp_simulation`. Internally this builds a `ScenarioParams` from the
    float and hands it to the registry's now-widened `build`.

    The three optional components are attached after the scenario's own
    builder has run, rather than threaded through it, so all six builders stay
    as they were and a scenario that opts into none of them produces exactly
    the frame it produced before.
    """
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"unknown scenario {scenario_id!r}; available: {sorted(SCENARIOS)}"
        )
    params = ScenarioParams(confounding_strength=confounding_strength)
    dgp = SCENARIOS[scenario_id].build(params)
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


_SUPPORTED_REFERENCE_PARAMS = frozenset({"n_units", "confounding_strength"})


def resolve_scenario_reference(
    reference: ScenarioReference, schema: CausalSchema
) -> CausalDataset:
    """Regenerate the dataset a `ScenarioReference` names.

    Lives here rather than in `cp_core.manifest` beside the reference type
    itself because it needs `build_scenario`, and `.importlinter`'s
    `estimators-cannot-see-truth` contract forbids `cp_core` from importing
    this package.

    Supports only what every `build_scenario(...).generate(...)` call in this
    project's own test suite varies: `n_units` (required) and
    `confounding_strength` (optional). A reference built from a scenario with a
    `negative_control`, `misclassification`, `missingness`, or `outcome_proxy`
    override cannot be reconstructed from those two params alone, and is
    refused rather than silently regenerating a different dataset -- see
    `ScenarioReference`'s own docstring.

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
    confounding_strength = float(
        reference.params.get("confounding_strength", 1.0)  # type: ignore[arg-type]
    )

    dgp = build_scenario(reference.scenario, confounding_strength)
    generated = dgp.generate(n_units=n_units, rng=np.random.default_rng(reference.seed))
    dataset = generated.dataset
    if dataset.schema != schema:
        raise ValueError(
            f"regenerating scenario {reference.scenario!r} produced a schema that does "
            "not match the manifest's recorded schema -- the scenario registry may have "
            "changed since this reference was built"
        )
    return dataset
