"""Assembly of components into a complete data-generating process."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from cp_core.dataset import AssignmentProvenance, CausalDataset, CausalSchema

from cp_simulation._hashing import component_hash, describe_components
from cp_simulation.components import (
    LogisticBaseline,
    MissingnessMechanism,
    NegativeControlOutcome,
    NonDifferentialMisclassification,
    UnitGenerator,
)
from cp_simulation.truth import GroundTruth, SimulatedDataset


class EffectFunction(Protocol):
    # Named `tau_values` rather than `tau` so that implementations may expose a
    # `tau` dataclass field without the field shadowing the method.
    def tau_values(self, covariates: np.ndarray) -> np.ndarray: ...


class Assignment(Protocol):
    def assign(
        self,
        covariates: np.ndarray,
        baseline: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    # Each mechanism names its own confounders. Inferring them from the
    # mechanism's type instead would silently report `()` — an unconfoundedness
    # claim — for any mechanism the caller had not been taught about.
    def confounders(self, covariate_names: tuple[str, ...]) -> tuple[str, ...]: ...

    def provenance(self) -> AssignmentProvenance:
        """How this mechanism assigns treatment, in the validity score's terms.

        Reported by the mechanism rather than inferred from its type, following
        the precedent `confounders()` set: an `isinstance` check would claim
        randomisation for any future mechanism that confounds without matching
        the class, and claiming randomisation wrongly raises the validity
        ceiling rather than lowering it.
        """


@dataclass(frozen=True)
class CrossSectionDGP:
    """A cross-sectional data-generating process with computed ground truth.

    Potential outcome probabilities are built explicitly as p0 and p1, the true
    effects are measured from them, and only the realised outcome reaches the
    dataset handed to estimators.
    """

    unit_generator: UnitGenerator
    baseline: LogisticBaseline
    effect: EffectFunction
    assignment: Assignment
    scenario_id: str
    negative_control: NegativeControlOutcome | None = None
    misclassification: NonDifferentialMisclassification | None = None
    missingness: MissingnessMechanism | None = None

    @property
    def config_hash(self) -> str:
        return component_hash(
            {
                "scenario_id": self.scenario_id,
                "covariates": list(self.unit_generator.covariate_names),
                "baseline": asdict(self.baseline),
                **describe_components(
                    effect=self.effect,
                    assignment=self.assignment,
                    negative_control=self.negative_control,
                    misclassification=self.misclassification,
                    missingness=self.missingness,
                ),
            }
        )

    def generate(self, n_units: int, rng: np.random.Generator) -> SimulatedDataset:
        if n_units <= 0:
            raise ValueError(f"n_units must be positive, got {n_units}")

        names = self.unit_generator.covariate_names
        frame = self.unit_generator.generate(n_units, rng)
        covariates = frame[list(names)].to_numpy()

        p0 = self.baseline.probabilities(covariates)
        tau = self.effect.tau_values(covariates)
        p1 = np.clip(p0 + tau, 0.0, 1.0)

        treatment, propensity = self.assignment.assign(covariates, p0, rng)
        realised_probability = np.where(treatment == 1, p1, p0)
        outcome = rng.binomial(1, realised_probability)

        # Truth is measured from the realised potential outcomes, never copied
        # from the tau parameter: clipping at the unit interval would otherwise
        # make the recorded effect disagree with the actual one.
        cate = p1 - p0
        treated_mask = treatment == 1
        true_att = float(cate[treated_mask].mean()) if treated_mask.any() else float("nan")

        # Outcomes are realised from the TRUE treatment above; only the
        # observed column is mismeasured. That ordering is what keeps
        # `true_ate` the effect of real treatment, so an estimator fitted on
        # the observed column is genuinely attenuated rather than attenuated
        # by definition.
        true_treatment: np.ndarray | None = None
        observed_treatment = treatment
        if self.misclassification is not None:
            true_treatment = treatment
            observed_treatment = self.misclassification.observe(treatment, rng)

        # The outcome is realised above; missingness is applied only to what
        # the frame carries. The complete vector goes to GroundTruth, so a
        # harness has something to measure complete-case bias against.
        complete_outcome: np.ndarray | None = None
        # Cast unconditionally, not only when a mechanism is configured. It
        # makes the outcome column float64 on every scenario rather than the
        # int64 it was before, which is a real change -- but a dtype that
        # depended on whether missingness happened to be configured would be
        # worse, and only float can carry the NaN. Verified: values stay
        # exactly 0/1, nothing in the codebase branches on this column's
        # dtype, and `CausalDataset`'s binary check compares against `{0, 1}`
        # by value, which floats satisfy.
        observed_outcome = outcome.astype(float)
        if self.missingness is not None:
            complete_outcome = observed_outcome.copy()
            hidden = self.missingness.missing_mask(covariates, observed_outcome, rng)
            observed_outcome = observed_outcome.copy()
            observed_outcome[hidden] = np.nan

        frame = frame.assign(treated=observed_treatment, outcome=observed_outcome)
        if self.negative_control is not None:
            nc_probability = self.negative_control.probabilities(covariates)
            frame = frame.assign(
                **{self.negative_control.column: rng.binomial(1, nc_probability)}
            )
        schema = CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=names,
            outcome_type="binary",
            assignment_provenance=self.assignment.provenance(),
            negative_control_outcome=(
                self.negative_control.column if self.negative_control is not None else None
            ),
        )
        truth = GroundTruth(
            true_ate=float(cate.mean()),
            true_att=true_att,
            true_cate=cate,
            true_propensity=propensity,
            true_confounders=self.assignment.confounders(names),
            scenario_id=self.scenario_id,
            config_hash=self.config_hash,
            true_treatment=true_treatment,
            complete_outcome=complete_outcome,
        )
        return SimulatedDataset(
            dataset=CausalDataset(frame=frame, schema=schema), truth=truth
        )
