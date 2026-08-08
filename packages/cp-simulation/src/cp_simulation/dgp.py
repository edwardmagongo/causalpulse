"""Assembly of components into a complete data-generating process."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from cp_core.dataset import CausalDataset, CausalSchema

from cp_simulation.components import (
    LogisticBaseline,
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

    @property
    def config_hash(self) -> str:
        payload = {
            "scenario_id": self.scenario_id,
            "covariates": list(self.unit_generator.covariate_names),
            "baseline": asdict(self.baseline),
            "effect": {type(self.effect).__name__: asdict(self.effect)},  # type: ignore[call-overload]
            "assignment": {type(self.assignment).__name__: asdict(self.assignment)},  # type: ignore[call-overload]
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

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

        frame = frame.assign(treated=treatment, outcome=outcome)
        schema = CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=names,
            outcome_type="binary",
        )
        truth = GroundTruth(
            true_ate=float(cate.mean()),
            true_att=true_att,
            true_cate=cate,
            true_propensity=propensity,
            true_confounders=self.assignment.confounders(names),
            scenario_id=self.scenario_id,
            config_hash=self.config_hash,
        )
        return SimulatedDataset(
            dataset=CausalDataset(frame=frame, schema=schema), truth=truth
        )
