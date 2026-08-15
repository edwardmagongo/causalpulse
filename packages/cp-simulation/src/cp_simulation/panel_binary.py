"""A binary-outcome panel, with the effect applied on the logit scale.

Separate from `PanelDGP` rather than a mode of it: the potential outcomes are
built through a link function, truth lives on a different scale from the input
parameter, and the scenario exists to *measure a bias* rather than to be
recovered cleanly. Folding both into one class would put a branch through the
middle of the most safety-critical method in the simulation package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from cp_core.dataset import CausalDataset, CausalSchema
from scipy.special import expit

from cp_simulation._hashing import component_hash, describe_components
from cp_simulation.components import UnitGenerator
from cp_simulation.panel_components import Adoption, TimePath, UnitEffects
from cp_simulation.panel_truth import (
    PanelGroundTruth,
    SimulatedPanel,
    measure_panel_truth,
)


@dataclass(frozen=True)
class BinaryPanelDGP:
    """Binary outcomes with parallel trends holding on the logit scale.

    Because the effect is applied as `expit(index + tau_logit)` rather than
    `clip(p0 + tau, 0, 1)`, the result is inside [0, 1] by construction: there
    is no additive clip and therefore no clip-induced distortion. The
    `[0.05, 0.85]` baseline bound that governs the cross-sectional scenarios is
    an artefact of that clip and does not apply here.
    """

    unit_generator: UnitGenerator
    unit_effects: UnitEffects
    time_path: TimePath
    adoption: Adoption
    scenario_id: str
    intercept: float = -1.5
    tau_logit: float = 0.45

    @property
    def config_hash(self) -> str:
        return component_hash(
            {
                "scenario_id": self.scenario_id,
                "covariates": list(self.unit_generator.covariate_names),
                "intercept": self.intercept,
                "tau_logit": self.tau_logit,
                **describe_components(
                    unit_effects=self.unit_effects,
                    time_path=self.time_path,
                    adoption=self.adoption,
                ),
            }
        )

    def generate(
        self, n_units: int, n_periods: int, rng: np.random.Generator
    ) -> SimulatedPanel:
        if n_units <= 0:
            raise ValueError(f"n_units must be positive, got {n_units}")
        if n_periods < 2:
            raise ValueError(f"n_periods must be at least 2, got {n_periods}")

        names = self.unit_generator.covariate_names
        units = self.unit_generator.generate(n_units, rng)
        covariates = units[list(names)].to_numpy(dtype=float)

        alpha, delta = self.unit_effects.draw(n_units, rng)
        gamma = self.time_path.values(n_periods, rng)
        periods = np.arange(1, n_periods + 1, dtype=float)

        index = self.intercept + alpha[:, None] + gamma[None, :]
        p0 = np.asarray(expit(index))
        p1 = np.asarray(expit(index + self.tau_logit))

        cohorts = self.adoption.cohorts(
            covariates, delta, n_periods, rng, levels=alpha
        ).astype(int)
        treated = (cohorts[:, None] > 0) & (periods[None, :] >= cohorts[:, None])

        realised = np.where(treated, p1, p0)
        outcome = rng.binomial(1, realised).astype(float)

        # Truth on the probability scale, measured from the realised potential
        # outcomes. This is not `tau_logit` and cannot be confused with it.
        effect = p1 - p0
        truth = self._measure(cohorts, effect, treated, p0, periods)

        frame = pd.DataFrame(
            {
                "unit_id": np.repeat(units["unit_id"].to_numpy(), n_periods),
                "period": np.tile(periods.astype(int), n_units),
                "cohort": np.repeat(cohorts, n_periods),
                "treated": treated.astype(int).ravel(),
                "outcome": outcome.ravel(),
            }
        )
        for name in names:
            frame[name] = np.repeat(units[name].to_numpy(), n_periods)

        schema = CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=names,
            outcome_type="binary",
            time="period",
            cohort="cohort",
        )
        return SimulatedPanel(
            dataset=CausalDataset(frame=frame, schema=schema), truth=truth
        )

    def _measure(
        self,
        cohorts: np.ndarray,
        effect: np.ndarray,
        treated: np.ndarray,
        p0: np.ndarray,
        periods: np.ndarray,
    ) -> PanelGroundTruth:
        """`effect` is `p1 - p0`, the probability-scale gap; `p0` is the
        untreated probability path the pre-trend gap is measured against, so
        the reported gap is on the same scale as the effect the estimators
        report. The aggregation itself is shared with `PanelDGP`, whose
        `_measure` works in `tau` space instead -- see `measure_panel_truth`.
        """
        return measure_panel_truth(
            cohorts=cohorts,
            effect_matrix=effect,
            treated=treated,
            untreated_path=p0,
            periods=periods,
            scenario_id=self.scenario_id,
            config_hash=self.config_hash,
        )
