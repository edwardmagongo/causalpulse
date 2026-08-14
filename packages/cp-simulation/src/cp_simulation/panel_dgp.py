"""Assembly of panel components into a complete data-generating process."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from cp_core.dataset import CausalDataset, CausalSchema

from cp_simulation._hashing import component_hash, describe_components
from cp_simulation.components import UnitGenerator
from cp_simulation.panel_components import (
    Adoption,
    DynamicEffect,
    TimePath,
    UnitEffects,
)
from cp_simulation.panel_truth import PanelGroundTruth, SimulatedPanel


@dataclass(frozen=True)
class PanelDGP:
    """A panel data-generating process with computed ground truth.

    Potential outcomes are built explicitly as Y(0) and Y(1) = Y(0) + tau, the
    true effects are measured from them, and only the realised outcome reaches
    the dataset handed to estimators.
    """

    unit_generator: UnitGenerator
    unit_effects: UnitEffects
    time_path: TimePath
    adoption: Adoption
    effect: DynamicEffect
    scenario_id: str
    noise_sd: float = 1.0
    error_rho: float = 0.0
    covariate_coefficients: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.noise_sd < 0:
            raise ValueError(f"noise_sd must be non-negative, got {self.noise_sd}")
        if not -1.0 < self.error_rho < 1.0:
            raise ValueError(
                f"error_rho must lie in (-1, 1) for a stationary AR(1) process, "
                f"got {self.error_rho}"
            )

    @property
    def config_hash(self) -> str:
        return component_hash(
            {
                "scenario_id": self.scenario_id,
                "covariates": list(self.unit_generator.covariate_names),
                "noise_sd": self.noise_sd,
                "error_rho": self.error_rho,
                "covariate_coefficients": list(self.covariate_coefficients),
                **describe_components(
                    unit_effects=self.unit_effects,
                    time_path=self.time_path,
                    adoption=self.adoption,
                    effect=self.effect,
                ),
            }
        )

    def _errors(
        self, n_units: int, n_periods: int, rng: np.random.Generator
    ) -> np.ndarray:
        eps = rng.normal(scale=self.noise_sd, size=(n_units, n_periods))
        if self.error_rho == 0.0:
            return eps
        # Scaled so the marginal variance stays noise_sd^2 whatever rho is;
        # otherwise raising rho would confound serial correlation with a larger
        # error, and the clustering test could pass for the wrong reason.
        scale = np.sqrt(1.0 - self.error_rho**2)
        for t in range(1, n_periods):
            eps[:, t] = self.error_rho * eps[:, t - 1] + scale * eps[:, t]
        return eps

    def generate(
        self, n_units: int, n_periods: int, rng: np.random.Generator
    ) -> SimulatedPanel:
        if n_units <= 0:
            raise ValueError(f"n_units must be positive, got {n_units}")
        if n_periods < 2:
            raise ValueError(
                f"n_periods must be at least 2 for a difference to exist, "
                f"got {n_periods}"
            )

        names = self.unit_generator.covariate_names
        units = self.unit_generator.generate(n_units, rng)
        covariates = units[list(names)].to_numpy(dtype=float)

        alpha, delta = self.unit_effects.draw(n_units, rng)
        gamma = self.time_path.values(n_periods, rng)
        periods = np.arange(1, n_periods + 1, dtype=float)

        if self.covariate_coefficients:
            beta = np.asarray(self.covariate_coefficients, dtype=float)
            if beta.size != covariates.shape[1]:
                raise ValueError(
                    f"expected {covariates.shape[1]} covariate coefficients, "
                    f"got {beta.size}"
                )
            covariate_term = covariates @ beta
        else:
            covariate_term = np.zeros(n_units)

        # `levels` is the unit's own baseline level, passed so that a
        # level-selecting mechanism selects on the baseline rather than on noise.
        cohorts = self.adoption.cohorts(
            covariates, delta, n_periods, rng, levels=alpha + covariate_term
        ).astype(int)

        y0 = (
            alpha[:, None]
            + covariate_term[:, None]
            + gamma[None, :]
            + delta[:, None] * periods[None, :]
            + self._errors(n_units, n_periods, rng)
        )
        tau = self.effect.tau_matrix(cohorts, n_periods, covariates)
        treated = (cohorts[:, None] > 0) & (periods[None, :] >= cohorts[:, None])
        outcome = y0 + np.where(treated, tau, 0.0)

        truth = self._measure(cohorts, tau, treated, y0, periods)
        frame = self._to_long(units, names, cohorts, treated, outcome, periods)
        schema = CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=names,
            outcome_type="continuous",
            time="period",
            cohort="cohort",
        )
        return SimulatedPanel(
            dataset=CausalDataset(frame=frame, schema=schema), truth=truth
        )

    def _measure(
        self,
        cohorts: np.ndarray,
        tau: np.ndarray,
        treated: np.ndarray,
        y0: np.ndarray,
        periods: np.ndarray,
    ) -> PanelGroundTruth:
        """Every quantity averaged over realised potential outcomes."""
        att_gt: dict[tuple[int, int], float] = {}
        for cohort in sorted({int(c) for c in cohorts if c > 0}):
            members = cohorts == cohort
            for index, period in enumerate(periods.astype(int)):
                if period < cohort:
                    continue
                att_gt[(cohort, int(period))] = float(tau[members, index].mean())

        by_cohort: dict[int, float] = {}
        for cohort in sorted({int(c) for c in cohorts if c > 0}):
            members = cohorts == cohort
            cells = treated[members]
            by_cohort[cohort] = float(tau[members][cells].mean())

        by_event_time: dict[int, float] = {}
        exposure = periods.astype(int)[None, :] - cohorts[:, None]
        for horizon in sorted({int(h) for h in exposure[treated]}):
            mask = treated & (exposure == horizon)
            by_event_time[horizon] = float(tau[mask].mean())

        return PanelGroundTruth(
            true_att_gt=att_gt,
            true_att_overall=(
                float(tau[treated].mean()) if treated.any() else float("nan")
            ),
            true_att_by_cohort=by_cohort,
            true_att_by_event_time=by_event_time,
            true_effect_matrix=np.where(treated, tau, 0.0),
            true_cohorts=cohorts,
            true_pre_trend_gap=pre_trend_gap(y0, cohorts, periods),
            scenario_id=self.scenario_id,
            config_hash=self.config_hash,
        )

    def _to_long(
        self,
        units: pd.DataFrame,
        names: tuple[str, ...],
        cohorts: np.ndarray,
        treated: np.ndarray,
        outcome: np.ndarray,
        periods: np.ndarray,
    ) -> pd.DataFrame:
        n_units, n_periods = outcome.shape
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
        return frame


def pre_trend_gap(
    y0: np.ndarray, cohorts: np.ndarray, periods: np.ndarray
) -> float:
    """Measured Y(0) trend-slope gap, ever-treated minus never-treated.

    Measured from the untreated potential outcomes rather than declared, so
    parallel trends is an observable property of a scenario rather than a claim
    about it — the precedent `confounders()` set for the cross-sectional DGP.
    Returns nan when either arm is empty, since there is then no gap to report.
    """
    ever = cohorts > 0
    if not ever.any() or ever.all():
        return float("nan")
    centred = periods - periods.mean()
    denominator = float((centred**2).sum())
    slopes = (y0 * centred[None, :]).sum(axis=1) / denominator
    return float(slopes[ever].mean() - slopes[~ever].mean())
