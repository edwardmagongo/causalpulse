"""Swappable pieces a data-generating process is assembled from.

Composition rather than ten hardcoded generators, because the benchmark study
needs to vary confounding strength continuously rather than pick from a fixed
menu of scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit

# Baselines are held away from 0 and 1 so that clipping p0 + tau at the unit
# interval cannot distort a scenario's declared effect. See spec section 4.3.
MIN_BASELINE_PROBABILITY = 0.05
MAX_BASELINE_PROBABILITY = 0.85


def sigmoid(z: np.ndarray) -> np.ndarray:
    return expit(z)


@dataclass(frozen=True)
class UnitGenerator:
    """Draws unit identifiers and baseline covariates."""

    covariate_names: tuple[str, ...]

    def generate(self, n_units: int, rng: np.random.Generator) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {"unit_id": np.arange(n_units)}
        for index, name in enumerate(self.covariate_names):
            # Alternate continuous and binary covariates so balance diagnostics
            # and matching are exercised on both kinds.
            if index % 2 == 0:
                data[name] = rng.normal(size=n_units)
            else:
                data[name] = rng.binomial(1, 0.4, size=n_units).astype(float)
        return pd.DataFrame(data)


@dataclass(frozen=True)
class LogisticBaseline:
    """Untreated outcome probabilities, bounded away from 0 and 1."""

    intercept: float
    coefficients: tuple[float, ...]

    def probabilities(self, covariates: np.ndarray) -> np.ndarray:
        beta = np.asarray(self.coefficients, dtype=float)
        if covariates.shape[1] != beta.size:
            raise ValueError(
                f"expected {beta.size} covariates, got {covariates.shape[1]}"
            )
        raw = sigmoid(self.intercept + covariates @ beta)
        return np.asarray(np.clip(raw, MIN_BASELINE_PROBABILITY, MAX_BASELINE_PROBABILITY))


@dataclass(frozen=True)
class ConstantEffect:
    """A homogeneous treatment effect on the probability scale."""

    tau: float

    def tau_values(self, covariates: np.ndarray) -> np.ndarray:
        return np.full(covariates.shape[0], self.tau)


@dataclass(frozen=True)
class HeterogeneousEffect:
    """Effect that increases for units whose indicator covariate equals one."""

    base_tau: float
    covariate_index: int
    increment: float

    def tau_values(self, covariates: np.ndarray) -> np.ndarray:
        width = covariates.shape[1]
        if self.covariate_index < 0 or self.covariate_index >= width:
            raise ValueError(
                f"covariate_index must lie in [0, {width}), got {self.covariate_index}"
            )
        indicator = covariates[:, self.covariate_index]
        return self.base_tau + self.increment * indicator


@dataclass(frozen=True)
class RandomAssignment:
    """Bernoulli assignment independent of everything else."""

    share: float = 0.5

    def assign(
        self,
        covariates: np.ndarray,
        baseline: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = covariates.shape[0]
        propensity = np.full(n, self.share)
        return rng.binomial(1, propensity), propensity

    def confounders(self, covariate_names: tuple[str, ...]) -> tuple[str, ...]:
        """No covariate influences assignment, so nothing confounds."""
        return ()


@dataclass(frozen=True)
class PropensityAssignment:
    """Assignment driven by covariates, producing confounding.

    `strength` scales the covariate contribution: 0 reduces to random
    assignment, larger values sharpen the dependence and worsen overlap.
    """

    coefficients: tuple[float, ...]
    strength: float = 1.0

    def assign(
        self,
        covariates: np.ndarray,
        baseline: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        beta = np.asarray(self.coefficients, dtype=float)
        if covariates.shape[1] != beta.size:
            raise ValueError(
                f"expected {beta.size} covariates, got {covariates.shape[1]}"
            )
        propensity = sigmoid(self.strength * (covariates @ beta))
        return rng.binomial(1, propensity), propensity

    def confounders(self, covariate_names: tuple[str, ...]) -> tuple[str, ...]:
        """Name the covariates this mechanism actually assigns on.

        Measured rather than declared: at `strength == 0` the propensity
        collapses to a constant and no covariate influences assignment at all,
        and a zero coefficient drops its covariate out of the linear index. Both
        cases must report honestly, because a covariate wrongly named a
        confounder and a confounder wrongly omitted are equally misleading.
        """
        if self.strength == 0.0:
            return ()
        return tuple(
            name
            for name, coefficient in zip(covariate_names, self.coefficients, strict=True)
            if coefficient != 0.0
        )


@dataclass(frozen=True)
class BaselineSelectionAssignment:
    """Assignment driven by a unit's own untreated outcome probability.

    Qualitatively different from PropensityAssignment: selection runs through
    the outcome-generating process rather than through a covariate an analyst
    would naturally think to adjust for. The effect stays identifiable here
    because the baseline is a function of the covariates, but a method that
    ignores them is badly biased.
    """

    strength: float = 8.0
    share: float = 0.5

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise ValueError(f"strength must be non-negative, got {self.strength}")
        if not 0.0 < self.share < 1.0:
            raise ValueError(f"share must lie in (0, 1), got {self.share}")

    def assign(
        self,
        covariates: np.ndarray,
        baseline: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if baseline.shape[0] != covariates.shape[0]:
            raise ValueError(
                f"baseline has {baseline.shape[0]} entries but covariates has "
                f"{covariates.shape[0]} rows"
            )
        # `share` sets the intercept so it actually controls the treated
        # fraction: centring the baseline makes the strength term average zero,
        # so the mean propensity lands near `share`, and at strength 0 it is
        # exactly `share`. logit(0.5) is 0, so the default is unchanged.
        centred = baseline - float(np.mean(baseline))
        propensity = expit(float(logit(self.share)) + self.strength * centred)
        return rng.binomial(1, propensity), propensity

    def confounders(self, covariate_names: tuple[str, ...]) -> tuple[str, ...]:
        # Selection runs through the baseline, which every covariate feeds, so
        # all of them confound. At zero strength assignment is independent of
        # everything and nothing confounds.
        if self.strength == 0.0:
            return ()
        return covariate_names


@dataclass(frozen=True)
class MultivariateHeterogeneousEffect:
    """Effect driven by an interaction between a continuous and a binary
    covariate, plus a smaller independent continuous driver.

    `HeterogeneousEffect` varies tau along exactly one binary covariate --
    too simple to meaningfully exercise an estimator whose entire purpose is
    discovering heterogeneity structure the analyst didn't specify (a single
    interaction term recovers a one-covariate step function as well as a
    forest would). This component gives Phase 6's CATE scenarios genuine
    multi-covariate structure: an interaction effect that only fires when a
    continuous covariate is positive AND a binary covariate is 1, plus an
    independent, clipped linear effect from a second continuous covariate.
    """

    base_tau: float
    continuous_index: int
    continuous_coefficient: float
    binary_interaction_index: int
    binary_interaction_continuous_index: int
    binary_interaction_coefficient: float
    secondary_continuous_index: int
    secondary_coefficient: float
    secondary_clip: float

    def tau_values(self, covariates: np.ndarray) -> np.ndarray:
        width = covariates.shape[1]
        for index in (
            self.continuous_index,
            self.binary_interaction_index,
            self.binary_interaction_continuous_index,
            self.secondary_continuous_index,
        ):
            if index < 0 or index >= width:
                raise ValueError(f"covariate index {index} out of range for width {width}")

        continuous = covariates[:, self.continuous_index]
        binary = covariates[:, self.binary_interaction_index]
        interaction_driver = covariates[:, self.binary_interaction_continuous_index]
        secondary = np.clip(
            covariates[:, self.secondary_continuous_index],
            -self.secondary_clip,
            self.secondary_clip,
        )

        interaction = self.binary_interaction_coefficient * (interaction_driver > 0) * binary
        return np.asarray(
            self.base_tau
            + self.continuous_coefficient * continuous
            + interaction
            + self.secondary_coefficient * secondary
        )
