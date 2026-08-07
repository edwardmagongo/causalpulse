"""Swappable pieces a data-generating process is assembled from.

Composition rather than ten hardcoded generators, because the benchmark study
needs to vary confounding strength continuously rather than pick from a fixed
menu of scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

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
        return np.clip(raw, MIN_BASELINE_PROBABILITY, MAX_BASELINE_PROBABILITY)


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
        self, covariates: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        n = covariates.shape[0]
        propensity = np.full(n, self.share)
        return rng.binomial(1, propensity), propensity


@dataclass(frozen=True)
class PropensityAssignment:
    """Assignment driven by covariates, producing confounding.

    `strength` scales the covariate contribution: 0 reduces to random
    assignment, larger values sharpen the dependence and worsen overlap.
    """

    coefficients: tuple[float, ...]
    strength: float = 1.0

    def assign(
        self, covariates: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        beta = np.asarray(self.coefficients, dtype=float)
        if covariates.shape[1] != beta.size:
            raise ValueError(
                f"expected {beta.size} covariates, got {covariates.shape[1]}"
            )
        propensity = sigmoid(self.strength * (covariates @ beta))
        return rng.binomial(1, propensity), propensity
