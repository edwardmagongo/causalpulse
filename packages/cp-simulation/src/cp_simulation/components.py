"""Swappable pieces a data-generating process is assembled from.

Composition rather than ten hardcoded generators, because the benchmark study
needs to vary confounding strength continuously rather than pick from a fixed
menu of scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from cp_core.dataset import AssignmentProvenance
from scipy.special import expit, logit

# Baselines are held away from 0 and 1 so that clipping p0 + tau at the unit
# interval cannot distort a scenario's declared effect.
MIN_BASELINE_PROBABILITY = 0.05
MAX_BASELINE_PROBABILITY = 0.85

# The probability every binary covariate `UnitGenerator` draws is fixed at.
# Named so a scenario's label -- e.g. a heterogeneous effect's population
# mean CATE, which depends on P(x = 1) -- can read the generator's own
# configuration directly instead of estimating it from a generated sample.
BINARY_COVARIATE_PROBABILITY = 0.4


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
                data[name] = rng.binomial(
                    1, BINARY_COVARIATE_PROBABILITY, size=n_units
                ).astype(float)
        return pd.DataFrame(data)


@dataclass(frozen=True)
class LogisticBaseline:
    """Untreated outcome probabilities, bounded away from 0 and 1.

    `noise_scale` adds a standard-normal offset inside the logit -- variation
    the covariates do not explain -- rather than scaling `coefficients`:
    those same covariates drive assignment elsewhere in the DGP, so scaling
    them would move confounding bias at the same time as signal-to-noise,
    which is exactly the non-orthogonal axis the benchmark's one-at-a-time
    grid cannot tolerate. It defaults to 0.0, reproducing
    the pre-noise behaviour exactly.

    `rng` is drawn from only when `noise_scale != 0.0`. The generator is
    shared and sequential across an entire `CrossSectionDGP.generate` call,
    so consuming a draw unconditionally would shift every downstream draw and
    silently change every existing scenario's data at the default.
    """

    intercept: float
    coefficients: tuple[float, ...]
    noise_scale: float = 0.0

    def probabilities(self, covariates: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        beta = np.asarray(self.coefficients, dtype=float)
        if covariates.shape[1] != beta.size:
            raise ValueError(
                f"expected {beta.size} covariates, got {covariates.shape[1]}"
            )
        linear = self.intercept + covariates @ beta
        if self.noise_scale != 0.0:
            linear = linear + self.noise_scale * rng.normal(size=covariates.shape[0])
        raw = sigmoid(linear)
        return np.asarray(np.clip(raw, MIN_BASELINE_PROBABILITY, MAX_BASELINE_PROBABILITY))


@dataclass(frozen=True)
class NegativeControlOutcome:
    """A second outcome sharing the confounders, unaffected by treatment.

    Treatment is not an argument here, and that is the whole construction: no
    configuration of this component can make the column depend on treatment,
    so an estimate that is not null on it points at residual confounding
    rather than at an effect.

    Its own intercept and coefficients rather than the untreated
    potential-outcome surface's, so it shares the confounders without being a
    copy of the outcome, which would make the check trivially self-fulfilling.
    """

    intercept: float
    coefficients: tuple[float, ...]
    column: str = "outcome_nc"

    def probabilities(self, covariates: np.ndarray) -> np.ndarray:
        beta = np.asarray(self.coefficients, dtype=float)
        if covariates.shape[1] != beta.size:
            raise ValueError(
                f"expected {beta.size} covariates, got {covariates.shape[1]}"
            )
        raw = sigmoid(self.intercept + covariates @ beta)
        return np.asarray(np.clip(raw, MIN_BASELINE_PROBABILITY, MAX_BASELINE_PROBABILITY))


@dataclass(frozen=True)
class OutcomeProxy:
    """A noisy proxy for the realised outcome, recorded for every unit.

    Read as a cheap metric collected on everyone while the expensive outcome
    is collected on only some, which is the situation in which imputing an
    outcome is worth doing at all.

    Built from the REALISED outcome rather than from its latent probability,
    and that is the whole construction. A column correlated with the outcome
    only through the covariates carries nothing an analysis model that already
    conditions on them does not have, so multiple imputation without it
    should gain nothing over complete-case -- confirmed by measurement
    through shipped code rather than assumed: paired against complete-case
    (`lin_regression`, MAR, 40 seeds), imputation with NO auxiliary is
    statistically indistinguishable from complete-case (paired diff
    -7.106e-06, MCSE 2.854e-05, RMSE ratio 0.984), while imputation WITH this
    auxiliary beats complete-case with a clear gradient in `noise_sd`: paired
    diff -2.094e-04 (ratio -4.43) at noise_sd=0.25, -1.243e-04 (ratio -2.73)
    at noise_sd=0.5. It also partially rescues MNAR: with an auxiliary built
    from the complete outcome, missingness that depends on the outcome
    becomes partially ignorable given the proxy, repairing roughly two
    thirds of the MNAR bias rather than none of it. See
    `tests/causal/test_phase12_measured.py` for the full protocol and every
    threshold's derivation.

    It is a descendant of the outcome, so no estimator may adjust for it:
    conditioning on a post-outcome variable biases every one of them.
    `CausalSchema` enforces that by refusing an auxiliary that is also a
    covariate; this component only produces the column.
    """

    noise_sd: float
    column: str = "outcome_proxy"

    def __post_init__(self) -> None:
        if self.noise_sd <= 0.0:
            raise ValueError(f"noise_sd must be positive, got {self.noise_sd}")

    def observe(self, outcome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The proxy, from the complete outcome vector.

        The caller passes the complete outcome, not the frame's column: a
        proxy built from a vector with gaps in it would be missing exactly
        where it is needed.
        """
        noise = rng.normal(0.0, self.noise_sd, size=outcome.shape[0])
        return np.asarray(outcome + noise, dtype=float)


@dataclass(frozen=True)
class NonDifferentialMisclassification:
    """Treatment observed with error, at rates that do not depend on the outcome.

    Non-differential is the tractable case and the one modelled here: because
    the error rates are the same whatever the outcome, the effect of the
    observed treatment is an attenuated version of the effect of the real one.
    Rates that did depend on the outcome could manufacture an association
    where none exists, which is a different and harder analysis.
    """

    sensitivity: float
    specificity: float

    def __post_init__(self) -> None:
        for label, value in (
            ("sensitivity", self.sensitivity),
            ("specificity", self.specificity),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1], got {value}")
        if self.sensitivity + self.specificity <= 1.0:
            raise ValueError(
                "sensitivity + specificity must exceed 1; at or below it the "
                "observed label is independent of the true one, which is noise "
                "rather than mismeasurement"
            )

    def observe(self, treatment: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Observed treatment: kept with probability `sensitivity` when truly
        treated, and flipped on with probability `1 - specificity` otherwise."""
        draw = rng.random(treatment.shape[0])
        kept = draw < self.sensitivity
        flipped = draw > self.specificity
        return np.asarray(np.where(treatment == 1, kept, flipped).astype(int))


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

    def provenance(self) -> AssignmentProvenance:
        return "randomised"


@dataclass(frozen=True)
class PropensityAssignment:
    """Assignment driven by covariates, producing confounding.

    `strength` scales the covariate contribution: 0 reduces to random
    assignment, larger values sharpen the dependence and worsen overlap.

    `intercept` shifts the treated share without changing which covariates
    confound: it moves the linear index by a constant, so it raises or lowers
    the overall propensity level but does not alter `confounders()` or
    `provenance()`, both of which depend only on `strength` and
    `coefficients`. It defaults to 0.0, reproducing the pre-intercept
    behaviour exactly.
    """

    coefficients: tuple[float, ...]
    strength: float = 1.0
    intercept: float = 0.0

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
        propensity = sigmoid(self.intercept + self.strength * (covariates @ beta))
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

    def provenance(self) -> AssignmentProvenance:
        # Mirrors `confounders()`: at zero strength the propensity is constant
        # and assignment is independent of everything, so the design really is
        # randomised. Disagreeing with `confounders()` about the same mechanism
        # would put two contradictory claims in one object.
        return "observational" if self.strength != 0.0 else "randomised"


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

    def provenance(self) -> AssignmentProvenance:
        return "observational" if self.strength != 0.0 else "randomised"


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


_MISSINGNESS_MECHANISMS = ("MAR", "MNAR")


def _solve_intercept(z: np.ndarray, strength: float, rate: float) -> float:
    """Intercept `a` with `mean(sigmoid(a + strength*z)) == rate`.

    Strictly increasing in `a`, so bisection converges. Solved rather than
    approximated because the harness varies `strength` while holding the
    marginal rate fixed, and an intercept that drifted with `strength` would
    confound the two.
    """
    low, high = -50.0, 50.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if float(sigmoid(mid + strength * z).mean()) < rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


@dataclass(frozen=True)
class MissingnessMechanism:
    """Outcome missingness at a declared marginal rate.

    `MAR` drives missingness from the covariates, which are observed, so an
    estimator adjusting for them can in principle recover. `MNAR` drives it
    from the outcome itself, which is unobserved exactly where it is missing,
    so no adjustment repairs it -- that is the case the tipping-point bounds
    exist for.

    The intercept is solved so the realised marginal rate matches `rate` at
    any `strength`, which lets a harness vary the dependence while holding
    the amount of missingness fixed. At `strength = 0` every unit shares the
    same probability, which is missingness completely at random.
    """

    mechanism: Literal["MAR", "MNAR"]
    rate: float
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.mechanism not in _MISSINGNESS_MECHANISMS:
            raise ValueError(
                f"mechanism must be one of {list(_MISSINGNESS_MECHANISMS)}, "
                f"got {self.mechanism!r}"
            )
        if not 0.0 < self.rate < 1.0:
            raise ValueError(f"rate must lie in (0, 1), got {self.rate}")
        if self.strength < 0.0:
            raise ValueError(f"strength must be non-negative, got {self.strength}")

    def missing_mask(
        self,
        covariates: np.ndarray,
        outcome: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Boolean mask of the units whose outcome is not observed."""
        driver = (
            covariates.sum(axis=1)
            if self.mechanism == "MAR"
            else outcome.astype(float)
        )
        spread = float(driver.std())
        if spread == 0.0 or self.strength == 0.0:
            # A constant driver carries no information, so the honest
            # behaviour is missingness completely at random at the requested
            # rate rather than an arbitrary tilt.
            probability = np.full(driver.shape[0], self.rate)
        else:
            z = (driver - driver.mean()) / spread
            intercept = _solve_intercept(z, self.strength, self.rate)
            probability = sigmoid(intercept + self.strength * z)
        return np.asarray(rng.random(driver.shape[0]) < probability)
