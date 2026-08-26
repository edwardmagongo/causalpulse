"""Swappable pieces a panel data-generating process is assembled from.

Mirrors the protocol style `CrossSectionDGP` established, so a panel DGP
composes the same way. The generative model is

    Y_it(0) = alpha_i + gamma_t + delta_i * t + X_i . beta + eps_it
    Y_it(1) = Y_it(0) + tau_it
    D_it    = 1[g_i > 0 and t >= g_i]

with treatment absorbing: once treated, always treated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from cp_core.dataset import AssignmentProvenance


class UnitEffects(Protocol):
    def draw(
        self, n_units: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (alpha, delta): unit levels and unit trend slopes."""


class TimePath(Protocol):
    def values(self, n_periods: int, rng: np.random.Generator) -> np.ndarray:
        """Returns gamma_t, the common time path shared by every unit."""


class Adoption(Protocol):
    def cohorts(
        self,
        covariates: np.ndarray,
        unit_slopes: np.ndarray,
        n_periods: int,
        rng: np.random.Generator,
        levels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Returns g_i per unit; 0 means never treated."""

    def violates_parallel_trends(self) -> bool:
        """Whether this mechanism selects on the unit trend slope.

        Reported by the mechanism rather than inferred from its type, following
        the precedent `confounders()` set for the cross-sectional assignment
        protocol: an `isinstance` check would claim parallel trends for any
        future mechanism that violates them without matching the class.
        """

    def provenance(self) -> AssignmentProvenance:
        """How this mechanism decides adoption timing, in the validity score's
        terms.

        Reported by the mechanism rather than inferred from its type, following
        the precedent `confounders()` and `violates_parallel_trends()` set: an
        `isinstance` check would claim a natural experiment for any future
        mechanism that selects on the outcome path without matching the class,
        and claiming that wrongly raises the validity ceiling rather than
        lowering it.
        """


class DynamicEffect(Protocol):
    def tau_matrix(
        self, cohorts: np.ndarray, n_periods: int, covariates: np.ndarray
    ) -> np.ndarray:
        """Returns tau_it, shape (n_units, n_periods)."""


@dataclass(frozen=True)
class NormalUnitEffects:
    """Unit levels and trend slopes, both normal.

    `slope_sd = 0` gives every unit the common time path exactly, which is what
    makes parallel trends hold by construction rather than approximately.
    """

    level_sd: float = 1.0
    slope_sd: float = 0.0

    def __post_init__(self) -> None:
        if self.level_sd < 0 or self.slope_sd < 0:
            raise ValueError(
                f"standard deviations must be non-negative, got level_sd="
                f"{self.level_sd}, slope_sd={self.slope_sd}"
            )

    def draw(
        self, n_units: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = rng.normal(scale=self.level_sd, size=n_units)
        # Drawn as exact zeros rather than from a zero-scale normal so that
        # "no dispersion" is bit-exact rather than merely tiny.
        delta = (
            rng.normal(scale=self.slope_sd, size=n_units)
            if self.slope_sd > 0
            else np.zeros(n_units)
        )
        return alpha, delta


@dataclass(frozen=True)
class LinearTimePath:
    """A common linear time trend, optionally with per-period shocks."""

    slope: float = 0.1
    shock_sd: float = 0.0

    def values(self, n_periods: int, rng: np.random.Generator) -> np.ndarray:
        periods = np.arange(1, n_periods + 1, dtype=float)
        gamma = self.slope * periods
        if self.shock_sd > 0:
            gamma = gamma + rng.normal(scale=self.shock_sd, size=n_periods)
        return np.asarray(gamma)


def _check_start(start: int, n_periods: int) -> None:
    # At least one pre-period and one post-period, or the design is not a DiD.
    if not 2 <= start <= n_periods:
        raise ValueError(
            f"adoption start must lie in [2, {n_periods}] so that at least one "
            f"pre-period and one post-period exist, got {start}"
        )


def _check_periods(periods: tuple[int, ...], n_periods: int) -> None:
    # Same bound as `_check_start`, applied to every period in a cohort
    # schedule; named separately so the error names "periods" rather than
    # "start" when a multi-cohort schedule is the one out of range.
    for start in periods:
        if not 2 <= start <= n_periods:
            raise ValueError(
                f"adoption periods must each lie in [2, {n_periods}] so that at "
                f"least one pre-period and one post-period exist, got {periods}"
            )


@dataclass(frozen=True)
class UniformAdoption:
    """Every treated unit adopts at one period, giving a 2x2 design."""

    start: int = 5
    share: float = 0.5

    def cohorts(
        self,
        covariates: np.ndarray,
        unit_slopes: np.ndarray,
        n_periods: int,
        rng: np.random.Generator,
        levels: np.ndarray | None = None,
    ) -> np.ndarray:
        _check_start(self.start, n_periods)
        n_units = covariates.shape[0]
        treated = rng.random(n_units) < self.share
        return np.where(treated, self.start, 0)

    def violates_parallel_trends(self) -> bool:
        return False

    def provenance(self) -> AssignmentProvenance:
        return "natural_experiment"


@dataclass(frozen=True)
class StaggeredAdoption:
    """Treated units spread evenly over several adoption periods."""

    periods: tuple[int, ...] = (3, 5, 7)
    never_share: float = 0.4

    def cohorts(
        self,
        covariates: np.ndarray,
        unit_slopes: np.ndarray,
        n_periods: int,
        rng: np.random.Generator,
        levels: np.ndarray | None = None,
    ) -> np.ndarray:
        _check_periods(self.periods, n_periods)
        # Annotated explicitly: `covariates.shape[0]` types as `Any`, and
        # passing an `Any` size to the overloaded `rng.random` makes mypy
        # silently pick the `size=None -> float` overload instead of the
        # array one, which then fails at the `draw[treated]` index below.
        n_units: int = covariates.shape[0]
        cohorts = np.zeros(n_units, dtype=int)
        draw = rng.random(n_units)
        treated = draw >= self.never_share
        # Ranking the treated units' own draws spreads them evenly across the
        # cohorts rather than leaving cohort sizes to a second random draw.
        ranks = np.argsort(np.argsort(draw[treated]))
        cohorts[treated] = np.asarray(self.periods)[ranks % len(self.periods)]
        return cohorts

    def violates_parallel_trends(self) -> bool:
        return False

    def provenance(self) -> AssignmentProvenance:
        return "natural_experiment"


@dataclass(frozen=True)
class TrendSelectedAdoption:
    """Steeper-trending units adopt earlier — the parallel-trends violation."""

    periods: tuple[int, ...] = (3, 5, 7)
    never_share: float = 0.4

    def cohorts(
        self,
        covariates: np.ndarray,
        unit_slopes: np.ndarray,
        n_periods: int,
        rng: np.random.Generator,
        levels: np.ndarray | None = None,
    ) -> np.ndarray:
        _check_periods(self.periods, n_periods)
        n_units = covariates.shape[0]
        cohorts = np.zeros(n_units, dtype=int)
        n_treated = int(round(n_units * (1.0 - self.never_share)))
        if n_treated == 0:
            return cohorts
        steepest_first = np.argsort(-unit_slopes)[:n_treated]
        # Chop the steepest units into equal blocks, earliest cohort first.
        block = np.arange(n_treated) * len(self.periods) // n_treated
        cohorts[steepest_first] = np.asarray(self.periods)[block]
        return cohorts

    def violates_parallel_trends(self) -> bool:
        return True

    def provenance(self) -> AssignmentProvenance:
        # Selects on the unit's own trend slope, which the outcome process
        # generates. Observational even though the timing is imposed.
        return "observational"


@dataclass(frozen=True)
class LevelSelectedAdoption:
    """Units with a higher baseline level adopt.

    Load-bearing for the binary scenario: link-function bias in DiD arises only
    when the two arms sit at *different points on the logistic curve*, because
    that is what makes a common logit-scale time shift translate into unequal
    probability-scale changes. With random adoption the arms are
    compositionally identical and the measured bias is indistinguishable from
    zero.
    """

    start: int = 5

    def cohorts(
        self,
        covariates: np.ndarray,
        unit_slopes: np.ndarray,
        n_periods: int,
        rng: np.random.Generator,
        levels: np.ndarray | None = None,
    ) -> np.ndarray:
        _check_start(self.start, n_periods)
        if levels is None:
            raise ValueError(
                "LevelSelectedAdoption requires unit levels; the DGP must pass "
                "`levels=` so selection runs on the baseline rather than on noise"
            )
        return np.where(levels > np.median(levels), self.start, 0)

    def violates_parallel_trends(self) -> bool:
        # Selects on the level, not the slope. On the continuous additive scale
        # the level differences out and trends stay parallel.
        return False

    def provenance(self) -> AssignmentProvenance:
        # Selects on the unit's own baseline level. Note this is observational
        # while `violates_parallel_trends()` is False -- the two answer
        # different questions and are expected to disagree here.
        return "observational"


@dataclass(frozen=True)
class ConstantDynamicEffect:
    """A homogeneous effect, flat over exposure and identical across cohorts."""

    tau: float

    def tau_matrix(
        self, cohorts: np.ndarray, n_periods: int, covariates: np.ndarray
    ) -> np.ndarray:
        return np.full((cohorts.shape[0], n_periods), float(self.tau))


@dataclass(frozen=True)
class RampDynamicEffect:
    """Effect grows linearly with exposure.

    Effects then differ across cohorts at any given calendar period, so
    already-treated units serve as controls with negative weights and two-way
    fixed effects is biased. This is what P4 exists to demonstrate.
    """

    tau: float

    def tau_matrix(
        self, cohorts: np.ndarray, n_periods: int, covariates: np.ndarray
    ) -> np.ndarray:
        periods = np.arange(1, n_periods + 1)
        exposure = periods[None, :] - cohorts[:, None] + 1
        treated = (cohorts[:, None] > 0) & (periods[None, :] >= cohorts[:, None])
        # 0.0 on untreated cells, where exposure is undefined. See the note
        # below on why zero rather than `tau`.
        return np.where(treated, self.tau * exposure, 0.0)
