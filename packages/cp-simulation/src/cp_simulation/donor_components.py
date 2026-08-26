"""Adoption and unit-effects components for donor-pool scenarios.

Donor-pool scenarios reuse `PanelDGP` (`panel_dgp.py`) unchanged -- see the
design doc's §3.1. These are the pieces `PanelDGP`'s existing protocol style
(`panel_components.py`) does not provide: deterministic single-unit
selection, and two ways to make that one unit diverge from its donor pool in
a controlled, specific dimension rather than by however extreme its own
random draw happens to be.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cp_core.dataset import AssignmentProvenance

from cp_simulation.panel_components import UnitEffects


@dataclass(frozen=True)
class SingleUnitAdoption:
    """Exactly one unit adopts, chosen by a fixed index rather than a draw.

    `UniformAdoption`/`StaggeredAdoption` are per-unit coin flips; a `share`
    tuned toward `1 / n_units` does not reliably produce exactly one treated
    unit, and "reliably" is not optional when a scenario's ground truth
    assumes a single-cohort panel.
    """

    start: int
    unit_index: int = 0

    def cohorts(
        self,
        covariates: np.ndarray,
        unit_slopes: np.ndarray,
        n_periods: int,
        rng: np.random.Generator,
        levels: np.ndarray | None = None,
    ) -> np.ndarray:
        n_units = covariates.shape[0]
        if not 0 <= self.unit_index < n_units:
            raise ValueError(f"unit_index {self.unit_index} is out of range for {n_units} units")
        cohorts = np.zeros(n_units, dtype=int)
        cohorts[self.unit_index] = self.start
        return cohorts

    def violates_parallel_trends(self) -> bool:
        return False

    def provenance(self) -> AssignmentProvenance:
        return "natural_experiment"


@dataclass(frozen=True)
class ClusteredUnitEffects:
    """Two slope clusters; the treated unit sits in cluster A.

    Gives the donor pool a genuine subset the treated unit resembles
    (cluster A) and a subset it does not (cluster B), which is what makes a
    donor-weighting scheme's choice of *which* donors to weight actually
    matter: an equal-weighted average of the whole pool mixes both clusters
    and is biased toward cluster B's mean; a scheme that concentrates weight
    on cluster A is not. Measured (design doc §2 item 5): at
    `n_similar=9, within_sd=0.02`, synthetic control's bias is 0.17 against a
    naive average's 4.42, on the same 17-donor, 400-seed protocol.
    """

    n_similar: int  # cluster A size, including the treated unit
    mean_a: float
    mean_b: float
    within_sd: float
    level_sd: float = 0.3
    treated_index: int = 0

    def __post_init__(self) -> None:
        if self.n_similar < 1:
            raise ValueError(f"n_similar must be at least 1, got {self.n_similar}")
        if not 0 <= self.treated_index < self.n_similar:
            raise ValueError(
                f"treated_index {self.treated_index} must lie inside cluster A "
                f"(the first {self.n_similar} units) so the treated unit is "
                "actually a cluster-A member"
            )

    def draw(self, n_units: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.n_similar >= n_units:
            raise ValueError(
                f"n_similar ({self.n_similar}) must leave at least one unit "
                f"for cluster B, got n_units={n_units}"
            )
        alpha = rng.normal(scale=self.level_sd, size=n_units)
        delta = np.empty(n_units)
        delta[: self.n_similar] = rng.normal(self.mean_a, self.within_sd, self.n_similar)
        delta[self.n_similar :] = rng.normal(self.mean_b, self.within_sd, n_units - self.n_similar)
        return alpha, delta


@dataclass(frozen=True)
class LevelOffsetUnitEffects:
    """Wraps another `UnitEffects`, adding a fixed level offset to one unit.

    Gives the chosen unit a baseline level that lies outside the donor
    pool's range by a fixed, deterministic amount every replication --
    guaranteeing a convex-hull violation, rather than relying on that unit
    happening to be the extreme draw among the pool, which is unreliable at
    realistic donor-pool sizes: measured (design doc §2 item 5) at
    `level_offset=1.0`, a 17-donor pool still produced a higher pre-period
    mean than the offset unit 80% of the time. `level_offset=5.0` is
    reliable: 300/300 seeds.
    """

    base: UnitEffects
    unit_index: int
    level_offset: float

    def draw(self, n_units: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= self.unit_index < n_units:
            raise ValueError(f"unit_index {self.unit_index} is out of range for {n_units} units")
        alpha, delta = self.base.draw(n_units, rng)
        alpha = alpha.copy()
        alpha[self.unit_index] += self.level_offset
        return alpha, delta
