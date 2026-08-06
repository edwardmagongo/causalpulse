"""Value types shared across every CausalPulse package."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

BootstrapKind = Literal["iid", "cluster", "block"]
_VALID_BOOTSTRAP_KINDS = frozenset({"iid", "cluster", "block"})


@dataclass(frozen=True)
class Interval:
    """A two-sided interval estimate carrying the level it was computed at."""

    lower: float
    upper: float
    level: float = 0.95

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(
                f"lower bound {self.lower} exceeds upper bound {self.upper}"
            )
        if not 0.0 < self.level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {self.level}")

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class Diagnostic:
    """A quantitative model-fit or data-quality measurement.

    Distinct from an AssumptionCheck: a Diagnostic reports a number, while an
    AssumptionCheck renders a judgement about identification.
    """

    name: str
    value: float | np.ndarray
    unit: str | None
    interpretation: str


@dataclass(frozen=True)
class EstimatorConfig:
    """Per-fit configuration.

    Shared inference settings are explicit so the benchmark harness can hold
    them constant across estimators; estimator-specific keys live in `options`.
    """

    bootstrap_reps: int = 1000
    bootstrap_kind: BootstrapKind = "iid"
    ci_level: float = 0.95
    random_seed: int | None = None
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bootstrap_kind not in _VALID_BOOTSTRAP_KINDS:
            raise ValueError(
                f"bootstrap_kind must be one of {sorted(_VALID_BOOTSTRAP_KINDS)}, "
                f"got {self.bootstrap_kind!r}"
            )
        if self.bootstrap_reps < 1:
            raise ValueError(f"bootstrap_reps must be positive, got {self.bootstrap_reps}")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError(f"ci_level must lie in (0, 1), got {self.ci_level}")
