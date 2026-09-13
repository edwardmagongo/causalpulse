"""The knobs a scenario's data-generating process exposes to the grid.

One frozen object rather than a widening positional signature, because the
same field set has to be readable from three places -- the scenario
registry, the benchmark grid, and a persisted manifest that `replay` reads
back -- and a signature plus a frozenset of permitted keys would be three
lists to keep in sync by hand.

Every default reproduces the behaviour this project had before these axes
existed. That is load-bearing rather than tidy: `resolve_scenario_reference`
fills a pre-Phase-18 manifest's missing keys from these defaults, so
changing one silently changes what an old manifest replays as.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioParams:
    """What the grid varies. `n_units` is deliberately not here -- it belongs
    to `generate(n_units=...)`, not to building the process."""

    confounding_strength: float = 1.0
    treatment_intercept: float = 0.0
    noise_scale: float = 0.0
    effect_scale: float = 1.0
    heterogeneity_scale: float = 1.0


AXIS_NAMES: tuple[str, ...] = (
    "confounding_strength",
    "treatment_intercept",
    "noise_scale",
    "effect_scale",
    "heterogeneity_scale",
)
