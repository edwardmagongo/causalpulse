"""Turning a cell into the data it names, and the truth to score against.

The truth mapping is keyed by **estimand**, not by a single scalar, because
the estimators in a single cell do not all target the same quantity: every
cross-sectional estimator in scope reports `ATE` and every panel estimator
reports `ATT`. Scoring an ATT estimate against ATE truth would manufacture
bias that is not there, so the mapping is total over what a cell actually has
and raises on what it does not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cp_core.dataset import CausalDataset
from cp_simulation.oracle import oracle_ate, oracle_att
from cp_simulation.panel_scenarios import build_panel_scenario
from cp_simulation.params import ScenarioParams
from cp_simulation.scenarios import build_scenario

from cp_evaluation.grid import Cell, CrossSectionCell, cell_rng


@dataclass(frozen=True, eq=False)
class ResolvedCell:
    """A cell's generated dataset alongside its measured truth.

    eq=False mirrors `CausalDataset`, which declares it because a frame does
    not compare structurally.
    """

    dataset: CausalDataset
    truth: Mapping[str, float]


def resolve_cell(cell: Cell, root_entropy: int) -> ResolvedCell:
    """Generate a cell's dataset and measure its ground truth.

    Truth comes from the oracle functions, which read the realised
    potential-outcome matrices. It is never taken from `declared_effect`,
    which is a label attached to the scenario rather than a property of the
    sample this cell actually drew.
    """
    rng = cell_rng(cell, root_entropy)

    if isinstance(cell, CrossSectionCell):
        # Routed through `ScenarioParams` explicitly, not the bare-float
        # facade: `CrossSectionCell` now carries all four axes Phase 18
        # adds, alongside `confounding_strength`, and `build_scenario` is
        # the one entry point every caller goes through, never
        # `SCENARIOS[id].build` directly.
        dgp = build_scenario(
            cell.scenario,
            ScenarioParams(
                confounding_strength=cell.confounding_strength,
                treatment_intercept=cell.treatment_intercept,
                noise_scale=cell.noise_scale,
                effect_scale=cell.effect_scale,
                heterogeneity_scale=cell.heterogeneity_scale,
            ),
        )
        simulated = dgp.generate(n_units=cell.n_units, rng=rng)
        return ResolvedCell(
            dataset=simulated.dataset,
            truth={"ATE": oracle_ate(simulated), "ATT": oracle_att(simulated)},
        )

    panel_dgp = build_panel_scenario(cell.scenario)
    panel = panel_dgp.generate(n_units=cell.n_units, n_periods=cell.n_periods, rng=rng)
    return ResolvedCell(
        dataset=panel.dataset,
        truth={"ATT": float(panel.truth.true_att_overall)},
    )


def truth_for(resolved: ResolvedCell, estimand: str) -> float:
    """The truth for one estimand, or a refusal.

    Raises rather than falling back. A panel cell has no ATE, and returning the
    ATT in its place would silently score every estimate against the wrong
    quantity -- which is exactly how two earlier defects in this project got
    their plausible-looking wrong numbers.
    """
    if estimand not in resolved.truth:
        raise KeyError(
            f"this cell carries no {estimand!r} truth; it has "
            f"{sorted(resolved.truth)}. Scoring against a different estimand "
            "would manufacture bias that is not there."
        )
    return resolved.truth[estimand]
