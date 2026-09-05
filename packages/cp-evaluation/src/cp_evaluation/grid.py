"""What a benchmark cell is, and how a configuration expands into cells.

The two scenario registries are not symmetric, so neither are the cells. A
cross-sectional scenario is built by a `Callable[[float], CrossSectionDGP]`
taking confounding strength, and generated with `n_units`. A panel scenario is
built by a `Callable[[], PanelDGP | BinaryPanelDGP]` taking nothing at all, and
generated with `n_units` and `n_periods`. Forcing one dataclass over both would
mean a field that is meaningless for half its instances.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from cp_simulation.panel_scenarios import PANEL_SCENARIOS
from cp_simulation.scenarios import SCENARIOS


@dataclass(frozen=True)
class CrossSectionCell:
    """One cross-sectional benchmark cell.

    `seed` is a **replication index**, not an RNG seed. The generator for this
    cell is derived from its whole identity by `cell_rng`; passing this integer
    to `default_rng` directly would produce different data.
    """

    scenario: str
    n_units: int
    confounding_strength: float
    seed: int


@dataclass(frozen=True)
class PanelCell:
    """One panel benchmark cell.

    `seed` is a replication index, exactly as in `CrossSectionCell`.
    """

    scenario: str
    n_units: int
    n_periods: int
    seed: int


Cell = CrossSectionCell | PanelCell


@dataclass(frozen=True)
class CrossSectionAxes:
    scenarios: Sequence[str]
    n_units: Sequence[int]
    confounding_strength: Sequence[float]


@dataclass(frozen=True)
class PanelAxes:
    scenarios: Sequence[str]
    n_units: Sequence[int]
    n_periods: Sequence[int]


@dataclass(frozen=True)
class GridConfig:
    """The axes of a benchmark run.

    There is deliberately no axis for treatment share, baseline noise, effect
    size or heterogeneity. Those four are named by the parent spec's §8.1 but
    do not exist in the data-generating processes: a scenario builder takes
    confounding strength and nothing else, and every other parameter is fixed
    inside its closure. Adding such an axis means extending the builders first.
    """

    cross_section: CrossSectionAxes | None
    panel: PanelAxes | None
    replications: int

    def __post_init__(self) -> None:
        if self.replications < 1:
            raise ValueError(f"replications must be positive, got {self.replications}")


def expand_cells(config: GridConfig) -> tuple[Cell, ...]:
    """Expand a configuration into its sorted, deduplicated list of cells.

    Sorted rather than merely enumerated: the order is part of the output's
    identity, and a grid whose row order depended on dict iteration would make
    two runs of the same configuration hard to compare.
    """
    if config.cross_section is None and config.panel is None:
        raise ValueError("a GridConfig needs at least one of cross_section or panel")

    cells: list[Cell] = []

    if config.cross_section is not None:
        axes = config.cross_section
        unknown = sorted(set(axes.scenarios) - set(SCENARIOS))
        if unknown:
            raise ValueError(
                f"unknown cross-sectional scenarios {unknown}; available: {sorted(SCENARIOS)}"
            )
        cells.extend(
            CrossSectionCell(
                scenario=scenario,
                n_units=n_units,
                confounding_strength=float(strength),
                seed=seed,
            )
            for scenario in axes.scenarios
            for n_units in axes.n_units
            for strength in axes.confounding_strength
            for seed in range(config.replications)
        )

    if config.panel is not None:
        axes_p = config.panel
        unknown = sorted(set(axes_p.scenarios) - set(PANEL_SCENARIOS))
        if unknown:
            raise ValueError(
                f"unknown panel scenarios {unknown}; available: {sorted(PANEL_SCENARIOS)}"
            )
        cells.extend(
            PanelCell(scenario=scenario, n_units=n_units, n_periods=n_periods, seed=seed)
            for scenario in axes_p.scenarios
            for n_units in axes_p.n_units
            for n_periods in axes_p.n_periods
            for seed in range(config.replications)
        )

    return tuple(sorted(set(cells), key=cell_sort_key))


def cell_sort_key(cell: Cell) -> tuple[object, ...]:
    """A total order over both cell shapes.

    The shape tag leads, so cross-sectional and panel cells never interleave
    and the two blocks stay independently readable.
    """
    if isinstance(cell, CrossSectionCell):
        return (0, cell.scenario, cell.n_units, cell.confounding_strength, cell.seed)
    return (1, cell.scenario, cell.n_units, cell.n_periods, cell.seed)


def _identity_payload(cell: Cell) -> dict[str, object]:
    """A cell's identity as JSON-serialisable data, tagged by shape.

    The shape tag is part of the payload, not just of the sort key: without it
    a `CrossSectionCell` and a `PanelCell` whose numbers coincide would hash to
    the same stream.
    """
    if isinstance(cell, CrossSectionCell):
        return {
            "shape": "cross_section",
            "scenario": cell.scenario,
            "n_units": cell.n_units,
            "confounding_strength": cell.confounding_strength,
            "seed": cell.seed,
        }
    return {
        "shape": "panel",
        "scenario": cell.scenario,
        "n_units": cell.n_units,
        "n_periods": cell.n_periods,
        "seed": cell.seed,
    }


def cell_rng(cell: Cell, root_entropy: int) -> np.random.Generator:
    """The generator for one cell, derived from the cell's own identity.

    Deliberately not `SeedSequence(root).spawn(i)` indexed by position in the
    grid, which the parent spec's §8.1 suggests. Position-indexed spawning
    gives order-independence but not configuration-stability: adding one value
    to one axis renumbers every index, so every other cell's data changes even
    though nothing about those cells did. Deriving from identity gives both,
    and removes the migration a content-addressed cache would otherwise force.

    sha256 over sorted JSON, not `hash()`, for the reason
    `cp_simulation._hashing` already documents: `hash()` is salted per process,
    so a stream derived from it would not reproduce across runs at all.
    """
    encoded = json.dumps(_identity_payload(cell), sort_keys=True).encode()
    digest = hashlib.sha256(encoded).digest()
    # Four 32-bit words is ample entropy to distinguish cells, and keeps the
    # SeedSequence entropy vector short enough to stay readable when logged.
    words = [int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4)]
    return np.random.default_rng(np.random.SeedSequence([root_entropy, *words]))
