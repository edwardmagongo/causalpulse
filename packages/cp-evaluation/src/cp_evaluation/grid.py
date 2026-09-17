"""What a benchmark cell is, and how a configuration expands into cells.

The two scenario registries are not symmetric, so neither are the cells. A
cross-sectional scenario is built by a `Callable[[ScenarioParams], CrossSectionDGP]`
taking the full set of axis values (`build_scenario` also accepts a bare
`confounding_strength` float for the pre-Phase-18 call shape, which it widens
to a `ScenarioParams` before reaching the registry), and generated with
`n_units`. A panel scenario is built by a `Callable[[], PanelDGP |
BinaryPanelDGP]` taking nothing at all, and generated with `n_units` and
`n_periods`. Forcing one dataclass over both would mean a field that is
meaningless for half its instances.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from cp_simulation.panel_scenarios import PANEL_SCENARIOS
from cp_simulation.params import AXIS_NAMES, ScenarioParams
from cp_simulation.scenarios import SCENARIOS

# `ScenarioParams()`'s own defaults, read once rather than duplicated as
# literals here: every "baseline" value this module refers to (for
# n_units-x-confounding_strength's baseline cross-product, and for each new
# axis's own default) comes from this instance, so a default changing in
# `cp_simulation.params` cannot silently drift out of sync with what the grid
# considers "unperturbed".
_DEFAULT_PARAMS = ScenarioParams()

# The four axes Phase 18 adds, in the order `ScenarioParams`/`AXIS_NAMES`
# already declares them, minus `confounding_strength` -- that one stays fully
# crossed against `n_units` (spec section 5), so it is handled separately
# from the "one axis at a time" loop the other four go through.
_NEW_AXIS_NAMES: tuple[str, ...] = tuple(
    name for name in AXIS_NAMES if name != "confounding_strength"
)


@dataclass(frozen=True)
class CrossSectionCell:
    """One cross-sectional benchmark cell.

    `seed` is a **replication index**, not an RNG seed. The generator for this
    cell is derived from its whole identity by `cell_rng`; passing this integer
    to `default_rng` directly would produce different data.

    The four fields after `confounding_strength` all default to
    `ScenarioParams()`'s own defaults, matching what a pre-Phase-18 caller
    that never mentions them gets: a `CrossSectionCell` built the old way
    (`scenario`, `n_units`, `confounding_strength`, `seed` only) is exactly
    the baseline cell it always was, byte-identical down to `cell_rng`'s
    stream (see `_identity_payload`).
    """

    scenario: str
    n_units: int
    confounding_strength: float
    seed: int
    treatment_intercept: float = _DEFAULT_PARAMS.treatment_intercept
    noise_scale: float = _DEFAULT_PARAMS.noise_scale
    effect_scale: float = _DEFAULT_PARAMS.effect_scale
    heterogeneity_scale: float = _DEFAULT_PARAMS.heterogeneity_scale


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
    """The cross-sectional grid's axes.

    `n_units` x `confounding_strength` stays a full cross-product (spec
    section 5) -- that interaction is what lets the study show where
    asymptotic approximations fail. The four axes below default to a single
    value, `ScenarioParams()`'s own default for each: at that default,
    `expand_cells` produces exactly the pre-Phase-18 grid (see
    `CrossSectionCell`'s docstring), and a caller that widens one of them
    gets that axis varied **one value at a time** against a fixed baseline,
    not crossed against `n_units`/`confounding_strength` or against each
    other -- see `expand_cells`. **Of the four axes below**, a scenario is
    only varied against one it declares (`SCENARIOS[id].axes`, checked by
    `tests/simulation/test_axis_declarations.py`); one it does not declare is
    left at that axis's default in every cell, never crossed against it.
    `confounding_strength` is deliberately not subject to that rule: it stays
    fully crossed for every scenario, declared or not (spec section 5 keeps
    that cross-product "exactly as today"), so A, B, C and F are crossed
    against all of its values despite declaring none of it.
    """

    scenarios: Sequence[str]
    n_units: Sequence[int]
    confounding_strength: Sequence[float]
    treatment_intercept: Sequence[float] = (_DEFAULT_PARAMS.treatment_intercept,)
    noise_scale: Sequence[float] = (_DEFAULT_PARAMS.noise_scale,)
    effect_scale: Sequence[float] = (_DEFAULT_PARAMS.effect_scale,)
    heterogeneity_scale: Sequence[float] = (_DEFAULT_PARAMS.heterogeneity_scale,)


@dataclass(frozen=True)
class PanelAxes:
    scenarios: Sequence[str]
    n_units: Sequence[int]
    n_periods: Sequence[int]


@dataclass(frozen=True)
class GridConfig:
    """The axes of a benchmark run.

    Cross-sectional generation reads `ScenarioParams` (`cp_simulation.params`)
    through `build_scenario`, which is why `CrossSectionAxes` carries
    `treatment_intercept`, `noise_scale`, `effect_scale` and
    `heterogeneity_scale` alongside `confounding_strength` -- see that
    dataclass's own docstring for how they enumerate. The panel arm has no
    such axes: `PanelAxes` builds each scenario with
    `Callable[[], PanelDGP | BinaryPanelDGP]`, which takes nothing at all.
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
        _check_the_anchor_is_on_the_grid(axes)
        cells.extend(_expand_cross_section(axes, config.replications))

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


def _perturbed_axes(axes: CrossSectionAxes) -> tuple[str, ...]:
    """The new axes that will actually produce a one-at-a-time cell.

    An axis counts only if some scenario in this config declares it *and* it
    carries a value away from `ScenarioParams`' default -- the same two
    conditions `_expand_cross_section` applies before it yields anything.
    """
    perturbed = []
    for name in _NEW_AXIS_NAMES:
        declared_somewhere = any(name in SCENARIOS[s].axes for s in axes.scenarios)
        baseline = getattr(_DEFAULT_PARAMS, name)
        if declared_somewhere and any(v != baseline for v in getattr(axes, name)):
            perturbed.append(name)
    return tuple(perturbed)


def _check_the_anchor_is_on_the_grid(axes: CrossSectionAxes) -> None:
    """Refuse a config whose one-at-a-time anchor is not a cell the grid has.

    `_expand_cross_section` pins every perturbed cell to a fixed anchor so
    that exactly one axis differs from baseline. That only *means* anything
    if the anchor is itself a baseline cell: the comparison a one-at-a-time
    grid exists to support is "this cell versus the otherwise-identical cell
    at the axis's default", and there is no such cell to compare against if
    the anchor sits at a `confounding_strength` the config never asked for.

    `n_units`' anchor is `max(axes.n_units)`, so it is on the grid by
    construction. `confounding_strength`'s anchor is `ScenarioParams`' own
    default, which is not -- a config of `confounding_strength=(0.5, 2.0)`
    that widens any new axis would otherwise silently emit perturbed cells at
    1.0, a strength with no baseline twin anywhere in the output, and nothing
    downstream could tell those cells apart from a like-for-like comparison.
    Raising here is the same treatment `expand_cells` already gives an
    unknown scenario: a configuration error, named at the point it is made.
    """
    perturbed = _perturbed_axes(axes)
    if not perturbed:
        return
    if _DEFAULT_PARAMS.confounding_strength in {float(v) for v in axes.confounding_strength}:
        return
    raise ValueError(
        f"varying {list(perturbed)} one at a time anchors on "
        f"confounding_strength={_DEFAULT_PARAMS.confounding_strength} "
        f"(ScenarioParams' default), which is not in "
        f"{list(axes.confounding_strength)}; the perturbed cells would have no "
        f"baseline cell to be compared against. Add it to the axis, or leave "
        f"those axes at their defaults."
    )


def _expand_cross_section(
    axes: CrossSectionAxes, replications: int
) -> Iterator[CrossSectionCell]:
    """One-at-a-time enumeration for the four axes Phase 18 adds (spec section 5).

    `n_units` x `confounding_strength` is generated first, as a full
    cross-product, with every new axis at its `ScenarioParams` default --
    this reproduces the pre-Phase-18 grid exactly when none of the four axes
    is widened past its single-value default.

    Each new axis then varies **alone**, one non-default value at a time,
    against a fixed anchor: `n_units` pinned to this config's largest value
    (more units gives more power to detect the axis's own move) and
    `confounding_strength` pinned to `ScenarioParams`'s own default (1.0) --
    both members of the cross-product above, so this reuses a point on the
    existing grid rather than adding a new one. `n_units` is one by
    construction; `confounding_strength` is one because
    `_check_the_anchor_is_on_the_grid` refuses a config where it would not
    be. Every other new axis
    sits at its own default throughout, so only one axis differs from
    baseline in any single cell.

    A scenario is only varied against an axis it declares
    (`SCENARIOS[id].axes`): crossing it against an axis its builder ignores
    would add cells that are all identical to the baseline, reporting the
    axis as varied when nothing moved (spec section 5.1, the same
    absent-versus-zero confusion Phase 16's heatmaps were built to avoid).
    """
    anchor_n_units = max(axes.n_units)
    anchor_confounding_strength = _DEFAULT_PARAMS.confounding_strength

    for scenario in axes.scenarios:
        declared = SCENARIOS[scenario].axes

        for n_units in axes.n_units:
            for strength in axes.confounding_strength:
                for seed in range(replications):
                    yield CrossSectionCell(
                        scenario=scenario,
                        n_units=n_units,
                        confounding_strength=float(strength),
                        seed=seed,
                    )

        for axis_name in _NEW_AXIS_NAMES:
            if axis_name not in declared:
                continue
            baseline_value = getattr(_DEFAULT_PARAMS, axis_name)
            for value in getattr(axes, axis_name):
                if value == baseline_value:
                    continue
                for seed in range(replications):
                    yield CrossSectionCell(
                        scenario=scenario,
                        n_units=anchor_n_units,
                        confounding_strength=anchor_confounding_strength,
                        seed=seed,
                        **{axis_name: float(value)},
                    )


def cell_sort_key(cell: Cell) -> tuple[object, ...]:
    """A total order over both cell shapes.

    The shape tag leads, so cross-sectional and panel cells never interleave
    and the two blocks stay independently readable. The four new axes sort
    after `confounding_strength` and before `seed`, in `AXIS_NAMES` order, so
    the extra cells `_expand_cross_section` adds sort predictably alongside
    the pre-existing axes rather than trailing off separately.
    """
    if isinstance(cell, CrossSectionCell):
        return (
            0,
            cell.scenario,
            cell.n_units,
            cell.confounding_strength,
            *(getattr(cell, name) for name in _NEW_AXIS_NAMES),
            cell.seed,
        )
    return (1, cell.scenario, cell.n_units, cell.n_periods, cell.seed)


def _identity_payload(cell: Cell) -> dict[str, object]:
    """A cell's identity as JSON-serialisable data, tagged by shape.

    The shape tag is part of the payload, not just of the sort key: without it
    a `CrossSectionCell` and a `PanelCell` whose numbers coincide would hash to
    the same stream.

    A new axis is only added to the payload when it differs from
    `ScenarioParams`'s own default. This is what keeps `cell_rng`'s stream
    for a pre-Phase-18-shaped `CrossSectionCell` (every new axis left at its
    default) byte-identical to what it was before these fields existed --
    `ci_config()`'s cells are exactly that shape, and its `TOLERANCE_BANDS`
    were measured against the pre-Phase-18 stream. Folding every axis into
    the payload unconditionally, even at its default value, would silently
    redraw every cell `ci_config()` and `full_config()`'s baseline
    cross-product already produce, with no test able to tell the difference
    from a genuine data change.
    """
    if isinstance(cell, CrossSectionCell):
        payload: dict[str, object] = {
            "shape": "cross_section",
            "scenario": cell.scenario,
            "n_units": cell.n_units,
            "confounding_strength": cell.confounding_strength,
            "seed": cell.seed,
        }
        for name in _NEW_AXIS_NAMES:
            value = getattr(cell, name)
            if value != getattr(_DEFAULT_PARAMS, name):
                payload[name] = value
        return payload
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
