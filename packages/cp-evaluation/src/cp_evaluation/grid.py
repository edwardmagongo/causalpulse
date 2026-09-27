"""What a benchmark cell is, and how a configuration expands into cells.

The four scenario registries are not symmetric, so neither are the cells. A
cross-sectional scenario is built by a `Callable[[ScenarioParams], CrossSectionDGP]`
taking the full set of axis values (`build_scenario` also accepts a bare
`confounding_strength` float for the pre-Phase-18 call shape, which it widens
to a `ScenarioParams` before reaching the registry), and generated with
`n_units`. A panel scenario is built by a `Callable[[], PanelDGP |
BinaryPanelDGP]` taking nothing at all, and generated with `n_units` and
`n_periods`. A donor-pool scenario (`DONOR_SCENARIOS`, S0-S5) fixes its own
size, so its cell carries only a scenario and a seed. Forcing one dataclass
over all four would mean a field that is meaningless for some of its instances.
A CATE scenario (`CATE_SCENARIOS`, RICH/NULL/NOISY) fixes its size too, so its
cell is also only a scenario and a seed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from cp_simulation.cate_scenarios import CATE_SCENARIOS
from cp_simulation.donor_scenarios import DONOR_SCENARIOS
from cp_simulation.panel_scenarios import PANEL_SCENARIOS
from cp_simulation.params import AXIS_NAMES, ScenarioParams
from cp_simulation.scenarios import SCENARIOS

# `ScenarioParams()`'s own defaults, read once rather than duplicated as
# literals here: every "baseline" value this module refers to (for
# n_units-x-confounding_strength's baseline cross-product, for the single
# strength a scenario that does not declare `confounding_strength` is crossed
# against, and for each new axis's own default) comes from this instance, so a
# default changing in `cp_simulation.params` cannot silently drift out of sync
# with what the grid considers "unperturbed".
_DEFAULT_PARAMS = ScenarioParams()

# The value each axis sits at when it is *not* the one being varied, derived
# from `_DEFAULT_PARAMS` rather than restated, so there is one source and not
# two that happen to agree today.
#
# Public because the figures package's axis-profile function needs it to select a
# one-at-a-time slice out of a summary: "every other axis at its default" is
# only well defined against these exact numbers. A figures module that
# hard-coded them would reintroduce precisely the drift the comment above
# exists to prevent, and the figures package has no `cp_simulation` dependency to read
# `ScenarioParams` through. This follows the treatment `GROUP_COLUMNS` already
# got in `cp_evaluation.metrics` for the same cross-package need.
#
# A read-only mapping rather than a dict: an importer must not be able to
# rewrite what the grid considers unperturbed for every other importer in the
# process. `float` rather than the `ScenarioParams` instance itself keeps a
# `cp_simulation` type out of the figures package's annotations.
AXIS_BASELINES: Mapping[str, float] = MappingProxyType(
    {name: float(getattr(_DEFAULT_PARAMS, name)) for name in AXIS_NAMES}
)

# Which axes each cross-sectional scenario declares (`SCENARIOS[id].axes`) --
# any of the five cross-sectional axes, including `confounding_strength`
# itself for D, G and H. Published for the same reason `AXIS_BASELINES`
# above is: the figures package's axis-profile function needs it to refuse a
# profile along an axis a scenario does not vary. That refusal matters
# differently by where the summary came from. `_confounding_values` now gives
# a scenario that does not declare `confounding_strength` a single baseline
# strength, as `_expand_cross_section` already does for the other four axes,
# so a summary the current grid produced collapses an undeclared axis to one
# value and a narrower single-value check would catch it. The declaration
# refusal earns its place against a summary persisted *before* that gating:
# there A, B, C and F carry a full column of strengths describing
# byte-identical data, so the *column* varies where the data behind it does
# not and no single-value check can fire. The figures package has no
# `cp_simulation` dependency to read `SCENARIOS` through directly.
#
# A read-only mapping for the same reason `AXIS_BASELINES` is one: an
# importer must not be able to rewrite what the grid considers "declared"
# for every other importer in the process.
SCENARIO_AXES: Mapping[str, frozenset[str]] = MappingProxyType(
    {scenario_id: spec.axes for scenario_id, spec in SCENARIOS.items()}
)

# The four axes Phase 18 adds, in the order `ScenarioParams`/`AXIS_NAMES`
# already declares them, minus `confounding_strength` -- that one is crossed
# against `n_units` rather than varied one value at a time (spec section 5), so
# it is handled separately from the "one axis at a time" loop the other four go
# through. Separately, but not exempt from declaration: like the four, it is
# only crossed against a scenario that declares it. See `_confounding_values`.
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


@dataclass(frozen=True)
class DonorPoolCell:
    """One donor-pool benchmark cell.

    Unlike `CrossSectionCell`/`PanelCell`, there is no size axis: `DONOR_SCENARIOS`
    (`cp_simulation.donor_scenarios`) bakes n_units, n_periods and donor count into
    each of S0-S5 as a fixed protocol ("a test that quietly uses fewer donors or
    periods is measuring a different, unvalidated regime"), so a donor-pool cell's
    identity is just its scenario and its replication index.
    """

    scenario: str
    seed: int


@dataclass(frozen=True)
class CateCell:
    """One CATE benchmark cell.

    Like `DonorPoolCell`, there is no size axis: `CATE_SCENARIOS`
    (`cp_simulation.cate_scenarios`) fixes every scenario at `CATE_N_UNITS`, and
    its heterogeneity is set from a detection floor measured at that size, so a
    CATE cell's identity is its scenario and its replication index.
    """

    scenario: str
    seed: int


Cell = CrossSectionCell | PanelCell | DonorPoolCell | CateCell


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
    other -- see `expand_cells`. A scenario is only varied against an axis it
    declares (`SCENARIOS[id].axes`, checked by
    `tests/simulation/test_axis_declarations.py`); one it does not declare is
    left at that axis's default in every cell, never crossed against it.
    `confounding_strength` is subject to that rule too, so what sets it apart
    from the four axes below is only *how* it enumerates -- crossed against
    `n_units`, rather than varied one value at a time -- and not whether
    declaration gates it: A, B, C and F declare none of it, so each
    contributes a single baseline strength per `n_units` (`ScenarioParams`'
    own default) instead of one cell per value. Spec section 5's "exactly as
    today", which left that cross-product ungated through Phase 18, is
    superseded; section 12 of
    `docs/specs/2026-09-13-causal-core-phase18-design.md` records the change.
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
class DonorPoolAxes:
    """Which of S0-S5 to include. No size axis -- see `DonorPoolCell`."""

    scenarios: Sequence[str]


@dataclass(frozen=True)
class CateAxes:
    """Which of `CATE_SCENARIOS` to include. No size axis -- see `CateCell`."""

    scenarios: Sequence[str]


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
    `donor_pool` is the optional third arm (`DonorPoolAxes`, S0-S5); it defaults
    to `None`, so a config that does not mention it has no donor cells. `cate`
    is the optional fourth arm (`CateAxes`, `CATE_SCENARIOS`), defaulting to
    `None` the same way.
    """

    cross_section: CrossSectionAxes | None
    panel: PanelAxes | None
    replications: int
    donor_pool: DonorPoolAxes | None = None
    cate: CateAxes | None = None

    def __post_init__(self) -> None:
        if self.replications < 1:
            raise ValueError(f"replications must be positive, got {self.replications}")


def expand_cells(config: GridConfig) -> tuple[Cell, ...]:
    """Expand a configuration into its sorted, deduplicated list of cells.

    Sorted rather than merely enumerated: the order is part of the output's
    identity, and a grid whose row order depended on dict iteration would make
    two runs of the same configuration hard to compare.
    """
    if (
        config.cross_section is None
        and config.panel is None
        and config.donor_pool is None
        and config.cate is None
    ):
        raise ValueError(
            "a GridConfig needs at least one of cross_section, panel, donor_pool or cate"
        )

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

    if config.donor_pool is not None:
        axes_dp = config.donor_pool
        unknown = sorted(set(axes_dp.scenarios) - set(DONOR_SCENARIOS))
        if unknown:
            raise ValueError(
                f"unknown donor-pool scenarios {unknown}; available: {sorted(DONOR_SCENARIOS)}"
            )
        cells.extend(
            DonorPoolCell(scenario=scenario, seed=seed)
            for scenario in axes_dp.scenarios
            for seed in range(config.replications)
        )

    if config.cate is not None:
        axes_c = config.cate
        unknown = sorted(set(axes_c.scenarios) - set(CATE_SCENARIOS))
        if unknown:
            raise ValueError(
                f"unknown CATE scenarios {unknown}; available: {sorted(CATE_SCENARIOS)}"
            )
        cells.extend(
            CateCell(scenario=scenario, seed=seed)
            for scenario in axes_c.scenarios
            for seed in range(config.replications)
        )

    return tuple(sorted(set(cells), key=cell_sort_key))


def _confounding_values(axes: CrossSectionAxes, scenario: str) -> tuple[float, ...]:
    """The confounding strengths one scenario's baseline block is crossed against.

    A scenario that declares the axis is crossed against every value the
    config asks for; one that does not gets a single baseline strength,
    `ScenarioParams`' own default. That is the same treatment the four axes
    Phase 18 adds already receive -- `_expand_cross_section` leaves an
    undeclared one at its default and never varies it -- applied to the axis
    that predates them.

    A, B, C and F assign at random, so nothing their builders do reads this
    axis. Crossing them against three strengths produced three cells per
    `n_units` differing only in RNG stream, which `summarise` then reported as
    though the strength had been varied: spec section 5.1's absent-versus-zero
    hazard, for the one axis section 5 had left ungated. The gating removes 16
    such cells from `full_config()`.

    **The single strength is the default even when the config's axis does not
    contain it**, so a config of `confounding_strength=(0.5, 2.0)` over
    scenario A produces cells at 1.0 and none at either requested value
    (pinned by
    `tests/evaluation/test_grid.py::test_an_undeclared_confounding_scenario_never_needs_the_anchor`).
    That is deliberate: for a scenario that never reads the axis the field is
    a label, and the honest label is "unperturbed", not a value implying the
    grid varied something it did not. It is also exactly what the four axes
    Phase 18 adds already do -- widen `heterogeneity_scale` to `(0.0, 2.0)`
    and scenario A's cells still carry its 1.0 default.

    The label is not by itself a provenance gap, and
    `_check_the_anchor_is_on_the_grid` deliberately does **not** refuse it: an
    undeclared scenario's anchor is on its own baseline block by construction,
    so there is nothing uninterpretable to refuse. What did have to be closed
    is the record of the run. the CLI's `bench` command derives `run.json`'s
    axis block from this dataclass, so a run at `(0.5, 2.0)` recorded those
    two strengths beside a parquet whose A rows all read 1.0. It now writes a
    `cross_section_realised`/`panel_realised` block as well, read off the
    frame it is describing, and `bench report` names any axis where the two
    disagree. Neither shipped tier produces a disagreement -- both carry the
    default strength on their axis -- so the note stays silent for `ci` and
    `full`, which `tests/cli/test_bench.py` checks for each of them.
    """
    if "confounding_strength" in SCENARIOS[scenario].axes:
        return tuple(float(value) for value in axes.confounding_strength)
    return (_DEFAULT_PARAMS.confounding_strength,)


def _perturbed_axes(axes: CrossSectionAxes, scenario: str) -> tuple[str, ...]:
    """The new axes that will actually produce a one-at-a-time cell for `scenario`.

    An axis counts only if this scenario declares it *and* the config carries
    a value away from `ScenarioParams`' default -- the same two conditions
    `_expand_cross_section` applies before it yields anything. Asked per
    scenario rather than across the whole config because whether the anchor
    has to be present on the confounding axis is itself a per-scenario
    question; see `_check_the_anchor_is_on_the_grid`.
    """
    declared = SCENARIOS[scenario].axes
    perturbed = []
    for name in _NEW_AXIS_NAMES:
        baseline = AXIS_BASELINES[name]
        if name in declared and any(v != baseline for v in getattr(axes, name)):
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

    **Asked per scenario**, because `_confounding_values` makes the answer
    one. A scenario that does not declare `confounding_strength` has its
    entire baseline block at the default strength whatever the config's
    confounding axis says, so its anchor is on the grid by construction as
    well, and refusing that config would reject a grid every cell of which is
    interpretable. Only a scenario that both declares the axis and widens a
    new axis can be left without a baseline twin. Before the axis was gated
    by declaration this distinction did not exist -- every scenario was
    crossed against every strength -- so the check could be, and was, asked
    once for the whole config.
    """
    if AXIS_BASELINES["confounding_strength"] in {float(v) for v in axes.confounding_strength}:
        return
    for scenario in axes.scenarios:
        if "confounding_strength" not in SCENARIOS[scenario].axes:
            continue
        perturbed = _perturbed_axes(axes, scenario)
        if not perturbed:
            continue
        raise ValueError(
            f"scenario {scenario} varying {list(perturbed)} one at a time anchors "
            f"on confounding_strength={AXIS_BASELINES['confounding_strength']} "
            f"(ScenarioParams' default), which is not in "
            f"{list(axes.confounding_strength)}; the perturbed cells would have no "
            f"baseline cell to be compared against. Add it to the axis, or leave "
            f"those axes at their defaults."
        )


def _expand_cross_section(
    axes: CrossSectionAxes, replications: int
) -> Iterator[CrossSectionCell]:
    """One-at-a-time enumeration for the four axes Phase 18 adds (spec section 5).

    `n_units` x `confounding_strength` is generated first, as a cross-product
    over the strengths this scenario is actually crossed against
    (`_confounding_values` -- every config value if it declares the axis, a
    single default strength if it does not), with every new axis at its
    `ScenarioParams` default. This reproduces the pre-Phase-18 grid exactly
    when none of the four axes is widened past its single-value default *and*
    the config asks only for the default strength, which is `ci_config()`'s
    case: its cells, and so their `cell_rng` streams and the
    `TOLERANCE_BANDS` measured against them, are untouched by the gating.

    Each new axis then varies **alone**, one non-default value at a time,
    against a fixed anchor: `n_units` pinned to this config's largest value
    (more units gives more power to detect the axis's own move) and
    `confounding_strength` pinned to `ScenarioParams`'s own default (1.0) --
    both members of the cross-product above, so this reuses a point on the
    existing grid rather than adding a new one. `n_units` is one by
    construction. `confounding_strength` is one by construction too for a
    scenario that does not declare it (`_confounding_values` returns exactly
    that strength and nothing else), and for one that does because
    `_check_the_anchor_is_on_the_grid` refuses a config where it would not be.
    Every other new axis sits at its own default throughout, so only one axis
    differs from baseline in any single cell.

    A scenario is only varied against an axis it declares
    (`SCENARIOS[id].axes`): crossing it against an axis its builder ignores
    would add cells that are all identical to the baseline, reporting the
    axis as varied when nothing moved (spec section 5.1, the same
    absent-versus-zero confusion Phase 16's heatmaps were built to avoid).
    That holds for `confounding_strength` as well as for the four axes this
    loop varies -- the baseline block above reads `_confounding_values`, not
    `axes.confounding_strength`, for exactly that reason.
    """
    anchor_n_units = max(axes.n_units)
    anchor_confounding_strength = AXIS_BASELINES["confounding_strength"]

    for scenario in axes.scenarios:
        declared = SCENARIOS[scenario].axes

        strengths = _confounding_values(axes, scenario)

        for n_units in axes.n_units:
            for strength in strengths:
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
            baseline_value = AXIS_BASELINES[axis_name]
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
    """A total order over all four cell shapes.

    The shape tag leads, so the shapes never interleave
    and the blocks stay independently readable. The four new axes sort
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
    if isinstance(cell, PanelCell):
        return (1, cell.scenario, cell.n_units, cell.n_periods, cell.seed)
    if isinstance(cell, DonorPoolCell):
        return (2, cell.scenario, cell.seed)
    return (3, cell.scenario, cell.seed)


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
    if isinstance(cell, PanelCell):
        return {
            "shape": "panel",
            "scenario": cell.scenario,
            "n_units": cell.n_units,
            "n_periods": cell.n_periods,
            "seed": cell.seed,
        }
    if isinstance(cell, DonorPoolCell):
        return {"shape": "donor_pool", "scenario": cell.scenario, "seed": cell.seed}
    return {"shape": "cate", "scenario": cell.scenario, "seed": cell.seed}


def cell_key(cell: Cell) -> str:
    """A short, stable name for one cell, derived from its identity payload.

    The same identity `cell_rng` hashes, so two cells share a key exactly when
    they share a generator: a stored result can be matched to the cell that
    produced it without reading a row. sha256 over sorted JSON, not `hash()`,
    for the reason `cell_rng` documents.

    24 hex digits is 96 bits, far beyond what a grid of thousands of cells can
    collide in, and short enough to name a file.
    """
    encoded = json.dumps(_identity_payload(cell), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


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
