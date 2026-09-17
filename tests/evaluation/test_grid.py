import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from cp_evaluation.grid import (
    CrossSectionAxes,
    CrossSectionCell,
    GridConfig,
    PanelAxes,
    PanelCell,
    cell_rng,
    expand_cells,
)
from cp_simulation.params import AXIS_NAMES, ScenarioParams
from cp_simulation.scenarios import SCENARIOS


def test_cross_section_axes_expand_to_a_cross_product():
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A", "D"), n_units=(500, 1000), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=3,
    )

    cells = expand_cells(config)

    assert len(cells) == 2 * 2 * 1 * 3
    assert all(isinstance(c, CrossSectionCell) for c in cells)
    assert {c.seed for c in cells} == {0, 1, 2}


def test_panel_axes_expand_over_n_periods_not_confounding():
    config = GridConfig(
        cross_section=None,
        panel=PanelAxes(scenarios=("P1", "P2"), n_units=(200,), n_periods=(6, 10)),
        replications=2,
    )

    cells = expand_cells(config)

    assert len(cells) == 2 * 1 * 2 * 2
    assert all(isinstance(c, PanelCell) for c in cells)
    assert {c.n_periods for c in cells} == {6, 10}


def test_both_shapes_expand_independently_and_concatenate():
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(500,), confounding_strength=(1.0,)
        ),
        panel=PanelAxes(scenarios=("P1",), n_units=(200,), n_periods=(6,)),
        replications=2,
    )

    cells = expand_cells(config)

    assert len(cells) == 4
    assert sum(isinstance(c, CrossSectionCell) for c in cells) == 2
    assert sum(isinstance(c, PanelCell) for c in cells) == 2


def test_expansion_is_sorted_and_therefore_deterministic():
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("D", "A"), n_units=(1000, 500), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=2,
    )

    once = expand_cells(config)
    twice = expand_cells(config)

    assert once == twice
    assert list(once) == sorted(once, key=lambda c: (c.scenario, c.n_units, c.seed))


def test_an_unknown_cross_section_scenario_is_refused_by_name():
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A", "ZZ"), n_units=(500,), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=1,
    )

    with pytest.raises(ValueError, match="ZZ"):
        expand_cells(config)


def test_an_unknown_panel_scenario_is_refused_by_name():
    config = GridConfig(
        cross_section=None,
        panel=PanelAxes(scenarios=("P1", "P99"), n_units=(200,), n_periods=(6,)),
        replications=1,
    )

    with pytest.raises(ValueError, match="P99"):
        expand_cells(config)


def test_a_grid_with_no_axes_at_all_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        expand_cells(GridConfig(cross_section=None, panel=None, replications=1))


def test_replications_must_be_positive():
    with pytest.raises(ValueError, match="replications"):
        GridConfig(
            cross_section=CrossSectionAxes(
                scenarios=("A",), n_units=(500,), confounding_strength=(1.0,)
            ),
            panel=None,
            replications=0,
        )


def test_repeated_axis_values_are_deduplicated_to_prevent_double_weighting():
    # Duplicate cells would silently count twice in averaged metrics downstream,
    # double-weighting their contribution to the benchmark results. This test
    # verifies that repeated values in any axis are deduplicated.
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A", "A"), n_units=(500,), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=2,
    )

    cells = expand_cells(config)

    # Without deduplication, we would get 2 (repeated scenarios) * 1 * 1 * 2 = 4 cells.
    # With deduplication, repeated scenario "A" should merge to 1 unique cell per
    # combination of other dimensions, giving us 1 * 1 * 1 * 2 = 2 cells.
    assert len(cells) == 2
    assert all(c.scenario == "A" for c in cells)
    assert {c.seed for c in cells} == {0, 1}


def test_the_same_cell_always_gets_the_same_stream():
    cell = CrossSectionCell(scenario="D", n_units=500, confounding_strength=1.0, seed=0)

    first = cell_rng(cell, root_entropy=12345).normal(size=5)
    second = cell_rng(cell, root_entropy=12345).normal(size=5)

    assert np.array_equal(first, second)


def test_different_cells_get_different_streams():
    a = CrossSectionCell(scenario="D", n_units=500, confounding_strength=1.0, seed=0)
    b = CrossSectionCell(scenario="D", n_units=500, confounding_strength=1.0, seed=1)
    c = CrossSectionCell(scenario="A", n_units=500, confounding_strength=1.0, seed=0)

    draws = [tuple(cell_rng(x, root_entropy=1).normal(size=5)) for x in (a, b, c)]

    assert len(set(draws)) == 3


def test_a_different_root_changes_every_stream():
    cell = PanelCell(scenario="P1", n_units=200, n_periods=6, seed=0)

    assert not np.array_equal(
        cell_rng(cell, root_entropy=1).normal(size=5),
        cell_rng(cell, root_entropy=2).normal(size=5),
    )


def test_a_cell_s_stream_does_not_depend_on_the_rest_of_the_grid():
    # This is what the spec's spawn(cell_index) scheme could not deliver, and
    # is why this plan deviates from it: adding a value to any axis renumbers
    # every index and silently changes every other cell's data.
    small = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("D",), n_units=(500,), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=1,
    )
    larger = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A", "D"), n_units=(500, 1000), confounding_strength=(1.0, 2.0)
        ),
        panel=None,
        replications=4,
    )

    target = CrossSectionCell(scenario="D", n_units=500, confounding_strength=1.0, seed=0)
    cells_small = expand_cells(small)
    cells_larger = expand_cells(larger)
    from_small = next(c for c in cells_small if c == target)
    from_larger = next(c for c in cells_larger if c == target)

    # cell_sort_key orders by (shape_tag, scenario, n_units,
    # confounding_strength, seed), and every "A" cell sorts ahead of every "D"
    # cell. `small` has only "D" cells, so the target sits at index 0; adding
    # the "A" scenario in `larger` pushes it later. This assertion guards the
    # test itself: if both indices coincided, a position-indexed cell_rng
    # (SeedSequence(root).spawn(cell_index), the scheme this test exists to
    # reject) would produce matching streams purely by accident of shared
    # position, and the assertion below would pass without ever exercising
    # the property that the stream depends on cell identity, not location.
    index_small = cells_small.index(from_small)
    index_larger = cells_larger.index(from_larger)
    assert index_small != index_larger

    assert np.array_equal(
        cell_rng(from_small, root_entropy=7).normal(size=5),
        cell_rng(from_larger, root_entropy=7).normal(size=5),
    )


def test_cell_rng_reproduces_across_processes_with_a_non_default_hash_seed():
    # cell_rng derives its stream from hashlib.sha256 over the cell's
    # identity, not from the builtin hash(), precisely because hash() is
    # salted per process by PYTHONHASHSEED and would not reproduce across
    # runs. This test guards against a regression back to hash(): it runs
    # cell_rng in a subprocess with an explicit, non-default PYTHONHASHSEED
    # and checks the draws still match the in-process draws. If cell_rng ever
    # started hashing with the salted builtin, the two would disagree.
    cell = CrossSectionCell(scenario="D", n_units=500, confounding_strength=1.0, seed=0)
    expected = cell_rng(cell, root_entropy=12345).normal(size=5).tolist()

    script = (
        "import json\n"
        "from cp_evaluation.grid import CrossSectionCell, cell_rng\n"
        "cell = CrossSectionCell(\n"
        "    scenario='D', n_units=500, confounding_strength=1.0, seed=0\n"
        ")\n"
        "draws = cell_rng(cell, root_entropy=12345).normal(size=5)\n"
        "print(json.dumps(draws.tolist()))\n"
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    from_subprocess = json.loads(result.stdout)

    assert np.array_equal(expected, from_subprocess)


def test_the_two_shapes_cannot_collide_on_one_stream():
    # A cross-sectional cell and a panel cell with the same numbers must not
    # share a stream just because their field values happen to coincide.
    x = CrossSectionCell(scenario="X", n_units=200, confounding_strength=6.0, seed=0)
    p = PanelCell(scenario="X", n_units=200, n_periods=6, seed=0)

    assert not np.array_equal(
        cell_rng(x, root_entropy=1).normal(size=5),
        cell_rng(p, root_entropy=1).normal(size=5),
    )


def test_a_default_valued_cell_hashes_the_same_stream_as_before_phase_18():
    # Pinned 2026-09-15, on `causal-core-phase18` HEAD a2fe1b2, before any
    # Task 9 edit landed: `cell_rng(CrossSectionCell(scenario="D",
    # n_units=500, confounding_strength=1.0, seed=0), root_entropy=1)
    # .normal(size=5)`. `ci_config()`'s cells never carry a non-default value
    # for any of the four new axes (its own axes are single-valued, matching
    # `ScenarioParams()`'s defaults), so its `TOLERANCE_BANDS` -- measured
    # against the pre-Task-9 stream -- stay valid only if a cell whose new
    # axes are all at their default hashes identically to how it did before
    # those fields existed. A version that always folds every new axis into
    # the hash payload, even at its default value, breaks this.
    cell = CrossSectionCell(scenario="D", n_units=500, confounding_strength=1.0, seed=0)

    draws = cell_rng(cell, root_entropy=1).normal(size=5)

    expected = [
        -1.8698984549736246,
        -0.0009888440217891046,
        0.3048303293574386,
        -0.9543127874991957,
        -1.5253624498096867,
    ]
    assert np.allclose(draws, expected)


def test_a_scenario_is_not_crossed_against_an_axis_it_does_not_declare():
    # Scenario B is a ConstantEffect scenario with tau == 0: `effect_scale`
    # still multiplies it, but zero times any scale is zero, so B's builder
    # never reads `heterogeneity_scale` at all (only scenario F has an
    # `increment` for that axis to scale). Perturbing heterogeneity_scale in
    # the axes must not generate a single cell for B at a non-default value
    # -- without this, the grid would report the axis as varied for B when
    # every cell it produced was identical (spec section 5.1's
    # absent-versus-zero confusion).
    assert "heterogeneity_scale" not in SCENARIOS["B"].axes

    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("B",),
            n_units=(500,),
            confounding_strength=(1.0,),
            heterogeneity_scale=(0.0, 1.0, 2.0),
        ),
        panel=None,
        replications=1,
    )

    cells = expand_cells(config)

    default_heterogeneity_scale = ScenarioParams().heterogeneity_scale
    assert all(c.heterogeneity_scale == default_heterogeneity_scale for c in cells)
    # And the axis being declared-away must not shrink the grid to nothing:
    # B's one baseline cell (n_units x confounding_strength, both
    # single-valued here) still has to exist.
    assert len(cells) == 1


def test_the_new_axes_do_not_multiply_the_grid():
    # Hand-computed against spec section 5's enumeration rule: n_units x
    # confounding_strength stays a full cross-product (unaffected by the new
    # axes), and each new axis crosses one non-baseline value at a time,
    # only for the scenarios that declare it (SCENARIOS[id].axes).
    #
    # Scenario A declares {effect_scale, treatment_intercept, noise_scale}
    # (not heterogeneity_scale -- ConstantEffect has no `increment`).
    # Scenario B declares {treatment_intercept, noise_scale} only (its tau is
    # 0, so effect_scale moves nothing and heterogeneity_scale doesn't exist
    # for it either). Both declarations are asserted here so this count
    # cannot silently drift from what the registry actually declares.
    assert SCENARIOS["A"].axes == frozenset(
        {"effect_scale", "treatment_intercept", "noise_scale"}
    )
    assert SCENARIOS["B"].axes == frozenset({"treatment_intercept", "noise_scale"})

    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A", "B"),
            n_units=(500, 2000),
            confounding_strength=(1.0,),
            treatment_intercept=(-1.5, 0.0, 1.5),
            noise_scale=(0.0, 0.9, 1.0),
            effect_scale=(0.5, 1.0, 2.0),
            heterogeneity_scale=(0.0, 1.0, 2.0),
        ),
        panel=None,
        replications=1,
    )

    cells = expand_cells(config)

    # Baseline cross-product: 2 scenarios x 2 n_units x 1 confounding value,
    # each new axis sitting at its ScenarioParams default -> 4 cells.
    baseline = len(("A", "B")) * len((500, 2000)) * len((1.0,))
    # A declares 3 of the 4 new axes, each contributing 2 non-baseline
    # values (treatment_intercept: -1.5/1.5; noise_scale: 0.9/1.0 -- its
    # baseline is 0.0; effect_scale: 0.5/2.0) -> 3 * 2 = 6 extra cells.
    extra_a = 3 * 2
    # B declares 2 of the 4 -> 2 * 2 = 4 extra cells.
    extra_b = 2 * 2
    expected = baseline + extra_a + extra_b

    assert baseline == 4
    assert expected == 14
    assert len(cells) == expected


def test_a_new_axis_cell_holds_every_other_new_axis_at_its_default():
    # The "one at a time" half of the enumeration rule: a cell that varies
    # one axis away from baseline must leave every other new axis, and
    # n_units/confounding_strength, at a single fixed anchor point -- not
    # crossed against the whole existing grid, which is what keeps the extra
    # cell count linear in (scenarios x axes x values) rather than
    # multiplied by every n_units/confounding_strength combination too.
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",),
            n_units=(500, 2000),
            confounding_strength=(1.0, 2.0),
            treatment_intercept=(-1.5, 0.0, 1.5),
        ),
        panel=None,
        replications=1,
    )

    cells = expand_cells(config)
    default = ScenarioParams()
    perturbed = [c for c in cells if c.treatment_intercept != default.treatment_intercept]

    assert perturbed
    for cell in perturbed:
        assert cell.noise_scale == default.noise_scale
        assert cell.effect_scale == default.effect_scale
        assert cell.heterogeneity_scale == default.heterogeneity_scale
        assert cell.n_units == max(config.cross_section.n_units)
        assert cell.confounding_strength == default.confounding_strength


def test_a_widened_axis_needs_its_anchor_present_on_the_confounding_axis():
    # The one-at-a-time cells are pinned to confounding_strength=1.0
    # (ScenarioParams' default). If the config's own confounding axis does not
    # contain 1.0, those cells land at a strength the grid was never asked for
    # and have no baseline twin to be compared against -- the comparison the
    # whole one-at-a-time design exists to support. Refuse it at configuration
    # time rather than emitting cells nothing downstream can interpret.
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",),
            n_units=(500,),
            confounding_strength=(0.5, 2.0),
            treatment_intercept=(-1.5, 0.0, 1.5),
        ),
        panel=None,
        replications=1,
    )

    with pytest.raises(ValueError, match="no.*baseline cell"):
        expand_cells(config)


def test_an_unwidened_grid_does_not_need_the_anchor_present():
    # The check must not fire on a config that widens nothing: a pre-Phase-18
    # grid is free to omit confounding_strength=1.0, and always was. Without
    # this, the fix above would break every existing caller that happens not
    # to include the default strength.
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(500,), confounding_strength=(0.5, 2.0)
        ),
        panel=None,
        replications=1,
    )

    cells = expand_cells(config)

    assert {c.confounding_strength for c in cells} == {0.5, 2.0}


def test_the_anchor_check_ignores_an_axis_no_scenario_declares():
    # Scenario B does not declare heterogeneity_scale, so widening it produces
    # no cells at all for B -- and therefore needs no anchor. A check that
    # fired on the axis's *values* rather than on whether a cell would
    # actually be produced would reject this config for a cell that never
    # gets built.
    assert "heterogeneity_scale" not in SCENARIOS["B"].axes

    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("B",),
            n_units=(500,),
            confounding_strength=(0.5, 2.0),
            heterogeneity_scale=(0.0, 1.0, 2.0),
        ),
        panel=None,
        replications=1,
    )

    assert len(expand_cells(config)) == 2


def test_the_new_axes_are_part_of_the_sort_key_so_ties_cannot_form():
    # `expand_cells` returns `sorted(set(cells), key=cell_sort_key)`. The
    # one-at-a-time cells all share one anchor, so they agree with each other
    # and with their baseline twin on scenario, n_units, confounding_strength
    # AND seed -- every component the sort key had before Phase 18. Drop the
    # new axes from `cell_sort_key` and those cells tie completely; `sorted`
    # is stable, so they then come out in `set` iteration order, which is
    # salted per process. The grid's order is part of its identity (see
    # `expand_cells`' own docstring), so that is a real regression and not a
    # cosmetic one.
    #
    # Checked two ways: the exact expected order in-process, and agreement
    # across processes under several explicit PYTHONHASHSEEDs -- the same
    # guard, and the same idiom, as
    # `test_cell_rng_reproduces_across_processes_with_a_non_default_hash_seed`
    # above. The in-process half alone would pass under whichever hash seed
    # happened to order the tied cells correctly.
    # One source for the fixture, interpolated into both halves. Written
    # twice, widening an axis in one and forgetting the other would leave the
    # subprocess half comparing a different grid against `order` -- and that
    # half is the one actually carrying the guard on most hash seeds.
    #
    # Two axes are widened, not one: with only treatment_intercept varied,
    # nothing distinguishes AXIS_NAMES order from any other ordering of the
    # four, which is a property `cell_sort_key`'s docstring asserts.
    axes_kwargs = {
        "scenarios": ("A",),
        "n_units": (500, 1000),
        "confounding_strength": (1.0,),
        "treatment_intercept": (-1.5, 0.0, 1.5),
        "noise_scale": (0.0, 0.9),
    }
    config = GridConfig(
        cross_section=CrossSectionAxes(**axes_kwargs),
        panel=None,
        replications=1,
    )

    order = [(c.n_units, c.treatment_intercept, c.noise_scale) for c in expand_cells(config)]

    # n_units leads; within the anchor's n_units, treatment_intercept orders
    # the cells that would otherwise be indistinguishable, and noise_scale
    # breaks the remaining tie *after* it. Swap the two axes' positions in
    # the key and (0.0, 0.9) sorts ahead of (1.5, 0.0) instead of behind it,
    # so this pins the order of the block and not merely its presence.
    assert order == [
        (500, 0.0, 0.0),
        (1000, -1.5, 0.0),
        (1000, 0.0, 0.0),
        (1000, 0.0, 0.9),
        (1000, 1.5, 0.0),
    ]

    script = (
        "import json\n"
        "from cp_evaluation.grid import CrossSectionAxes, GridConfig, expand_cells\n"
        f"config = GridConfig(cross_section=CrossSectionAxes(**{axes_kwargs!r}),\n"
        "    panel=None, replications=1)\n"
        "cells = expand_cells(config)\n"
        "print(json.dumps([[c.n_units, c.treatment_intercept, c.noise_scale]\n"
        "                  for c in cells]))\n"
    )
    for hash_seed in ("1", "12345", "98765"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=dict(os.environ, PYTHONHASHSEED=hash_seed),
            capture_output=True,
            text=True,
            check=True,
        )
        assert [tuple(pair) for pair in json.loads(result.stdout)] == order, (
            f"cell order changed under PYTHONHASHSEED={hash_seed}; the sort key "
            f"is not total over the cells this config produces"
        )


def test_every_axis_name_has_a_field_on_the_cell_and_the_axes():
    # `grid` derives `_NEW_AXIS_NAMES` from `AXIS_NAMES` so the two cannot
    # drift -- but deriving alone only moves the failure: a fifth axis added
    # to `ScenarioParams` with no matching field here would surface as a bare
    # `AttributeError` out of `getattr` in `cell_sort_key` or
    # `_identity_payload`, at grid-expansion time, naming neither the axis nor
    # the reason. Fail here instead, where the message says what to add.
    # `metrics.GROUP_COLUMNS` derives from the same tuple for the same
    # reason, which is what makes this worth pinning in one place.
    cell_fields = {f.name for f in dataclasses.fields(CrossSectionCell)}
    axes_fields = {f.name for f in dataclasses.fields(CrossSectionAxes)}

    for name in AXIS_NAMES:
        assert name in cell_fields, f"{name} is in AXIS_NAMES but CrossSectionCell has no field"
        assert name in axes_fields, f"{name} is in AXIS_NAMES but CrossSectionAxes has no field"
