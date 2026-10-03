import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from cp_evaluation.grid import (
    CateAxes,
    CateCell,
    CrossSectionAxes,
    CrossSectionCell,
    DonorPoolAxes,
    DonorPoolCell,
    GridConfig,
    LongPanelAxes,
    LongPanelCell,
    PanelAxes,
    PanelCell,
    cell_key,
    cell_rng,
    expand_cells,
)
from cp_simulation.params import AXIS_NAMES, ScenarioParams
from cp_simulation.scenarios import SCENARIOS

# `cell_key(DonorPoolCell("S1", 0))` on `cate-metrics-arm` before Task 3's edit.
_DONOR_S1_0_KEY = "f2f8e3cc570c30a4c802daea"
# `cell_key(CateCell("RICH", 0))` on `p7-arm` before Task 1's edit.
_CATE_RICH_0_KEY = "bb8c7637c6c88a09d0df3d2f"


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


def test_donor_pool_axes_expand_one_cell_per_scenario_per_replication():
    config = GridConfig(
        cross_section=None,
        panel=None,
        donor_pool=DonorPoolAxes(scenarios=("S0", "S1", "S2")),
        replications=4,
    )

    cells = expand_cells(config)

    assert len(cells) == 3 * 4
    assert all(isinstance(c, DonorPoolCell) for c in cells)
    assert {c.scenario for c in cells} == {"S0", "S1", "S2"}
    assert {c.seed for c in cells} == {0, 1, 2, 3}


def test_all_three_shapes_expand_independently_and_concatenate():
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(500,), confounding_strength=(1.0,)
        ),
        panel=PanelAxes(scenarios=("P1",), n_units=(200,), n_periods=(6,)),
        donor_pool=DonorPoolAxes(scenarios=("S1",)),
        replications=2,
    )

    cells = expand_cells(config)

    assert len(cells) == 6
    assert sum(isinstance(c, CrossSectionCell) for c in cells) == 2
    assert sum(isinstance(c, PanelCell) for c in cells) == 2
    assert sum(isinstance(c, DonorPoolCell) for c in cells) == 2


def test_an_unknown_donor_pool_scenario_is_refused_by_name():
    config = GridConfig(
        cross_section=None,
        panel=None,
        donor_pool=DonorPoolAxes(scenarios=("S1", "S99")),
        replications=1,
    )

    with pytest.raises(ValueError, match="S99"):
        expand_cells(config)


def test_a_grid_of_only_donor_pool_axes_is_not_refused():
    config = GridConfig(
        cross_section=None, panel=None, donor_pool=DonorPoolAxes(scenarios=("S1",)), replications=1
    )

    cells = expand_cells(config)

    assert len(cells) == 1


def test_donor_pool_cells_sort_after_panel_cells():
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(500,), confounding_strength=(1.0,)
        ),
        panel=PanelAxes(scenarios=("P1",), n_units=(200,), n_periods=(6,)),
        donor_pool=DonorPoolAxes(scenarios=("S1",)),
        replications=1,
    )

    cells = expand_cells(config)

    shapes = [type(c).__name__ for c in cells]
    order = {"CrossSectionCell": 0, "PanelCell": 1, "DonorPoolCell": 2}
    assert shapes == sorted(shapes, key=order.__getitem__)


def test_the_same_donor_pool_cell_always_gets_the_same_stream():
    cell = DonorPoolCell(scenario="S1", seed=0)

    first = cell_rng(cell, root_entropy=1).normal(size=5)
    second = cell_rng(cell, root_entropy=1).normal(size=5)

    assert np.array_equal(first, second)


def test_donor_pool_cells_do_not_collide_with_the_other_two_shapes():
    d = DonorPoolCell(scenario="X", seed=0)
    p = PanelCell(scenario="X", n_units=18, n_periods=24, seed=0)
    x = CrossSectionCell(scenario="X", n_units=18, confounding_strength=1.0, seed=0)

    draws = [tuple(cell_rng(c, root_entropy=1).normal(size=5)) for c in (d, p, x)]

    assert len(set(draws)) == 3


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


def test_donor_pool_cell_rng_reproduces_across_processes_with_a_non_default_hash_seed():
    # The same guard as the cross-section test above, for the third shape: a
    # DonorPoolCell's stream must come from sha256, not the salted builtin hash.
    cell = DonorPoolCell("S1", 0)
    expected = cell_rng(cell, root_entropy=12345).normal(size=5).tolist()

    script = (
        "import json\n"
        "from cp_evaluation.grid import DonorPoolCell, cell_rng\n"
        "draws = cell_rng(DonorPoolCell('S1', 0), root_entropy=12345).normal(size=5)\n"
        "print(json.dumps(draws.tolist()))\n"
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )

    assert np.array_equal(expected, json.loads(result.stdout))


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


def test_a_scenario_is_not_crossed_against_a_confounding_strength_it_does_not_declare():
    # `confounding_strength` predates the other four axes and was exempt from
    # the declaration rule through Phase 18: every cross-sectional scenario
    # was crossed against every strength, declared or not. Scenarios A, B, C
    # and F assign at random, so their builders never read the axis, and those
    # cells differed from each other only in RNG stream -- while `summarise`
    # grouped them as though the strength had been varied. That is spec
    # section 5.1's absent-versus-zero shape, for the axis section 5 left
    # ungated.
    #
    # A declares none of it and D declares it, so one config shows both sides
    # of the rule and neither can pass by accident of the other.
    assert "confounding_strength" not in SCENARIOS["A"].axes
    assert "confounding_strength" in SCENARIOS["D"].axes

    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A", "D"),
            n_units=(500, 2000),
            confounding_strength=(0.5, 1.0, 2.0),
        ),
        panel=None,
        replications=1,
    )

    cells = expand_cells(config)

    # Hand-derived from the enumeration rule, not read off the output: A
    # contributes |n_units| x 1 baseline strength = 2 cells, D contributes
    # |n_units| x |confounding_strength| = 2 x 3 = 6, so 8 in total. Ungated,
    # A would contribute 6 too and the total would be 12.
    assert len(cells) == 2 + 6

    by_scenario = {"A": [], "D": []}
    for cell in cells:
        by_scenario[cell.scenario].append(cell.confounding_strength)

    # The one value A gets is `ScenarioParams`' own default -- the same value
    # `_expand_cross_section` anchors its one-at-a-time cells on, and the same
    # treatment an undeclared one of the four new axes already gets. Not, say,
    # the first or smallest value on the config's axis, which would make an
    # undeclared scenario's data depend on an axis it does not read.
    assert set(by_scenario["A"]) == {ScenarioParams().confounding_strength}
    assert set(by_scenario["D"]) == {0.5, 1.0, 2.0}


def test_an_undeclared_confounding_scenario_never_needs_the_anchor():
    # The other half of `_check_the_anchor_is_on_the_grid`'s per-scenario
    # question. A widened new axis anchors on confounding_strength=1.0, and a
    # config omitting 1.0 is refused for a scenario that declares the axis
    # (see `test_a_widened_axis_needs_its_anchor_present_on_the_confounding_axis`).
    # For a scenario that does not declare it, the whole baseline block is at
    # 1.0 by construction, so the anchor always has its baseline twin and the
    # config is fine. A check left asking the question once for the whole
    # config -- as it correctly did while the axis was ungated -- would reject
    # this grid even though every cell in it is interpretable.
    assert "confounding_strength" not in SCENARIOS["A"].axes

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

    cells = expand_cells(config)

    # 1 baseline cell (1 n_units x 1 baseline strength) + 2 perturbed
    # treatment_intercept cells, all three at the default strength.
    assert len(cells) == 1 + 2
    assert {c.confounding_strength for c in cells} == {
        ScenarioParams().confounding_strength
    }
    assert {c.treatment_intercept for c in cells} == {-1.5, 0.0, 1.5}


def test_the_new_axes_do_not_multiply_the_grid():
    # Hand-computed against the enumeration rule: n_units x
    # confounding_strength is crossed rather than varied one value at a time
    # (unaffected by the new axes), and each new axis crosses one non-baseline
    # value at a time, only for the scenarios that declare it
    # (SCENARIOS[id].axes).
    #
    # Deliberately not "spec section 5's rule": section 5 states that
    # cross-product ungated, which the confounding_strength follow-up
    # superseded (see that section's own banner, and section 12). This config
    # cannot tell the two apart -- its confounding axis is the single default
    # value, where gated and ungated agree -- so citing the superseded rule
    # here would go unflagged by the test itself.
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
    #
    # Scenario D, not A. "Pinned to one strength rather than crossed against
    # them all" is only a property a scenario that *has* several strengths can
    # exercise, and since `confounding_strength` became a declared axis A has
    # exactly one: its baseline block sits at the default whatever this
    # config's confounding axis says. Measured rather than assumed -- a
    # mutation that crossed each perturbed cell against the scenario's own
    # strengths instead of pinning it to the anchor fails this test with
    # `scenarios=("D",)` and passes it with `scenarios=("A",)`. D declares the
    # axis, so (1.0, 2.0) genuinely varies for it and the pinning has
    # something to get wrong.
    assert "confounding_strength" in SCENARIOS["D"].axes
    assert "treatment_intercept" in SCENARIOS["D"].axes

    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("D",),
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
    #
    # Scenario D, not A: since `confounding_strength` became a declared axis
    # like any other, only a scenario that declares it can be left without a
    # baseline twin. A's baseline block sits at 1.0 whatever this axis says,
    # so A is the case that must *not* raise -- covered by
    # `test_an_undeclared_confounding_scenario_never_needs_the_anchor` below.
    assert "confounding_strength" in SCENARIOS["D"].axes

    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("D",),
            n_units=(500,),
            confounding_strength=(0.5, 2.0),
            treatment_intercept=(-1.5, 0.0, 1.5),
        ),
        panel=None,
        replications=1,
    )

    with pytest.raises(ValueError, match="no.*baseline cell"):
        expand_cells(config)


def test_the_anchor_check_keeps_looking_past_an_undeclared_scenario():
    # `_check_the_anchor_is_on_the_grid` walks every scenario and `continue`s
    # past one that does not declare `confounding_strength`, because such a
    # scenario's baseline block is at the anchor by construction. `continue`,
    # not `return`: a later scenario in the same config may still declare the
    # axis and still be left without a baseline twin.
    #
    # Every other test that omits 1.0 from the confounding axis passes a
    # single scenario, so none of them can tell the two apart. This one puts
    # the undeclared scenario FIRST, where a `return` would swallow the check
    # before D is ever reached. Verified to fail that way: with `return` in
    # place of `continue`, this config expands silently and D's two
    # `effect_scale` cells land at confounding_strength=1.0, a strength D's
    # own baseline block (0.5 and 2.0) never contains -- precisely the
    # orphaned cells the check exists to refuse.
    assert "confounding_strength" not in SCENARIOS["A"].axes
    assert "confounding_strength" in SCENARIOS["D"].axes
    assert "effect_scale" in SCENARIOS["D"].axes

    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A", "D"),
            n_units=(500,),
            confounding_strength=(0.5, 2.0),
            effect_scale=(0.5, 1.0, 2.0),
        ),
        panel=None,
        replications=1,
    )

    # The scenario is named, so the message points at the one that actually
    # needs the anchor rather than at the config as a whole.
    with pytest.raises(ValueError, match="scenario D.*no.*baseline cell"):
        expand_cells(config)


def test_an_unwidened_grid_does_not_need_the_anchor_present():
    # The check must not fire on a config that widens nothing: a pre-Phase-18
    # grid is free to omit confounding_strength=1.0, and always was. Without
    # this, the fix above would break every existing caller that happens not
    # to include the default strength.
    #
    # Scenario D, not A: the assertion is that the config's own two strengths
    # both reach the output, which only a scenario that declares the axis can
    # show. A declares none of it, so its cells sit at the default 1.0 and
    # would fail this for a reason that has nothing to do with the anchor
    # check -- see
    # `test_a_scenario_is_not_crossed_against_a_confounding_strength_it_does_not_declare`.
    assert "confounding_strength" in SCENARIOS["D"].axes

    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("D",), n_units=(500,), confounding_strength=(0.5, 2.0)
        ),
        panel=None,
        replications=1,
    )

    cells = expand_cells(config)

    assert {c.confounding_strength for c in cells} == {0.5, 2.0}


def test_the_anchor_check_ignores_an_axis_no_scenario_declares():
    # Scenario D does not declare heterogeneity_scale, so widening it produces
    # no cells at all for D -- and therefore needs no anchor. A check that
    # fired on the axis's *values* rather than on whether a cell would
    # actually be produced would reject this config for a cell that never
    # gets built.
    #
    # Scenario D, not B: B declares no `confounding_strength` either, so under
    # the declaration gating it would pass this for two independent reasons
    # and the anchor check's own behaviour would go untested. D is the case
    # where the check is live -- it declares the axis, the config omits 1.0 --
    # and must still stay silent because the widened axis produces nothing.
    assert "heterogeneity_scale" not in SCENARIOS["D"].axes
    assert "confounding_strength" in SCENARIOS["D"].axes

    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("D",),
            n_units=(500,),
            confounding_strength=(0.5, 2.0),
            heterogeneity_scale=(0.0, 1.0, 2.0),
        ),
        panel=None,
        replications=1,
    )

    # D's baseline block alone: 1 n_units x its 2 declared strengths.
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


def test_axis_baselines_matches_scenario_params_defaults():
    # AXIS_BASELINES is what cp_figures reads to decide which rows are
    # "every other axis at its default". If it ever drifted from
    # ScenarioParams' own defaults, the figure would silently select a
    # slice that is not the one-at-a-time slice the grid emitted -- with
    # no error, just the wrong rows.
    from cp_evaluation.grid import AXIS_BASELINES
    from cp_simulation.params import AXIS_NAMES, ScenarioParams

    defaults = ScenarioParams()

    assert tuple(AXIS_BASELINES) == AXIS_NAMES
    for name in AXIS_NAMES:
        assert AXIS_BASELINES[name] == getattr(defaults, name)


def test_axis_baselines_cannot_be_mutated_by_a_caller():
    # A plain dict here would let any importer rewrite what the grid
    # considers unperturbed, for every other importer in the process.
    from cp_evaluation.grid import AXIS_BASELINES

    with pytest.raises(TypeError):
        AXIS_BASELINES["effect_scale"] = 99.0  # type: ignore[index]


def test_scenario_axes_matches_the_scenario_registry():
    # SCENARIO_AXES is what cp_figures reads to decide which axes a
    # scenario declares. If it ever drifted from SCENARIOS' own axes, the
    # declaration refusal would silently pass a profile it should refuse,
    # or refuse one it should allow -- with no error, just the wrong
    # verdict.
    from cp_evaluation.grid import SCENARIO_AXES
    from cp_simulation.scenarios import SCENARIOS

    assert set(SCENARIO_AXES) == set(SCENARIOS)
    for scenario_id, spec in SCENARIOS.items():
        assert SCENARIO_AXES[scenario_id] == spec.axes


def test_scenario_axes_cannot_be_mutated_by_a_caller():
    # A plain dict here would let any importer rewrite what the grid
    # considers "declared", for every other importer in the process.
    from cp_evaluation.grid import SCENARIO_AXES

    with pytest.raises(TypeError):
        SCENARIO_AXES["C"] = frozenset()  # type: ignore[index]


def _cate_only(scenarios=("RICH", "NULL", "NOISY"), replications=3) -> GridConfig:
    return GridConfig(
        cross_section=None,
        panel=None,
        cate=CateAxes(scenarios=scenarios),
        replications=replications,
    )


def test_cate_axes_expand_one_cell_per_scenario_per_replication():
    cells = expand_cells(_cate_only())

    assert len(cells) == 9
    assert all(isinstance(cell, CateCell) for cell in cells)
    assert [(c.scenario, c.seed) for c in cells] == [
        (s, seed) for s in ("NOISY", "NULL", "RICH") for seed in range(3)
    ]


def test_a_grid_with_only_a_cate_arm_is_valid():
    assert expand_cells(_cate_only(scenarios=("RICH",), replications=1)) == (
        CateCell("RICH", 0),
    )


def test_an_unknown_cate_scenario_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown CATE scenarios"):
        expand_cells(_cate_only(scenarios=("RICH", "WIDE")))


def test_cate_cells_sort_after_every_other_shape():
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(200,), confounding_strength=(1.0,)
        ),
        panel=PanelAxes(scenarios=("P1",), n_units=(50,), n_periods=(7,)),
        donor_pool=DonorPoolAxes(scenarios=("S1",)),
        cate=CateAxes(scenarios=("RICH",)),
        replications=1,
    )

    shapes = [type(cell) for cell in expand_cells(config)]

    assert shapes == [CrossSectionCell, PanelCell, DonorPoolCell, CateCell]


def test_a_cate_cell_and_a_donor_cell_with_the_same_fields_never_share_a_stream():
    donor = DonorPoolCell("S1", 0)
    cate = CateCell("S1", 0)

    assert cell_key(donor) != cell_key(cate)
    assert not np.array_equal(
        cell_rng(donor, root_entropy=1).normal(size=5),
        cell_rng(cate, root_entropy=1).normal(size=5),
    )


def test_adding_the_cate_shape_does_not_move_a_donor_cells_stream():
    # Pinned before Task 3: the donor branch used to be the fall-through of
    # `_identity_payload`; making it explicit must not change its payload.
    assert cell_key(DonorPoolCell("S1", 0)) == _DONOR_S1_0_KEY


def test_cate_cell_rng_reproduces_across_processes_with_a_non_default_hash_seed():
    cell = CateCell("RICH", 0)
    expected = cell_rng(cell, root_entropy=12345).normal(size=5).tolist()

    script = (
        "import json\n"
        "from cp_evaluation.grid import CateCell, cell_rng\n"
        "draws = cell_rng(CateCell('RICH', 0), root_entropy=12345).normal(size=5)\n"
        "print(json.dumps(draws.tolist()))\n"
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )

    assert np.array_equal(expected, json.loads(result.stdout))


def _p7_only(replications=3) -> GridConfig:
    return GridConfig(
        cross_section=None,
        panel=None,
        long_panel=LongPanelAxes(scenarios=("P7",)),
        replications=replications,
    )


def test_long_panel_axes_expand_one_cell_per_replication():
    assert expand_cells(_p7_only()) == tuple(LongPanelCell("P7", s) for s in range(3))


def test_a_scenario_without_a_fixed_protocol_is_refused_by_name():
    with pytest.raises(ValueError, match="no fixed protocol"):
        expand_cells(
            GridConfig(
                cross_section=None,
                panel=None,
                long_panel=LongPanelAxes(scenarios=("P7", "P1")),
                replications=1,
            )
        )


def test_long_panel_cells_sort_after_every_other_shape():
    config = GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(200,), confounding_strength=(1.0,)
        ),
        panel=PanelAxes(scenarios=("P1",), n_units=(50,), n_periods=(7,)),
        donor_pool=DonorPoolAxes(scenarios=("S1",)),
        cate=CateAxes(scenarios=("RICH",)),
        long_panel=LongPanelAxes(scenarios=("P7",)),
        replications=1,
    )

    shapes = [type(c) for c in expand_cells(config)]

    assert shapes == [CrossSectionCell, PanelCell, DonorPoolCell, CateCell, LongPanelCell]


def test_adding_the_long_panel_shape_moves_no_existing_stream():
    assert cell_key(DonorPoolCell("S1", 0)) == _DONOR_S1_0_KEY
    assert cell_key(CateCell("RICH", 0)) == _CATE_RICH_0_KEY


def test_a_long_panel_cell_never_shares_a_stream_with_a_donor_or_cate_cell():
    long = LongPanelCell("S1", 0)

    assert cell_key(long) not in {cell_key(DonorPoolCell("S1", 0)), cell_key(CateCell("S1", 0))}


def test_long_panel_cell_rng_reproduces_across_processes_with_a_non_default_hash_seed():
    expected = cell_rng(LongPanelCell("P7", 0), root_entropy=12345).normal(size=5).tolist()
    script = (
        "import json\n"
        "from cp_evaluation.grid import LongPanelCell, cell_rng\n"
        "print(json.dumps(cell_rng(LongPanelCell('P7', 0), root_entropy=12345)"
        ".normal(size=5).tolist()))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, PYTHONHASHSEED="12345"),
        capture_output=True,
        text=True,
        check=True,
    )

    assert np.array_equal(expected, json.loads(result.stdout))
