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
