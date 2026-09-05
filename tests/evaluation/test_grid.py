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
            scenarios=("A",), n_units=(500,), confounding_strength=(1.0,)
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

    shared = CrossSectionCell(
        scenario="A", n_units=500, confounding_strength=1.0, seed=0
    )
    assert shared in expand_cells(small)
    assert shared in expand_cells(larger)

    assert np.array_equal(
        cell_rng(shared, root_entropy=7).normal(size=5),
        cell_rng(shared, root_entropy=7).normal(size=5),
    )


def test_the_two_shapes_cannot_collide_on_one_stream():
    # A cross-sectional cell and a panel cell with the same numbers must not
    # share a stream just because their field values happen to coincide.
    x = CrossSectionCell(scenario="X", n_units=200, confounding_strength=6.0, seed=0)
    p = PanelCell(scenario="X", n_units=200, n_periods=6, seed=0)

    assert not np.array_equal(
        cell_rng(x, root_entropy=1).normal(size=5),
        cell_rng(p, root_entropy=1).normal(size=5),
    )
