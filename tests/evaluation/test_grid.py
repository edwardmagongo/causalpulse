import pytest
from cp_evaluation.grid import (
    CrossSectionAxes,
    CrossSectionCell,
    GridConfig,
    PanelAxes,
    PanelCell,
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
