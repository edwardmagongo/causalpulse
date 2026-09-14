import numpy as np
import pytest
from cp_core.dataset import CausalDataset
from cp_evaluation.grid import CrossSectionCell, PanelCell
from cp_evaluation.resolve import resolve_cell, truth_for
from cp_simulation.panel_scenarios import PANEL_SCENARIOS
from cp_simulation.params import ScenarioParams
from cp_simulation.scenarios import SCENARIOS


def test_a_cross_section_cell_resolves_to_a_dataset_of_the_right_size():
    cell = CrossSectionCell(scenario="D", n_units=400, confounding_strength=1.0, seed=0)

    resolved = resolve_cell(cell, root_entropy=1)

    assert isinstance(resolved.dataset, CausalDataset)
    assert len(resolved.dataset.frame) == 400


def test_a_panel_cell_resolves_to_a_panel_of_the_right_shape():
    cell = PanelCell(scenario="P1", n_units=100, n_periods=6, seed=0)

    resolved = resolve_cell(cell, root_entropy=1)

    assert len(resolved.dataset.frame) == 100 * 6


def test_resolution_is_reproducible():
    cell = CrossSectionCell(scenario="D", n_units=300, confounding_strength=1.0, seed=2)

    first = resolve_cell(cell, root_entropy=5).dataset.frame
    second = resolve_cell(cell, root_entropy=5).dataset.frame

    assert first.equals(second)


def test_a_cross_section_cell_carries_both_ate_and_att_truth():
    cell = CrossSectionCell(scenario="D", n_units=2000, confounding_strength=1.0, seed=0)

    resolved = resolve_cell(cell, root_entropy=1)

    assert set(resolved.truth) == {"ATE", "ATT"}
    assert np.isfinite(resolved.truth["ATE"])
    assert np.isfinite(resolved.truth["ATT"])


def test_a_panel_cell_carries_att_truth_only():
    cell = PanelCell(scenario="P1", n_units=200, n_periods=6, seed=0)

    resolved = resolve_cell(cell, root_entropy=1)

    assert set(resolved.truth) == {"ATT"}


def test_truth_for_returns_the_value_for_the_requested_estimand():
    cell = CrossSectionCell(scenario="D", n_units=2000, confounding_strength=1.0, seed=0)
    resolved = resolve_cell(cell, root_entropy=1)

    assert truth_for(resolved, "ATE") == resolved.truth["ATE"]
    assert truth_for(resolved, "ATT") == resolved.truth["ATT"]


def test_asking_a_panel_cell_for_ate_truth_raises_rather_than_defaulting():
    # A panel cell has no ATE truth. Falling back to the ATT value would
    # manufacture bias that is not there -- the same defect family as the
    # Phase 9 bsts bug and Phase 10's HIGH 2, where a quantity was built from
    # fields that did not mean, for that estimator, what the formula assumed.
    cell = PanelCell(scenario="P1", n_units=200, n_periods=6, seed=0)
    resolved = resolve_cell(cell, root_entropy=1)

    with pytest.raises(KeyError, match="ATE"):
        truth_for(resolved, "ATE")


def test_truth_is_measured_not_declared():
    # `declared_effect` is a label. Ground truth is measured from the realised
    # potential outcomes, and on a finite sample the two differ.
    cell = CrossSectionCell(scenario="D", n_units=500, confounding_strength=1.0, seed=0)
    resolved = resolve_cell(cell, root_entropy=1)

    declared = SCENARIOS["D"].declared_effect(ScenarioParams())
    assert resolved.truth["ATE"] != declared


def test_panel_truth_is_measured_not_declared():
    # P1's constant effect is added identically to every unit-period with no
    # sampling noise in the effect itself, so its measured ATT equals its
    # declared 1.0 exactly at any sample size -- that equality is a property
    # of the scenario, not evidence the pipeline reads `declared_effect`.
    # P4's declared 2.8361 is itself a measured, cohort-size-weighted average
    # (see its registry docstring), so a *different* finite sample draws
    # different cohort sizes and its measured overall ATT does not coincide
    # with the declared value. n_periods=8 is P4's minimum: its staggered
    # cohorts adopt at periods (3, 5, 7), which need at least one post-period
    # for the last cohort.
    cell = PanelCell(scenario="P4", n_units=100, n_periods=8, seed=0)
    resolved = resolve_cell(cell, root_entropy=1)

    declared = PANEL_SCENARIOS["P4"].declared_effect
    assert resolved.truth["ATT"] != declared
