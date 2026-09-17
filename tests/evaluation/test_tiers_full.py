import pytest
from cp_evaluation.grid import CrossSectionCell, expand_cells
from cp_evaluation.tiers import (
    CI_ESTIMATORS,
    FULL_ESTIMATORS,
    FULL_FIT_CONFIG,
    ci_config,
    full_config,
)


def test_full_covers_every_cross_sectional_scenario_and_all_but_p7():
    from cp_simulation.panel_scenarios import PANEL_SCENARIOS
    from cp_simulation.scenarios import SCENARIOS

    config = full_config()

    assert set(config.cross_section.scenarios) == set(SCENARIOS)
    # P7 is deliberately excluded: it is generated at its own longer
    # protocol (HIER_N_UNITS=600, HIER_N_PERIODS=20, per
    # cp_simulation.panel_scenarios) and needs n_periods >= 16, but
    # PanelAxes applies one n_periods list to every scenario it carries.
    # Including P7 would force P1-P6 onto that far slower protocol just to
    # accommodate one scenario, so it is left out and every other panel
    # scenario is kept. See full_config's docstring for the measurement
    # that found this.
    assert set(config.panel.scenarios) == set(PANEL_SCENARIOS) - {"P7"}


def test_full_is_strictly_wider_than_ci():
    # If `full` were not a superset the two tiers would answer different
    # questions, and a band measured on one would say nothing about the other.
    assert len(expand_cells(full_config())) > len(expand_cells(ci_config()))


def test_full_uses_the_default_bootstrap_replications():
    # `ci` runs at 200 to fit a two-minute gate. `full` has no such budget and
    # should not inherit the compromise.
    assert FULL_FIT_CONFIG.bootstrap_reps == 1000


def test_full_includes_the_estimators_ci_had_to_drop():
    assert "causal_forest" in FULL_ESTIMATORS
    assert "hierarchical_did" in FULL_ESTIMATORS
    assert set(CI_ESTIMATORS) <= set(FULL_ESTIMATORS)


def test_full_excludes_the_donor_pool_estimators():
    # They need a donor-pool cell shape this phase does not add; including them
    # would produce nothing but `ineligible` rows.
    assert "synthetic_control" not in FULL_ESTIMATORS
    assert "bsts" not in FULL_ESTIMATORS


@pytest.mark.parametrize("name", FULL_ESTIMATORS)
def test_every_named_estimator_is_registered(name):
    from cp_causal.registry import all_estimators

    assert name in all_estimators()


def test_full_config_produces_the_designs_measured_84_cross_sectional_cells():
    # docs/specs/2026-09-13-causal-core-phase18-design.md section 12: the
    # original 42 cells (7 scenarios x n_units=(500, 2000) x
    # confounding_strength=(0.5, 1.0, 2.0)) plus 42 new one-at-a-time cells,
    # 2 per axis per scenario that declares it (A: 6, B: 4, C: 6, D: 6, F: 8,
    # G: 6, H: 6). Counted at seed == 0 only: replications multiplies every
    # distinct cell identically, so it would inflate this count without
    # testing the enumeration itself.
    config = full_config()

    distinct_cells = {
        cell
        for cell in expand_cells(config)
        if isinstance(cell, CrossSectionCell) and cell.seed == 0
    }

    assert len(distinct_cells) == 84
