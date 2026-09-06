import pytest
from cp_evaluation.grid import expand_cells
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
