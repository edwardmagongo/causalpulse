"""Reference parity against statsmodels.

statsmodels is an ORACLE here, not an engine: it appears in the dev dependency
group only, is imported only by this module, and no shipped package may depend
on it. Settled decision #1 of the parent spec is that estimators are implemented
from primitives and *validated* against reference implementations.

Marked slow and excluded from `make test-fast`. The `import statsmodels.api`
call itself is deferred to inside each test body (via `_statsmodels()` below)
rather than performed at module scope, so under `pytest -m "not slow"` a
deselected test's body never runs and statsmodels is never imported into
`sys.modules` at all -- not just its `.fit()` calls skipped. This is what
keeps version drift or an import-time warning in statsmodels (or a transitive
dependency such as scipy/numpy/pandas/patsy) from ever reaching an ordinary
`make test-fast` run, where `filterwarnings = ["error"]` would promote such a
warning to a hard failure.
"""

import numpy as np
import pytest
from cp_causal._panel import panel_arrays
from cp_causal.estimators.did import TwoWayFixedEffects, twfe_design
from cp_core.values import EstimatorConfig
from cp_simulation.panel_scenarios import build_panel_scenario

pytestmark = pytest.mark.slow

CONFIG = EstimatorConfig(bootstrap_reps=10, random_seed=1)


def _statsmodels():
    """Import statsmodels.api lazily, from inside a test body only.

    Keeping this call out of module scope means it only ever executes when a
    test that calls it is actually selected and run -- never during
    collection, and never under `pytest -m "not slow"`.
    """
    return pytest.importorskip("statsmodels.api")


def panel(scenario_id="P3", seed=0, n_units=120):
    return build_panel_scenario(scenario_id).generate(
        n_units=n_units, n_periods=8, rng=np.random.default_rng(seed)
    )


def test_twfe_point_estimate_matches_statsmodels_ols():
    statsmodels = _statsmodels()
    simulated = panel()
    arrays = panel_arrays(simulated.dataset)

    ours = TwoWayFixedEffects().fit(simulated.dataset, CONFIG)
    theirs = statsmodels.OLS(arrays.outcome, twfe_design(arrays)).fit()

    assert ours.point == pytest.approx(float(theirs.params[1]), abs=1e-8)


def test_twfe_cluster_robust_standard_error_matches_statsmodels():
    # The sandwich is hand-rolled in cp_statistics and has been under test since
    # Phase 1 against its own algebra. This is the first time it is checked
    # against an independent implementation of the same estimator.
    statsmodels = _statsmodels()
    simulated = panel()
    arrays = panel_arrays(simulated.dataset)

    ours = TwoWayFixedEffects().fit(simulated.dataset, CONFIG)
    theirs = statsmodels.OLS(arrays.outcome, twfe_design(arrays)).fit(
        cov_type="cluster", cov_kwds={"groups": arrays.unit_index}
    )

    assert ours.se == pytest.approx(float(theirs.bse[1]), rel=1e-6)


@pytest.mark.parametrize("scenario_id", ["P1", "P3", "P4"])
def test_parity_holds_across_adoption_designs(scenario_id):
    statsmodels = _statsmodels()
    simulated = panel(scenario_id)
    arrays = panel_arrays(simulated.dataset)

    ours = TwoWayFixedEffects().fit(simulated.dataset, CONFIG)
    theirs = statsmodels.OLS(arrays.outcome, twfe_design(arrays)).fit(
        cov_type="cluster", cov_kwds={"groups": arrays.unit_index}
    )

    assert ours.point == pytest.approx(float(theirs.params[1]), abs=1e-8)
    assert ours.se == pytest.approx(float(theirs.bse[1]), rel=1e-6)
