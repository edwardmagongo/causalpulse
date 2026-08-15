"""Reference parity against statsmodels and pysyncon.

Both libraries are ORACLES here, not engines: each appears in the dev
dependency group only, is imported only by this module, and no shipped
package may depend on either. Settled decision #1 of the parent spec is that
estimators are implemented from primitives and *validated* against reference
implementations.

Marked slow and excluded from `make test-fast`. Each reference library's
import is deferred to inside each test body (via `_statsmodels()` /
`_pysyncon()` below) rather than performed at module scope, so under
`pytest -m "not slow"` a deselected test's body never runs and the library is
never imported into `sys.modules` at all -- not just its fit calls skipped.
This is what keeps version drift or an import-time warning in the reference
library (or a transitive dependency such as scipy/numpy/pandas/patsy) from
ever reaching an ordinary `make test-fast` run, where
`filterwarnings = ["error"]` would promote such a warning to a hard failure.
"""

import numpy as np
import pytest
from cp_causal._panel import panel_arrays
from cp_causal.estimators.did import TwoWayFixedEffects, twfe_design
from cp_causal.estimators.synthetic_control import solve_donor_weights
from cp_core.values import EstimatorConfig
from cp_simulation.donor_scenarios import (
    DONOR_N_PERIODS,
    DONOR_N_UNITS,
    DONOR_START,
    build_donor_scenario,
)
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


def _pysyncon():
    """Import pysyncon lazily, from inside a test body only.

    Same reasoning as `_statsmodels` above -- this keeps pysyncon (and the
    matplotlib it pulls in transitively) out of `sys.modules` entirely under
    `pytest -m "not slow"`.

    pysyncon 1.5.x also has a real landmine worth insulating against here: an
    invalid escape sequence in a docstring in its `robust.py` submodule
    raises a `SyntaxWarning` at *compile* time, which only fires on a
    bytecode-cache-less import -- i.e. the first time anything imports
    pysyncon against a fresh `uv sync` (a clean CI checkout, a fresh clone).
    Under this project's `filterwarnings = ["error"]`, that warning is
    promoted to a hard `SyntaxError`, which would make this test fail
    non-deterministically depending on whether some earlier process happened
    to have already imported (and thereby byte-compiled) pysyncon. Verified
    directly: clearing pysyncon's `__pycache__` and importing it under
    pytest's real config reproduces the `SyntaxError`; wrapping just the
    import in a local `SyntaxWarning`-suppressing context, as done here,
    fixes it. Nothing downstream of the import (Dataprep construction, the
    actual fit) is affected -- those still run under the project's normal
    warnings-as-errors policy.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        import pysyncon
    return pysyncon


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


def test_donor_weights_match_pysyncon_on_the_clean_scenario():
    """Pins solve_donor_weights against pysyncon's Synth on S1.

    TwoWayFixedEffects was pinned against statsmodels in Phase 3 because
    parity there is exact and unambiguous -- OLS with fixed effects is
    something statsmodels computes directly. Synthetic control's weight
    solve is a nonconvex-adjacent constrained optimisation with potentially
    multiple near-optimal solutions when the pre-period is close to
    underdetermined, so this test checks that the two solvers reach
    comparably good pre-treatment fits (RMSPE within a relative tolerance),
    not that the weight vectors match elementwise -- two different SLSQP
    starting points and convergence paths can land on different vertices of
    a tied or near-tied optimum.

    pysyncon's `Dataprep` is given one *special predictor* per pre-treatment
    period (that period's own outcome value), on top of the plain mean-of-
    pre-period predictor. That matches `solve_donor_weights`'s own documented
    design of matching the standardised pre-treatment path itself, weighted
    uniformly (see that function's docstring) -- feeding pysyncon only the
    single mean predictor, the more obvious reading of its own example usage,
    would leave its inner optimisation matching just the treated unit's
    pre-period *mean*, not its path, and the two solvers would no longer be
    answering the same question. Measured directly (six seeds on S1): with
    per-period special predictors, pre-period RMSPE agrees to within 0.4%
    between the two solvers -- the `rel=0.2` tolerance below leaves roughly
    50x headroom over that, enough to absorb environment-specific numerical
    drift (BLAS, scipy version) while still catching a solver that regresses
    to a materially worse optimum.
    """
    pysyncon = _pysyncon()

    panel = build_donor_scenario("S1").generate(
        DONOR_N_UNITS, DONOR_N_PERIODS, np.random.default_rng(0)
    )
    frame = panel.dataset.frame
    treated_unit = frame.loc[frame["cohort"] > 0, "unit_id"].iloc[0]
    wide = frame.pivot(index="unit_id", columns="period", values="outcome")
    treated = wide.loc[treated_unit].to_numpy()
    donors = wide.drop(index=treated_unit).to_numpy()
    pre = panel.dataset.periods < DONOR_START
    pre_periods = [int(period) for period in panel.dataset.periods[pre]]

    ours = solve_donor_weights(treated[pre], donors[:, pre])
    our_synthetic = ours @ donors
    our_pre_rmspe = float(np.sqrt(np.mean((treated[pre] - our_synthetic[pre]) ** 2)))

    dataprep = pysyncon.Dataprep(
        foo=frame.rename(columns={"unit_id": "unit", "period": "time", "outcome": "y"}),
        predictors=["y"],
        predictors_op="mean",
        dependent="y",
        unit_variable="unit",
        time_variable="time",
        treatment_identifier=int(treated_unit),
        controls_identifier=[int(unit) for unit in wide.drop(index=treated_unit).index],
        time_predictors_prior=pre_periods,
        time_optimize_ssr=pre_periods,
        special_predictors=[("y", [period], "mean") for period in pre_periods],
    )
    synth = pysyncon.Synth()
    synth.fit(dataprep=dataprep)
    their_pre_rmspe = float(np.sqrt(synth.mspe()))

    # Both should fit the pre-period comparably well; a large relative gap
    # would mean one solver is finding a materially worse optimum.
    assert our_pre_rmspe == pytest.approx(their_pre_rmspe, rel=0.2)
