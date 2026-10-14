"""Reference parity against statsmodels, pysyncon and econml.

All three libraries are ORACLES here, not engines: each appears in the dev
dependency group only, is imported only by this module, and no shipped
package may depend on any of them. A settled decision is
that estimators are implemented from primitives and *validated* against
reference implementations.

Marked slow and excluded from `make test-fast`. Each reference library's
import is deferred to inside each test body (via `_statsmodels()`,
`_pysyncon()` and `_econml_causal_forest_dml()` below) rather than performed
at module scope, so under
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
from cp_causal.estimators.causal_forest import fit_forest
from cp_causal.estimators.did import TwoWayFixedEffects, twfe_design
from cp_causal.estimators.synthetic_control import solve_donor_weights
from cp_core.values import EstimatorConfig
from cp_simulation.cate_scenarios import CATE_N_UNITS, build_cate_scenario
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


def _econml_causal_forest_dml():
    """Import econml's CausalForestDML lazily, from inside a test body only.

    Same reasoning as `_statsmodels` above. econml is the heaviest oracle in
    this file -- it pulls in sklearn, scipy, lightgbm and numba transitively --
    so keeping it out of collection is what stops any deprecation warning
    anywhere in that stack from being promoted to a hard failure by this
    project's `filterwarnings = ["error"]` during an ordinary
    `make test-fast`. econml 0.17.0 imports cleanly under `-W error` today;
    this defers the import anyway, because the point of the convention is that
    a future version's warning cannot reach the fast suite in the first place.
    """
    return pytest.importorskip("econml.dml").CausalForestDML


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


def test_hand_rolled_causal_forest_agrees_with_econml_on_rich_heterogeneity():
    """Pins fit_forest's per-unit CATE against econml's CausalForestDML.

    Rough agreement, not elementwise parity -- and for a stronger reason than
    the pysyncon test's tied-optimum argument above. TwoWayFixedEffects and
    statsmodels solve the same OLS problem, so parity there is exact.
    solve_donor_weights and pysyncon solve the same optimisation problem from
    different starting points, so RMSPE is compared instead of weights. Here
    the two implementations do not share a problem at all: both are honest
    causal forests, but they differ in splitting criterion, honesty scheme,
    subsampling rate and nuisance-model choice. Correlation across units is
    the strongest claim the pair actually supports, and it is a real one --
    an implementation that had stopped ordering units by their true effect
    would lose it.
    """
    causal_forest_dml = _econml_causal_forest_dml()
    simulated = build_cate_scenario("RICH").generate(
        CATE_N_UNITS, np.random.default_rng(0)
    )
    frame = simulated.dataset.frame
    schema = simulated.dataset.schema
    covariates = frame[list(schema.covariates)].to_numpy(dtype=float)
    treatment = frame[schema.treatment].to_numpy()
    outcome = frame[schema.outcome].to_numpy(dtype=float)

    ours = fit_forest(
        covariates,
        treatment,
        outcome,
        n_trees=200,
        min_leaf_structure=20,
        max_depth=4,
        subsample_frac=0.5,
        cross_fit_folds=5,
        rng=np.random.default_rng(0),
    )
    theirs = causal_forest_dml(
        n_estimators=200, min_samples_leaf=20, max_depth=4, random_state=0
    )
    theirs.fit(outcome, treatment, X=covariates)

    agreement = float(np.corrcoef(ours.cate, theirs.effect(covariates))[0, 1])
    # Measured through the shipped `fit_forest` at exactly this protocol:
    # agreement 0.9580. Against the scenario's own measured `true_cate` the
    # two score 0.6686 (ours) and 0.6335 (econml) -- so the pair agree with
    # each other far more closely than either agrees with truth, which is
    # what makes this a test of implementation drift rather than a second,
    # weaker recovery test. Recovery is scored separately and against
    # measured truth in `test_causal_forest_recovery.py`.
    #
    # The 0.6 bound is deliberately far below the measured 0.9580. It is set
    # to catch a structural divergence, not to police the tenths: the two
    # implementations do not share hyperparameter semantics (econml's
    # `min_samples_leaf` and this forest's `min_leaf_structure` do not count
    # the same units), so an econml upgrade could legitimately move the
    # measured number without anything here being wrong.
    assert agreement > 0.6, (
        f"agreement with econml {agreement:.4f} -- implementations have diverged"
    )
