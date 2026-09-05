"""Can this benchmark detect a broken estimator?

Every other test in tests/evaluation/ checks a metric against a hand-computed
value on a synthetic frame. Those prove the arithmetic. They do not prove the
instrument works end to end, because a grid that silently scored against the
wrong truth, or dropped every inconvenient row, would still pass them.

This file registers estimators that are wrong in known, specific ways and
asserts the grid says so. A benchmark that cannot fail proves nothing.
"""

import pytest
from cp_causal.base import Requirements
from cp_causal.registry import register
from cp_core.results import EffectEstimate, FitMetadata
from cp_core.values import Interval
from cp_evaluation.grid import CrossSectionAxes, GridConfig
from cp_evaluation.metrics import summarise
from cp_evaluation.run import run_grid

BIAS = 0.25

# Every assertion in this file concerns only these three methods -- the two
# seeded-defect estimators and the one honest baseline they are compared
# against. Restricting `run_grid` to this set keeps it from also fitting
# every other registered estimator (including `causal_forest`, an econml
# forest, and a bootstrapped `aipw`) across 40 cells at n_units=600: minutes
# of work this file never asserts on.
_ASSERTED_ESTIMATORS = ("difference_in_means", "seeded_biased", "seeded_overconfident")

CROSS_SECTIONAL_REQUIREMENTS = Requirements(
    needs_panel=False,
    min_pre_periods=0,
    min_post_periods=0,
    needs_covariates=False,
    min_treated_units=2,
    min_control_units=2,
    needs_uniform_adoption=False,
    max_treated_units=None,
    supports_outcome_types=frozenset({"binary", "continuous"}),
)


def _estimate(name: str, point: float, half_width: float, dataset) -> EffectEstimate:  # noqa: ANN001
    treated = int((dataset.frame[dataset.schema.treatment] == 1).sum())
    control = int((dataset.frame[dataset.schema.treatment] == 0).sum())
    return EffectEstimate(
        estimand="ATE",
        method=name,
        point=point,
        se=half_width / 1.96,
        ci=Interval(lower=point - half_width, upper=point + half_width, level=0.95),
        ci_basis="analytic",
        bootstrap_ci=None,
        p_value=None,
        n_treated=treated,
        n_control=control,
        scale="percentage_point",
        diagnostics={},
        assumptions=(),
        counterfactual=None,
        fit_metadata=FitMetadata.capture(
            estimator=name, config_hash="x", runtime_seconds=0.001, random_seed=0
        ),
        subgroup_findings=(),
    )


def _truthful_point(dataset) -> float:  # noqa: ANN001
    """The naive difference in means: near-unbiased on a randomised scenario."""
    frame, schema = dataset.frame, dataset.schema
    treated = frame.loc[frame[schema.treatment] == 1, schema.outcome].mean()
    control = frame.loc[frame[schema.treatment] == 0, schema.outcome].mean()
    return float(treated - control)


class BiasedEstimator:
    """Correct spread, but every estimate shifted by a known constant."""

    name = "seeded_biased"
    requirements = CROSS_SECTIONAL_REQUIREMENTS

    def fit(self, dataset, config):  # noqa: ANN001, ANN201
        return _estimate(self.name, _truthful_point(dataset) + BIAS, 0.08, dataset)


class OverconfidentEstimator:
    """Roughly unbiased, but with an interval far too narrow to cover."""

    name = "seeded_overconfident"
    requirements = CROSS_SECTIONAL_REQUIREMENTS

    def fit(self, dataset, config):  # noqa: ANN001, ANN201
        return _estimate(self.name, _truthful_point(dataset), 0.0005, dataset)


@pytest.fixture
def seeded_defects():  # noqa: ANN201
    # Registered inside the fixture rather than at import time, so the two
    # broken estimators never exist for any other test in the session. The
    # autouse `_isolate_registry` fixture moved in Step 3 restores the registry
    # afterwards. `register` is a decorator taking the class alone.
    register(BiasedEstimator)
    register(OverconfidentEstimator)


def _grid() -> GridConfig:
    # Scenario A is randomised, so the naive difference in means is an
    # honest baseline and any bias the grid reports for the seeded estimators
    # is the bias that was seeded, not confounding.
    return GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(600,), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=20,
    )


def _heterogeneous_grid() -> GridConfig:
    # Scenario A's effect is homogeneous and its baseline probability is
    # bounded well inside (0, 1) by construction (see
    # cp_simulation.components.MIN/MAX_BASELINE_PROBABILITY), so its oracle
    # ATE -- mean(p1 - p0) -- reduces to the additive shift `tau` exactly,
    # bit for bit, on every replication: verified directly (build_scenario
    # ("A", ...) at five different seeds all returned 0.05 to full float
    # precision) and already pinned elsewhere in the suite at 1e-9 tolerance
    # (tests/simulation/test_dgp.py::test_true_ate_is_measured_not_declared).
    # Scenario A therefore cannot demonstrate the property this test checks.
    # Scenario F's own registry entry documents the fix: "The population
    # average depends on the covariate mix, so it is measured rather than
    # assumed" -- its heterogeneous effect makes the oracle ATE a function of
    # which units the sample happens to draw, so it genuinely differs
    # replication to replication under random assignment.
    return GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("F",), n_units=(600,), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=20,
    )


def test_the_grid_detects_a_seeded_bias(seeded_defects):  # noqa: ANN001
    summary = summarise(run_grid(_grid(), estimators=_ASSERTED_ESTIMATORS)).set_index("method")

    row = summary.loc["seeded_biased"]
    assert row["bias"] == pytest.approx(BIAS, abs=0.05)
    # And it is many MCSE from zero, so the detection is not luck.
    assert abs(row["bias"]) > 5 * row["bias_mcse"]


def test_the_grid_does_not_cry_wolf_on_an_honest_estimator(seeded_defects):  # noqa: ANN001
    # The other half of the claim. An instrument that flags everything is as
    # useless as one that flags nothing.
    summary = summarise(run_grid(_grid(), estimators=_ASSERTED_ESTIMATORS)).set_index("method")

    row = summary.loc["difference_in_means"]
    assert abs(row["bias"]) < 3 * row["bias_mcse"]


def test_the_grid_detects_an_overconfident_interval(seeded_defects):  # noqa: ANN001
    summary = summarise(run_grid(_grid(), estimators=_ASSERTED_ESTIMATORS)).set_index("method")

    row = summary.loc["seeded_overconfident"]
    assert row["coverage_analytic"] < 0.5
    assert row["se_calibration"] < 0.5


def test_an_honest_estimator_covers_near_nominal(seeded_defects):  # noqa: ANN001
    summary = summarise(run_grid(_grid(), estimators=_ASSERTED_ESTIMATORS)).set_index("method")

    row = summary.loc["difference_in_means"]
    assert row["coverage_analytic"] > 0.75


def test_the_seeded_bias_would_be_invisible_if_truth_were_read_from_the_label(
    seeded_defects,  # noqa: ANN001
):
    # Uses scenario F, not the shared `_grid()`. Scenario A's declared_effect
    # is 0.05, but its oracle ATE is *provably* exactly 0.05 on every
    # replication -- see `_heterogeneous_grid`'s docstring -- so it cannot
    # exercise this property: a grid that read the declared label instead of
    # the oracle would look identical to one that didn't. Scenario F's
    # heterogeneous effect makes the oracle genuinely depend on the sample
    # drawn, so if the grid scored against the label rather than the oracle,
    # the truth column would be constant here -- so this pins that truth
    # varies across replications, which a declared constant could not.
    frame = run_grid(_heterogeneous_grid(), estimators=_ASSERTED_ESTIMATORS)
    truths = frame.loc[frame.status == "fitted", "truth"]

    assert truths.nunique() > 1
