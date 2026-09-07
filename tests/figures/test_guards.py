import dataclasses

import pytest
from cp_core.results import EffectEstimate, FitMetadata
from cp_core.values import Interval
from cp_figures.guards import require_interval


def _estimate(**overrides):
    base = dict(
        estimand="ATE",
        method="difference_in_means",
        point=0.25,
        se=0.1,
        ci=Interval(lower=0.05, upper=0.45, level=0.95),
        ci_basis="analytic",
        bootstrap_ci=None,
        p_value=0.01,
        n_treated=100,
        n_control=100,
        scale="percentage_point",
        diagnostics={},
        assumptions=(),
        counterfactual=None,
        fit_metadata=FitMetadata.capture(
            estimator="difference_in_means", config_hash="h",
            runtime_seconds=0.1, random_seed=0,
        ),
        subgroup_findings=(),
    )
    base.update(overrides)
    return EffectEstimate(**base)


def test_it_returns_the_interval_when_one_is_present():
    estimate = _estimate()

    assert require_interval(estimate, "forest plot") is estimate.ci


def test_it_raises_when_the_interval_is_missing():
    # EffectEstimate.ci is typed Interval, not Interval | None, and
    # __post_init__ does not check it -- dataclasses do not enforce field
    # types at runtime, so dataclasses.replace(..., ci=None) constructs
    # successfully despite the static type. That is exactly the hand-built /
    # mutated object the guard exists to catch: a figure drawing a bare point
    # would show a number with no indication of how well it is known, which
    # is the one thing the parent spec forbids.
    estimate = dataclasses.replace(_estimate(), ci=None)

    with pytest.raises(ValueError, match="without an interval"):
        require_interval(estimate, "forest plot")


def test_the_error_names_the_figure_and_the_method():
    estimate = dataclasses.replace(_estimate(method="ipw"), ci=None)

    with pytest.raises(ValueError, match="ipw") as excinfo:
        require_interval(estimate, "love plot")
    assert "love plot" in str(excinfo.value)


def test_a_degenerate_interval_is_not_treated_as_missing():
    # A zero-width interval is a real, if unusual, statement about
    # uncertainty -- an estimator reporting one has said something, and the
    # figure should draw it rather than refuse.
    estimate = _estimate(ci=Interval(lower=0.25, upper=0.25, level=0.95))

    assert require_interval(estimate, "forest plot").lower == 0.25
