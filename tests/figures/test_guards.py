import dataclasses

import pytest
from cp_core.results import EffectEstimate, FitMetadata, SpecificationPoint
from cp_core.values import Interval
from cp_figures.guards import require_ci_level, require_interval, require_specification_interval


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
    # is the one thing this guard exists to prevent.
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


def _specification_point(**overrides):
    base = dict(
        specification={"trim": 0.01},
        point=0.20,
        ci=Interval(lower=0.05, upper=0.35, level=0.95),
    )
    base.update(overrides)
    return SpecificationPoint(**base)


def test_require_specification_interval_returns_the_interval_when_one_is_present():
    point = _specification_point()

    assert require_specification_interval(point, 0, "specification curve") is point.ci


def test_require_specification_interval_raises_when_the_interval_is_missing():
    # SpecificationPoint.ci carries the identical hazard as EffectEstimate.ci
    # (see the guards module docstring): it is typed Interval, not
    # Interval | None, and SpecificationCurve.__post_init__ never inspects
    # it, so dataclasses.replace(..., ci=None) constructs successfully
    # despite the static type.
    point = dataclasses.replace(_specification_point(), ci=None)

    with pytest.raises(ValueError, match="without an interval"):
        require_specification_interval(point, 0, "specification curve")


def test_require_specification_interval_names_the_figure_and_the_index():
    point = dataclasses.replace(_specification_point(), ci=None)

    with pytest.raises(ValueError, match="specification curve") as excinfo:
        require_specification_interval(point, 2, "specification curve")
    assert "2" in str(excinfo.value)


def test_require_specification_interval_degenerate_interval_is_not_missing():
    point = _specification_point(ci=Interval(lower=0.25, upper=0.25, level=0.95))

    assert require_specification_interval(point, 0, "specification curve").lower == 0.25


def test_require_ci_level_returns_the_level_when_an_interval_is_present():
    estimate = _estimate(ci=Interval(lower=0.05, upper=0.45, level=0.90))

    assert require_ci_level(estimate, "event study") == pytest.approx(0.90)


def test_require_ci_level_raises_when_the_interval_is_missing():
    # Same hazard require_interval guards against (EffectEstimate.ci is
    # typed Interval, not Interval | None, and __post_init__ never checks
    # it -- see the module docstring), reached through a caller that only
    # ever needs the level, not the bounds.
    estimate = dataclasses.replace(_estimate(method="event_study"), ci=None)

    with pytest.raises(ValueError, match="without a confidence level"):
        require_ci_level(estimate, "event study")


def test_require_ci_level_names_the_figure_and_the_method():
    estimate = dataclasses.replace(_estimate(method="event_study"), ci=None)

    with pytest.raises(ValueError, match="event_study") as excinfo:
        require_ci_level(estimate, "event study")
    assert "event study" in str(excinfo.value)
