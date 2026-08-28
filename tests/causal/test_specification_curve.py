"""The specification curve: the same question, refit under every defensible
variation of the analyst's discretionary choices.

Every point is a genuine refit. The tests below check the axes are what they
claim, that the analysis as actually run is findable inside its own curve,
and that the estimators whose curve would be meaningless refuse rather than
return one.
"""

import numpy as np
import pandas as pd
import pytest
from cp_causal.propensity import POSITIVITY_THRESHOLD
from cp_causal.registry import all_estimators
from cp_causal.sensitivity import CURVE_ESTIMATORS, TRIM_AXIS, specification_curve
from cp_core.dataset import CausalDataset, CausalSchema
from cp_core.values import EstimatorConfig
from cp_simulation.scenarios import build_scenario

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)


def cross_section(n_units=600):
    return build_scenario("A").generate(
        n_units=n_units, rng=np.random.default_rng(0)
    ).dataset


def wide_frame(n_covariates=6, n_rows=300):
    """A synthetic frame with more covariates than any scenario declares.

    The cap cannot bind on a real scenario -- they declare two covariates, so
    the largest real curve is 24 points -- so exercising it needs a frame
    built for the purpose.
    """
    rng = np.random.default_rng(0)
    names = tuple(f"x{i}" for i in range(n_covariates))
    frame = pd.DataFrame({name: rng.normal(size=n_rows) for name in names})
    frame["unit"] = np.arange(n_rows)
    frame["w"] = rng.integers(0, 2, size=n_rows)
    frame["y"] = frame["w"] * 0.5 + frame["x0"] + rng.normal(size=n_rows)
    schema = CausalSchema(
        unit="unit",
        treatment="w",
        outcome="y",
        covariates=names,
        outcome_type="continuous",
        assignment_provenance="randomised",
    )
    return CausalDataset(frame=frame, schema=schema)


def test_the_supported_set_is_exactly_pinned():
    supported = frozenset({"lin_regression", "ipw", "aipw"})
    assert supported == CURVE_ESTIMATORS


def test_lin_regression_gets_the_covariate_by_outlier_cross_product():
    # Two covariates -> 4 subsets, of which the empty one is rejected by the
    # estimator's own needs_covariates requirement, leaving 3; times 2 outlier
    # rules, and no trim axis. 6, not 8 -- the empty subset never survives.
    # Verified by enumerating Requirements.unmet over all 8 candidates: the
    # only two rejected are the empty subset under each outlier rule, and the
    # sole reason given is "requires covariates; dataset declares none".
    curve = specification_curve(cross_section(), "lin_regression", CONFIG)
    assert len(curve.specifications) == 6
    assert curve.capped is False


def test_a_propensity_estimator_adds_the_trim_axis_and_drops_the_empty_subset():
    # 3 non-empty subsets x 2 outlier rules x 3 trim values. The empty subset
    # is dropped by the estimator's own Requirements, not by a special case:
    # enumerated, ipw rejects exactly 6 of 24 candidates and every rejection
    # is the empty subset, never its min_treated_units=10 floor.
    curve = specification_curve(cross_section(), "ipw", CONFIG)
    assert len(curve.specifications) == 3 * 2 * len(TRIM_AXIS)
    assert all(
        point.specification["covariates"] != () for point in curve.specifications
    )


def test_the_trim_axis_starts_at_the_default_and_runs_tighter():
    # The analysis as actually run must be on its own axis, so the default
    # threshold is the loosest value rather than the midpoint. An axis
    # straddling it -- (0.005, 0.01, 0.02) -- was measured and rejected: see
    # the next test.
    assert min(TRIM_AXIS) == POSITIVITY_THRESHOLD
    assert list(TRIM_AXIS) == sorted(set(TRIM_AXIS))


def test_the_trim_axis_actually_moves_a_confounded_estimate():
    # The guard against narrowing this axis back to values that clip nothing.
    # The obvious axis (0.005, 0.01, 0.02) clips 0 rows at all three values on
    # scenarios A and D, so ipw returned 18 specification points that were
    # really 6 distinct analyses repeated three times -- exactly the "N copies
    # of one number reads as robustness" failure that excludes
    # difference_in_means from this analysis entirely.
    #
    # Scenario D's propensity scores span 0.027 to 0.972; measured, the
    # shipped axis clips 0, 3 and 18 of its 600 rows.
    dataset = build_scenario("D").generate(
        n_units=600, rng=np.random.default_rng(0)
    ).dataset
    curve = specification_curve(dataset, "ipw", CONFIG)
    by_group: dict[tuple, list[float]] = {}
    for point in curve.specifications:
        key = (
            point.specification["covariates"],
            point.specification["outliers_excluded"],
        )
        by_group.setdefault(key, []).append(point.point)
    moved = [group for group in by_group.values() if len(set(group)) > 1]
    assert moved, "the trim axis produced no variation on a confounded scenario"


def test_the_trim_axis_is_correctly_inert_under_randomisation():
    # The complement of the test above, and the reason it is scoped to a
    # confounded scenario. Scenario A is randomised: its propensity scores
    # span 0.40 to 0.54, so no threshold on this axis clips anything and every
    # trim value must return the same point. That is the honest answer, not a
    # defect -- and pinning it stops someone "fixing" the inertness by pushing
    # the axis somewhere it would clip well-overlapped rows.
    curve = specification_curve(cross_section(), "ipw", CONFIG)
    by_group: dict[tuple, set[float]] = {}
    for point in curve.specifications:
        key = (
            point.specification["covariates"],
            point.specification["outliers_excluded"],
        )
        by_group.setdefault(key, set()).add(point.point)
    assert all(len(group) == 1 for group in by_group.values())


def test_the_original_index_points_at_the_analysis_as_actually_run():
    dataset = cross_section()
    curve = specification_curve(dataset, "lin_regression", CONFIG)
    original = curve.specifications[curve.original_index]
    assert original.specification["covariates"] == dataset.schema.covariates
    assert original.specification["outliers_excluded"] is False


def test_the_original_point_matches_a_direct_fit_of_the_estimator():
    # The curve's own anchor must be the real analysis, not a near-copy of it.
    dataset = cross_section()
    curve = specification_curve(dataset, "lin_regression", CONFIG)
    direct = all_estimators()["lin_regression"]().fit(dataset, CONFIG)
    assert curve.specifications[curve.original_index].point == pytest.approx(direct.point)


def test_the_points_are_not_all_identical():
    # A curve of identical numbers is the failure mode that makes an estimator
    # ineligible for this analysis at all; it must not appear on an eligible one.
    curve = specification_curve(cross_section(), "lin_regression", CONFIG)
    points = [point.point for point in curve.specifications]
    assert len(set(points)) > 1


def test_the_outlier_axis_actually_removes_rows():
    curve = specification_curve(cross_section(), "lin_regression", CONFIG)
    excluded = [
        p for p in curve.specifications if p.specification["outliers_excluded"]
    ]
    kept = [
        p for p in curve.specifications if not p.specification["outliers_excluded"]
    ]
    assert len(excluded) == len(kept) == 3
    # Same covariate sets on both sides, different numbers -- so the axis moved
    # the data rather than relabelling the same fits. Measured: the leverage
    # rule flags 13 of scenario A's 600 rows at n_units=600, seed 0.
    by_covariates = {p.specification["covariates"]: p.point for p in kept}
    for point in excluded:
        assert point.point != by_covariates[point.specification["covariates"]]


def test_the_cap_binds_and_is_recorded():
    curve = specification_curve(
        wide_frame(), "lin_regression", CONFIG, max_specifications=8
    )
    assert len(curve.specifications) == 8
    assert curve.capped is True


def test_the_base_specification_survives_the_cap():
    # A curve that truncated away the analysis as actually run would be
    # unreadable, and SpecificationCurve would refuse to construct. The base
    # is the full covariate set, which enumeration order puts last, so this
    # is the case that exercises the swap rather than a lucky ordering.
    dataset = wide_frame()
    curve = specification_curve(
        dataset, "lin_regression", CONFIG, max_specifications=3
    )
    original = curve.specifications[curve.original_index]
    assert original.specification["covariates"] == dataset.schema.covariates
    assert original.specification["outliers_excluded"] is False


@pytest.mark.parametrize(
    "estimator_name",
    ["difference_in_means", "causal_forest", "twfe", "synthetic_control"],
)
def test_an_unsupported_estimator_raises_rather_than_returning_an_empty_curve(
    estimator_name,
):
    with pytest.raises(ValueError, match=estimator_name):
        specification_curve(cross_section(), estimator_name, CONFIG)


def test_an_unknown_estimator_name_raises():
    with pytest.raises((KeyError, ValueError)):
        specification_curve(cross_section(), "lin_regresion", CONFIG)


def test_a_max_specifications_below_one_is_refused():
    with pytest.raises(ValueError, match="max_specifications"):
        specification_curve(
            cross_section(), "lin_regression", CONFIG, max_specifications=0
        )
