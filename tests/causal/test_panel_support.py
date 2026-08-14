import numpy as np
import pandas as pd
import pytest
from cp_causal import _panel
from cp_causal._panel import (
    MIN_UNITS_PER_ARM,
    bootstrap_panel_effect,
    cluster_count_diagnostic,
    clustered_vs_iid_diagnostic,
    outcome_scale,
    panel_arrays,
)
from cp_causal.panel_checks import (
    absorbing_treatment_check,
    no_anticipation_check,
    no_concurrent_intervention_check,
    parallel_trends_check,
    sufficient_pre_periods_check,
)
from cp_core.dataset import CausalDataset, CausalSchema
from cp_core.values import EstimatorConfig

SCHEMA = CausalSchema(
    unit="unit_id",
    treatment="treated",
    outcome="outcome",
    covariates=(),
    outcome_type="continuous",
    time="period",
)


def dataset(n_units=6, n_periods=4, start=3, never=2, absorbing=True):
    rows = []
    for unit in range(n_units):
        cohort = 0 if unit < never else start
        for period in range(1, n_periods + 1):
            treated = int(cohort > 0 and period >= cohort)
            if not absorbing and unit == never and period == n_periods:
                treated = 0
            rows.append(
                {
                    "unit_id": unit,
                    "period": period,
                    "treated": treated,
                    "outcome": float(unit + period),
                }
            )
    return CausalDataset(frame=pd.DataFrame(rows), schema=SCHEMA)


def test_panel_arrays_align_long_data_with_unit_and_period_indices():
    panel = panel_arrays(dataset())

    assert panel.outcome.shape == (24,)
    assert panel.n_units == 6
    assert panel.n_periods == 4
    np.testing.assert_array_equal(panel.periods, [1, 2, 3, 4])
    # Every row's index pair must point back at its own unit and period.
    assert panel.units[panel.unit_index[5]] == 1
    assert panel.periods[panel.period_index[5]] == 2


def test_cohorts_are_reported_per_unit_with_zero_for_never_treated():
    panel = panel_arrays(dataset())

    np.testing.assert_array_equal(panel.cohort, [0, 0, 3, 3, 3, 3])
    assert panel.n_treated_units == 4
    assert panel.n_never_treated_units == 2


def test_periods_must_be_strictly_positive():
    # 0 is the never-treated sentinel, so a period of 0 would be
    # indistinguishable from "never adopted" in every cohort comparison.
    frame = dataset().frame.copy()
    frame["period"] = frame["period"] - 1

    with pytest.raises(ValueError, match="strictly positive"):
        panel_arrays(CausalDataset(frame=frame, schema=SCHEMA))


def test_panel_arrays_refuse_a_cross_section():
    # One row per unit (period == 1), not `frame.head(6)`: with this
    # fixture's row order, `.head(6)` repeats unit 0 four times and unit 1
    # twice, which `CausalDataset`'s own duplicate-observation check (keyed
    # on unit alone, since this schema carries no time column) rejects
    # before `panel_arrays` is ever reached. A true cross section — one row
    # per unit — is what should exercise `panel_arrays`'s own "not a panel"
    # guard instead.
    frame = dataset().frame
    frame = frame[frame["period"] == 1]
    schema = CausalSchema(
        unit="unit_id",
        treatment="treated",
        outcome="outcome",
        covariates=(),
        outcome_type="continuous",
    )

    with pytest.raises(ValueError, match="not a panel"):
        panel_arrays(CausalDataset(frame=frame, schema=schema))


def test_parallel_trends_is_not_testable_with_a_single_pre_period():
    # The easy bug, and exactly what NOT_TESTABLE exists for: with one
    # pre-period there are no leads to test, and a confident PASS would be a
    # claim the data cannot support.
    check = parallel_trends_check(
        lead_coefficients=np.array([]), lead_covariance=np.zeros((0, 0))
    )

    assert check.status == "NOT_TESTABLE"
    assert "no pre-treatment leads" in check.interpretation


def test_parallel_trends_passes_when_leads_are_jointly_indistinguishable():
    check = parallel_trends_check(
        lead_coefficients=np.array([0.001, -0.002]),
        lead_covariance=np.diag([0.04, 0.04]),
    )

    assert check.status == "PASS"
    # The statement the parent spec mandates.
    assert "failure to reject" in check.interpretation.lower()


def test_parallel_trends_fails_on_a_large_joint_pre_trend():
    check = parallel_trends_check(
        lead_coefficients=np.array([0.9, 1.4]),
        lead_covariance=np.diag([0.01, 0.01]),
    )

    assert check.status == "FAIL"
    assert check.statistic is not None and check.statistic > 0


def test_no_anticipation_is_separate_from_the_joint_trend_test():
    # Anticipation and differential trends are different violations with
    # different remedies, so they are never collapsed into one check.
    check = no_anticipation_check(coefficient=0.8, standard_error=0.1)

    assert check.name == "no_anticipation"
    assert check.status == "FAIL"


def test_absorbing_treatment_fails_when_a_unit_switches_off():
    assert absorbing_treatment_check(dataset()).status == "PASS"
    assert absorbing_treatment_check(dataset(absorbing=False)).status == "FAIL"


def test_sufficient_pre_periods_measures_against_the_declared_minimum():
    panel = panel_arrays(dataset(start=3))

    assert sufficient_pre_periods_check(panel, minimum=2).status == "PASS"
    assert sufficient_pre_periods_check(panel, minimum=5).status == "FAIL"


def test_no_concurrent_intervention_is_hardcoded_not_testable():
    # Pinned with a fixture that would catch a bug coupling the status to a
    # computed value, exactly as `unconfoundedness` in propensity.py is.
    first = no_concurrent_intervention_check()
    second = no_concurrent_intervention_check()

    assert first.status == "NOT_TESTABLE"
    assert first.statistic is None
    assert first.threshold is None
    assert first == second


def test_bootstrap_panel_effect_clusters_even_when_config_says_iid(monkeypatch):
    # This is the adversarial case the whole function exists to guard
    # against: a config that explicitly asks for "iid" (or, via the
    # default, asks for nothing at all) must still be forced onto the
    # cluster path. Reading `_panel.py`'s source shows `kind="cluster"` is
    # hardcoded, but a static read is not a regression guard — nothing
    # would fail if a future refactor made `kind` derive from
    # `config.bootstrap_kind`. Spying on `bootstrap_interval` as imported
    # into `_panel` makes the guarantee an executable assertion instead.
    panel = panel_arrays(dataset())
    captured: dict[str, object] = {}
    real_bootstrap_interval = _panel.bootstrap_interval

    def spy(statistic, data, **kwargs):
        captured["kind"] = kwargs.get("kind")
        captured["clusters"] = kwargs.get("clusters")
        return real_bootstrap_interval(statistic, data, **kwargs)

    monkeypatch.setattr(_panel, "bootstrap_interval", spy)

    config = EstimatorConfig(bootstrap_kind="iid", bootstrap_reps=50, random_seed=1)

    def point_fn(indices: np.ndarray) -> float:
        return float(panel.outcome[indices].mean())

    result = bootstrap_panel_effect(point_fn, panel, config)

    assert captured["kind"] == "cluster"
    assert captured["clusters"] is not None
    np.testing.assert_array_equal(captured["clusters"], panel.unit_index)
    assert np.isfinite(result.standard_error)


def test_bootstrap_panel_effect_still_clusters_under_the_default_config():
    # `EstimatorConfig.bootstrap_kind` defaults to "iid", so a caller that
    # never touches the option at all is the most common way this
    # guarantee could be silently lost.
    panel = panel_arrays(dataset())
    config = EstimatorConfig(bootstrap_reps=50, random_seed=1)

    def point_fn(indices: np.ndarray) -> float:
        return float(panel.outcome[indices].mean())

    result = bootstrap_panel_effect(point_fn, panel, config)

    assert np.isfinite(result.point_estimate)


def test_outcome_scale_is_percentage_point_for_binary_and_absolute_otherwise():
    binary_schema = CausalSchema(
        unit="unit_id",
        treatment="treated",
        outcome="outcome",
        covariates=(),
        outcome_type="binary",
        time="period",
    )
    binary_frame = dataset().frame.copy()
    binary_frame["outcome"] = (binary_frame["outcome"] % 2).astype(float)

    assert outcome_scale(dataset()) == "absolute"
    assert (
        outcome_scale(CausalDataset(frame=binary_frame, schema=binary_schema))
        == "percentage_point"
    )


def test_clustered_vs_iid_diagnostic_reports_the_widening_ratio():
    diagnostic = clustered_vs_iid_diagnostic(clustered_se=0.4, iid_se=0.2)

    assert diagnostic.name == "clustered_se_ratio"
    assert diagnostic.value == pytest.approx(2.0)
    assert "2.00x" in diagnostic.interpretation


def test_cluster_count_diagnostic_reports_the_unit_count():
    diagnostic = cluster_count_diagnostic(n_units=6)

    assert diagnostic.name == "cluster_count"
    assert diagnostic.value == 6.0
    assert "6 units" in diagnostic.interpretation


def test_min_units_per_arm_is_the_shared_threshold():
    # One constant rather than four copies is the point of this module;
    # pinning the value catches an accidental edit as much as a missing one.
    assert MIN_UNITS_PER_ARM == 10
