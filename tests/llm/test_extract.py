import dataclasses

import numpy as np
import pytest
from cp_causal.registry import get_estimator
from cp_core.results import estimate_summary
from cp_llm.extract import fact_table
from cp_report.pipeline import analyse

from llm.conftest import REFERENCE


def _value(facts, fact_id):
    fact = facts.get(fact_id)
    assert fact is not None, f"no fact {fact_id}"
    return fact.value


def test_estimate_facts_are_the_stored_summary(outcome, facts):
    for report in outcome.reports:
        summary = estimate_summary(report.estimate)
        method = report.estimate.method
        assert _value(facts, f"est.{method}.method") == method
        for field in ("point", "se", "ci_lower", "ci_upper", "ci_level", "p_value"):
            if summary[field] is not None:
                assert _value(facts, f"est.{method}.{field}") == summary[field]
        for field in ("estimand", "scale", "ci_basis", "n_treated", "n_control"):
            assert _value(facts, f"est.{method}.{field}") == summary[field]
        assert _value(facts, f"score.{method}.band") == report.score.band
        assert _value(facts, f"score.{method}.ceiling") == report.score.ceiling
        for index, line in enumerate(report.score.rationale):
            assert _value(facts, f"score.{method}.rationale.{index}") == line


def test_every_fact_about_a_method_names_that_methods_name_fact(outcome, facts):
    methods = {report.estimate.method for report in outcome.reports}
    for fact in facts:
        prefix, method = fact.id.split(".")[:2]
        if prefix in ("est", "score", "check") and method in methods:
            expected = None if fact.id == f"est.{method}.method" else f"est.{method}.method"
            assert fact.subject == expected, fact.id


def test_checks_are_the_ones_the_band_was_scored_from(outcome, facts):
    for report in outcome.reports:
        method = report.estimate.method
        assert report.score.checks, "the fixture no longer has checks to read"
        for check in report.score.checks:
            base = f"check.{method}.{check.name}"
            assert _value(facts, f"{base}.name") == check.name
            assert _value(facts, f"{base}.kind") == check.kind
            assert _value(facts, f"{base}.status") == check.status
            assert _value(facts, f"{base}.interpretation") == check.interpretation


def test_a_value_that_was_never_computed_gets_no_fact(outcome, facts):
    untested = [
        (report.estimate.method, check)
        for report in outcome.reports
        for check in report.score.checks
        if check.statistic is None
    ]
    assert untested, "the fixture no longer has a check without a statistic"
    for method, check in untested:
        assert facts.get(f"check.{method}.{check.name}.statistic") is None
        assert facts.get(f"check.{method}.{check.name}.status") is not None

    computed = [
        (report.estimate.method, check)
        for report in outcome.reports
        for check in report.score.checks
        if check.statistic is not None
    ]
    assert computed, "the fixture no longer has a check with a statistic"
    for method, check in computed:
        assert _value(facts, f"check.{method}.{check.name}.statistic") == check.statistic


def test_direction_and_the_zero_test_are_computed_from_the_estimate(outcome, facts):
    seen = set()
    for report in outcome.reports:
        estimate, method = report.estimate, report.estimate.method
        direction = "positive" if estimate.point > 0 else "negative"
        assert _value(facts, f"est.{method}.direction") == direction
        excludes = estimate.ci.lower > 0 or estimate.ci.upper < 0
        label = "excludes zero" if excludes else "includes zero"
        assert _value(facts, f"est.{method}.interval_excludes_zero") == label
        seen.add(label)
    assert seen == {"excludes zero", "includes zero"}, "the fixture covers one branch only"


def test_a_compared_group_carries_its_agreement_and_range(outcome, facts):
    (group,) = outcome.comparisons
    assert group.comparison is not None
    prefix = f"cmp.{group.estimand.lower()}.{group.scale}"

    assert _value(facts, f"{prefix}.n_methods") == 2
    assert _value(facts, f"{prefix}.agreement") == group.comparison.agreement
    assert _value(facts, f"{prefix}.verdict") == group.comparison.verdict
    assert _value(facts, f"{prefix}.point_low") == group.comparison.point_range[0]
    assert _value(facts, f"{prefix}.point_high") == group.comparison.point_range[1]
    assert _value(facts, f"{prefix}.dispersion") == group.comparison.dispersion
    assert facts.get(f"{prefix}.not_compared_reason") is None


def test_a_group_of_one_carries_its_reason_and_no_agreement(cross_section):
    facts = fact_table(analyse(cross_section, REFERENCE, ["difference_in_means"]))

    assert facts.get("cmp.ate.percentage_point.agreement") is None
    assert "only one method" in _value(facts, "cmp.ate.percentage_point.not_compared_reason")
    assert _value(facts, "cmp.ate.percentage_point.n_methods") == 1


def test_every_routing_decision_is_a_fact_with_its_reasons(outcome, facts):
    excluded = [d for d in outcome.decisions if not d.eligible]
    assert excluded, "the fixture no longer excludes any estimator"
    for decision in outcome.decisions:
        prefix = f"route.{decision.estimator_name}"
        assert _value(facts, f"{prefix}.estimator") == decision.estimator_name
        expected = "eligible" if decision.eligible else "not eligible"
        assert _value(facts, f"{prefix}.eligible") == expected
        for index, reason in enumerate(decision.reasons):
            assert _value(facts, f"{prefix}.reasons.{index}") == reason


def test_a_fit_failure_is_a_fact_and_leaves_no_estimate_facts(cross_section, monkeypatch):
    def boom(self, dataset, config):
        raise RuntimeError("boom")

    monkeypatch.setattr(get_estimator("lin_regression"), "fit", boom)

    facts = fact_table(
        analyse(cross_section, REFERENCE, ["difference_in_means", "lin_regression"])
    )

    assert _value(facts, "fail.lin_regression.method") == "lin_regression"
    assert _value(facts, "fail.lin_regression.error") == "boom"
    assert _value(facts, "analysis.n_failed") == 1
    assert _value(facts, "analysis.n_estimates") == 1
    assert facts.get("est.lin_regression.point") is None


def test_a_panel_analysis_is_keyed_by_its_own_estimand():
    from cp_simulation.panel_scenarios import build_panel_scenario

    panel = build_panel_scenario("P1").generate(
        n_units=100, n_periods=6, rng=np.random.default_rng(0)
    ).dataset

    facts = fact_table(analyse(panel, REFERENCE, ["did_2x2"]))

    assert _value(facts, "est.did_2x2.estimand") == "ATT"
    assert facts.get("cmp.att.absolute.not_compared_reason") is not None


def test_provenance_and_counts_describe_the_analysis(outcome, facts):
    assert (
        _value(facts, "analysis.assignment_provenance")
        == outcome.manifest.schema.assignment_provenance
    )
    assert _value(facts, "analysis.n_estimates") == 2
    assert _value(facts, "analysis.n_failed") == 0


def test_a_name_that_is_not_an_identifier_is_refused_not_escaped(outcome):
    report = outcome.reports[0]
    renamed = dataclasses.replace(
        report, estimate=dataclasses.replace(report.estimate, method="Diff in Means")
    )

    with pytest.raises(ValueError, match="'Diff in Means' cannot be a fact id segment"):
        fact_table(dataclasses.replace(outcome, reports=(renamed,)))


def _with_point(outcome, point):
    report = outcome.reports[0]
    changed = dataclasses.replace(
        report, estimate=dataclasses.replace(report.estimate, point=point)
    )
    return report.estimate.method, fact_table(
        dataclasses.replace(outcome, reports=(changed,))
    )


def test_a_non_finite_point_gets_neither_a_point_nor_a_direction_fact(outcome):
    method, facts = _with_point(outcome, float("nan"))

    assert facts.get(f"est.{method}.point") is None
    assert facts.get(f"est.{method}.direction") is None


def test_a_point_of_exactly_zero_has_the_direction_zero(outcome):
    method, facts = _with_point(outcome, 0.0)

    assert _value(facts, f"est.{method}.direction") == "zero"
