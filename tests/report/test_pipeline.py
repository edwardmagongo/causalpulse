import re

from cp_core.manifest import FixtureReference
from cp_report.pipeline import analyse, comparison_groups

from report.builders import _estimate, _score

REFERENCE = FixtureReference(path="d.csv", content_sha256="0" * 64)
PAIR = ["difference_in_means", "lin_regression"]


def test_each_report_is_built_from_its_own_estimates_checks(cross_section):
    # Real fits, not a fixture made to agree with itself: the two methods'
    # assumption tables differ, so a report that took one method's table and
    # another's band would show a check name it has no business showing.
    outcome = analyse(cross_section, REFERENCE, PAIR)
    by_method = {r.estimate.method: r for r in outcome.reports}
    dim, lin = by_method["difference_in_means"], by_method["lin_regression"]

    for report in outcome.reports:
        assert report.score.checks is report.estimate.assumptions
        assert report.score.band in report.html
        for check in report.estimate.assumptions:
            assert f"<td>{check.name}</td>" in report.html

    only_lin = {c.name for c in lin.estimate.assumptions} - {
        c.name for c in dim.estimate.assumptions
    }
    assert only_lin, "the fixture no longer separates the two methods' checks"
    for name in only_lin:
        assert f"<td>{name}</td>" not in dim.html


def test_every_markdown_figure_link_names_a_stored_asset(cross_section):
    outcome = analyse(cross_section, REFERENCE, PAIR)

    for report in outcome.reports:
        links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report.markdown)
        assert links, "a cross-sectional report carries at least its forest plot"
        assert set(links) == set(report.assets)
        for svg in report.assets.values():
            assert svg.lstrip().startswith("<svg")


def test_the_outcome_carries_routing_manifest_and_one_compared_group(cross_section):
    outcome = analyse(cross_section, REFERENCE, PAIR)

    assert not outcome.partial
    assert {d.estimator_name for d in outcome.decisions} >= set(PAIR)
    assert [name for name, _ in outcome.manifest.estimators] == PAIR
    (group,) = outcome.comparisons
    assert group.methods == tuple(PAIR)
    assert group.comparison is not None
    assert group.to_dict()["agreement"] in {"consistent", "divergent", "not_assessable"}


def test_an_ate_and_an_att_are_compared_within_their_groups_never_across():
    pairs = [
        (_estimate(method="a", estimand="ATE"), _score()),
        (_estimate(method="b", estimand="ATE", point=0.3), _score()),
        (_estimate(method="c", estimand="ATT"), _score()),
    ]

    groups = {g.estimand: g for g in comparison_groups(pairs)}

    assert groups["ATE"].methods == ("a", "b")
    assert groups["ATE"].comparison is not None
    assert groups["ATT"].comparison is None
    assert "only one method" in groups["ATT"].not_compared_reason
    assert groups["ATT"].to_dict()["not_compared_reason"] == groups["ATT"].not_compared_reason
