import pytest
from cp_llm import verify as verify_module
from cp_llm.verify import Violation, verify

from llm.builders import drop, good_narrative, swap

_BAND = good_narrative().validity[0]

# One planted defect per rule. Each is the verified narrative with a single
# change, so the rule named is the only thing standing between it and a pass.
PLANTED = {
    "resolves": swap(good_narrative(), "validity", 0, _BAND + " See {est.ipw.power}."),
    "bare_value": swap(good_narrative(), "validity", 0, _BAND + " That is 2 steps down."),
    "vocabulary": swap(good_narrative(), "validity", 0, _BAND + " The evidence is strong."),
    "attribution": swap(
        good_narrative(),
        "validity",
        0,
        "The validity band for {est.ipw.method} is {score.difference_in_means.band}, "
        "under a ceiling of {score.ipw.ceiling}.",
    ),
    "bare_point": swap(
        good_narrative(),
        "estimates",
        0,
        "Under its own assumptions, {est.difference_in_means.method} estimates "
        "{est.difference_in_means.point}.",
    ),
    "language": swap(
        good_narrative(), "validity", 0, _BAND + " The treatment causes the outcome."
    ),
    "completeness": drop(good_narrative(), "caveats", 0),
}


def test_the_fixture_has_a_check_the_completeness_rule_must_demand(facts):
    status = facts.get("check.ipw.positivity.status")
    assert status is not None and status.value == "WARN"


def test_a_narrative_that_cites_everything_and_writes_no_value_verifies(good, facts):
    assert verify(good, facts) == ()


@pytest.mark.parametrize("rule", sorted(PLANTED))
def test_each_planted_defect_is_caught_by_its_own_rule(rule, facts):
    violations = verify(PLANTED[rule], facts)

    assert violations, f"the {rule} defect verified"
    assert {v.rule for v in violations} == {rule}


@pytest.mark.parametrize("rule", sorted(PLANTED))
def test_with_its_rule_removed_each_planted_defect_verifies(rule, facts, monkeypatch):
    # The removal proof: no rule is covered only by another. If this fails,
    # some other rule also fires on the defect and `rule` is not load-bearing.
    registry = (
        verify_module.NARRATIVE_RULES
        if rule in verify_module.NARRATIVE_RULES
        else verify_module.STATEMENT_RULES
    )
    monkeypatch.delitem(registry, rule)

    assert verify(PLANTED[rule], facts) == ()


def test_the_planted_defects_cover_every_registered_rule():
    registered = set(verify_module.STATEMENT_RULES) | set(verify_module.NARRATIVE_RULES)

    assert set(PLANTED) == registered


def test_a_violation_names_its_section_and_statement(facts):
    (violation,) = verify(PLANTED["vocabulary"], facts)

    assert violation == Violation(
        rule="vocabulary",
        section="validity",
        statement=PLANTED["vocabulary"].validity[0],
        reason="'strong' states a validity band; cite the band or ceiling fact",
    )


def test_a_narrative_wide_violation_names_no_statement(facts):
    (violation,) = verify(PLANTED["completeness"], facts)

    assert (violation.section, violation.statement) == ("", "")
    assert violation.reason == "the narrative never cites {check.ipw.positivity.status}"


@pytest.mark.parametrize(
    "tail",
    [
        " {",
        " }",
        " {est.ipw.point:.2f}",
        " {}",
    ],
)
def test_a_malformed_placeholder_does_not_resolve(tail, facts):
    narrative = swap(good_narrative(), "validity", 1, good_narrative().validity[1] + tail)

    assert "resolves" in {v.rule for v in verify(narrative, facts)}


@pytest.mark.parametrize(
    "tail",
    [" Two checks ran.", " It fell by half.", " That is 10 percent."],
)
def test_a_number_word_is_a_bare_value(tail, facts):
    narrative = swap(good_narrative(), "validity", 1, good_narrative().validity[1] + tail)

    assert "bare_value" in {v.rule for v in verify(narrative, facts)}


@pytest.mark.parametrize(
    "tail",
    [
        " The result is statistically significant.",
        " The overlap check failed.",
        " The estimate is positive.",
        " The methods agree.",
        " The two readings are consistent.",
    ],
)
def test_a_finding_written_as_free_text_is_refused(tail, facts):
    narrative = swap(good_narrative(), "validity", 1, good_narrative().validity[1] + tail)

    assert "vocabulary" in {v.rule for v in verify(narrative, facts)}


def test_the_lints_own_hedge_consistent_with_is_allowed(facts):
    narrative = swap(
        good_narrative(),
        "validity",
        1,
        good_narrative().validity[1] + " This is consistent with the assumptions holding.",
    )

    assert verify(narrative, facts) == ()


def test_a_point_needs_both_bounds_not_one(facts):
    narrative = swap(
        good_narrative(),
        "estimates",
        0,
        "{est.difference_in_means.method} estimates {est.difference_in_means.point}, "
        "above {est.difference_in_means.ci_lower}.",
    )

    assert {v.rule for v in verify(narrative, facts)} == {"bare_point"}


def test_naming_both_methods_satisfies_attribution_for_a_comparison(facts):
    narrative = swap(
        good_narrative(),
        "validity",
        0,
        "The validity band for {est.difference_in_means.method} is "
        "{score.difference_in_means.band} and for {est.ipw.method} it is {score.ipw.band}.",
    )

    assert "attribution" not in {v.rule for v in verify(narrative, facts)}


def test_the_language_rule_reads_the_rendered_text_not_the_template(facts):
    # The offending word arrives through a fact, so only the rendered
    # statement contains it.
    from cp_llm.facts import Fact, FactTable

    poisoned = FactTable(
        (*facts.facts, Fact("analysis.note", "text", "The treatment causes the outcome.", None))
    )
    narrative = swap(
        good_narrative(), "validity", 1, good_narrative().validity[1] + " {analysis.note}"
    )

    assert {v.rule for v in verify(narrative, poisoned)} == {"language"}


def test_a_dropped_estimate_fails_completeness_for_its_point_and_band(facts):
    narrative = drop(drop(good_narrative(), "estimates", 1), "validity", 1)

    reasons = {v.reason for v in verify(narrative, facts) if v.rule == "completeness"}

    assert reasons == {
        "the narrative never cites {est.ipw.point}",
        "the narrative never cites {score.ipw.band}",
    }


def test_a_dropped_comparison_fails_completeness(facts):
    narrative = drop(good_narrative(), "summary", 0)

    assert [v.reason for v in verify(narrative, facts)] == [
        "the narrative never cites {cmp.ate.percentage_point.agreement}"
    ]
