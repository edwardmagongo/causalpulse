import dataclasses

import pytest
from cp_llm import verify as verify_module
from cp_llm.facts import Fact, FactTable
from cp_llm.verify import Violation, verify

from llm.builders import drop, good_narrative, swap

_BAND = good_narrative().validity[0]
_IPW_BAND = good_narrative().validity[1]

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
    "delimiters": swap(
        good_narrative(), "validity", 1, _IPW_BAND + " It has {est.ipw.n_treated}k units."
    ),
    "one_line": swap(
        good_narrative(),
        "validity",
        1,
        _IPW_BAND + "\n\n## Limits\n\nThis band is a probability that the effect is real",
    ),
    # The verified narrative, unchanged: what changes is the facts it is read
    # against (see `PLANTED_FACTS`), in which ipw's band is INSUFFICIENT and its
    # estimate statement quotes the point and interval without citing it.
    "insufficient_caveat": good_narrative(),
}


def with_band(facts: FactTable, method: str, band: str) -> FactTable:
    """`facts` with `method`'s validity band set to `band`."""
    band_id = f"score.{method}.band"
    return FactTable(
        tuple(
            dataclasses.replace(fact, value=band) if fact.id == band_id else fact
            for fact in facts.facts
        )
    )


# The fixture's bands are MODERATE and WEAK, so the one rule that only an
# INSUFFICIENT band can set off is planted in the facts instead.
PLANTED_FACTS = {"insufficient_caveat": lambda facts: with_band(facts, "ipw", "INSUFFICIENT")}


def planted(rule: str, facts: FactTable) -> FactTable:
    return PLANTED_FACTS.get(rule, lambda same: same)(facts)


def test_the_fixture_has_a_check_the_completeness_rule_must_demand(facts):
    status = facts.get("check.ipw.positivity.status")
    assert status is not None and status.value == "WARN"


def test_a_narrative_that_cites_everything_and_writes_no_value_verifies(good, facts):
    assert verify(good, facts) == ()


@pytest.mark.parametrize("rule", sorted(PLANTED))
def test_each_planted_defect_is_caught_by_its_own_rule(rule, facts):
    facts = planted(rule, facts)
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

    assert verify(PLANTED[rule], planted(rule, facts)) == ()


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
    [
        " Two checks ran.",
        " It fell by half.",
        " That is a few percent.",
        " It is twice as large.",
        " It would double.",
    ],
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
        " The estimates are inconsistent with each other.",
        " Its interval excludes zero.",
        " The effect is zero.",
        " The method is not assessable.",
        " The assessable methods differ.",
        " The effect is increasing.",
        " The effect is decreasing.",
        " The effect is strongly supported.",
        " The effect is weakly supported.",
        " The effect is moderately supported.",
        " The method is eligible.",
        " The method is ineligible.",
        # Calibration, 2026-10-04: a retry wrote every check's status this way,
        # after "FAIL or WARN" in free text had been refused.
        " Every check reflects untested assumptions rather than a flagged problem.",
    ],
)
def test_a_finding_written_as_free_text_is_refused(tail, facts):
    narrative = swap(good_narrative(), "validity", 1, good_narrative().validity[1] + tail)

    assert "vocabulary" in {v.rule for v in verify(narrative, facts)}


def test_a_percent_sign_after_a_cited_number_is_refused(facts):
    narrative = swap(
        good_narrative(),
        "validity",
        1,
        good_narrative().validity[1] + " It moved {est.ipw.point}% overall.",
    )

    violations = [v for v in verify(narrative, facts) if v.rule == "bare_value"]

    assert [v.reason for v in violations] == [
        "a percent sign outside a placeholder; a level fact carries its own"
    ]


def test_percentage_points_after_a_cited_number_are_refused(facts):
    narrative = swap(
        good_narrative(),
        "validity",
        1,
        good_narrative().validity[1] + " It moved {est.ipw.point} percentage points.",
    )

    assert "bare_value" in {v.rule for v in verify(narrative, facts)}


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


def test_a_causal_claim_arriving_through_a_text_fact_is_linted(facts):
    # The offending words arrive through a text fact, which the lint reads
    # verbatim (only numeric facts are masked), so the template alone hides it.
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


def _with_facts(facts, *extra):
    return FactTable((*facts.facts, *extra))


def test_a_failed_check_status_must_be_cited(facts):
    table = _with_facts(
        facts, Fact("check.ipw.overlap.status", "label", "FAIL", "est.ipw.method")
    )

    reasons = [v.reason for v in verify(good_narrative(), table) if v.rule == "completeness"]

    assert reasons == ["the narrative never cites {check.ipw.overlap.status}"]


def test_a_passing_or_untestable_check_status_is_not_demanded(facts):
    statuses = {f.value for f in facts if f.id.startswith("check.") and f.id.endswith(".status")}

    # The real table has both, and the good narrative cites neither.
    assert {"PASS", "NOT_TESTABLE"} <= statuses
    assert verify(good_narrative(), facts) == ()


def test_a_fit_failure_error_must_be_cited(facts):
    table = _with_facts(
        facts,
        Fact("fail.aipw.method", "name", "aipw", None),
        Fact("fail.aipw.error", "text", "did not converge", "fail.aipw.method"),
    )

    reasons = [v.reason for v in verify(good_narrative(), table) if v.rule == "completeness"]

    assert reasons == ["the narrative never cites {fail.aipw.error}"]


def test_a_reason_a_comparison_was_not_made_must_be_cited(facts):
    table = _with_facts(
        facts,
        Fact("cmp.att.percentage_point.not_compared_reason", "text", "one method fitted", None),
    )

    reasons = [v.reason for v in verify(good_narrative(), table) if v.rule == "completeness"]

    assert reasons == ["the narrative never cites {cmp.att.percentage_point.not_compared_reason}"]


def test_a_point_needs_both_bounds_not_the_upper_one_alone(facts):
    narrative = swap(
        good_narrative(),
        "estimates",
        0,
        "{est.difference_in_means.method} estimates {est.difference_in_means.point}, "
        "below {est.difference_in_means.ci_upper}.",
    )

    assert {v.rule for v in verify(narrative, facts)} == {"bare_point"}


_HEDGED = (
    "If the assumptions of {est.ipw.method} hold, with {est.ipw.point} between "
    "{est.ipw.ci_lower} and {est.ipw.ci_upper}, the treatment causes the outcome."
)


def test_a_hedge_before_a_cited_decimal_still_covers_a_later_causal_verb(facts):
    narrative = swap(good_narrative(), "estimates", 1, _HEDGED)

    assert verify(narrative, facts) == ()


def test_the_same_sentence_without_its_hedge_is_a_causal_claim(facts):
    narrative = swap(
        good_narrative(),
        "estimates",
        1,
        "With {est.ipw.point} between {est.ipw.ci_lower} and {est.ipw.ci_upper} "
        "for {est.ipw.method}, the treatment causes the outcome.",
    )

    assert {v.rule for v in verify(narrative, facts)} == {"language"}


def test_a_hedge_with_a_p_value_fact_masked_still_covers_a_later_causal_verb(facts):
    # Dropping `p_value` from the masked kinds would put "0.0" in the lint's
    # text and end the sentence before the verb.
    assert facts.get("est.ipw.p_value") is not None
    narrative = swap(
        good_narrative(),
        "estimates",
        1,
        good_narrative().estimates[1]
        + " If the assumptions of {est.ipw.method} hold, with a p-value of "
        "{est.ipw.p_value}, the treatment causes the outcome.",
    )

    assert verify(narrative, facts) == ()


@pytest.mark.parametrize(
    "statement",
    [
        "For {est.ipw.method}, {est.ipw.n_treated}o the treatment causes the outcome.",
        "For {est.ipw.method}, this will {est.ipw.n_treated}ot prove the effect.",
    ],
)
def test_a_placeholder_glued_to_letters_cannot_spell_a_hedge_or_a_negation(statement, facts):
    # Masked as a letter, `{n}o` would lint as the word "No" and `{n}ot` as
    # "Not", hedging a claim whose rendered text reads "218o" and "218ot".
    narrative = swap(
        good_narrative(), "estimates", 1, good_narrative().estimates[1] + " " + statement
    )

    # Both statements now also break `delimiters` (a letter glued after a
    # placeholder). The `language` assertion stays: the mask being a digit is
    # defence in depth for any glue the delimiter rule might one day allow.
    assert "language" in {v.rule for v in verify(narrative, facts)}


_GLUED = [
    "In {est.ipw.method}, the point is -{est.ipw.point} overall.",
    "In {est.ipw.method}, the point is \u2212{est.ipw.point} overall.",
    "In {est.ipw.method}, the groups hold {est.ipw.n_treated}{est.ipw.n_control} units.",
    "In {est.ipw.method}, the groups hold {est.ipw.n_treated}.{est.ipw.n_control} units.",
    "In {est.ipw.method}, the p-value is {est.ipw.p_value}e-{analysis.n_estimates}.",
    "In {est.ipw.method}, the treated group holds {est.ipw.n_treated}k units.",
    "The comparison is in{cmp.ate.percentage_point.agreement} for {est.ipw.method}.",
    "The route for {est.ipw.method} is in{route.ipw.eligible}.",
    # An apostrophe is a closing quote as well as a possessive; neither lets a
    # word follow it.
    "The estimate of {est.ipw.method}'stuff is shown here.",
    "The estimate of {est.ipw.method}\u2019stuff is shown here.",
    "The method is '{est.ipw.method}'x here.",
    "The method is ({est.ipw.method})x here.",
]


@pytest.mark.parametrize("statement", _GLUED)
def test_a_placeholder_glued_to_its_neighbours_is_refused(statement, facts):
    narrative = swap(
        good_narrative(), "validity", 1, good_narrative().validity[1] + " " + statement
    )

    assert "delimiters" in {v.rule for v in verify(narrative, facts)}


def _delimiter_reasons(statement, facts):
    narrative = swap(
        good_narrative(), "validity", 1, good_narrative().validity[1] + " " + statement
    )
    return [v.reason for v in verify(narrative, facts) if v.rule == "delimiters"]


@pytest.mark.parametrize(
    "statement",
    [
        "The point for {est.ipw.method} is ({est.ipw.point}) here.",
        'The method is "{est.ipw.method}" here.',
        "The method is [{est.ipw.method}] here.",
        "The method is \u201c{est.ipw.method}\u201d here.",
        "The method is \u2018{est.ipw.method}\u2019 here.",
        "The method is '{est.ipw.method}' here.",
        "The estimate of {est.ipw.method}'s is shown here.",
        "The estimate of {est.ipw.method}\u2019s is shown here.",
        "The estimate of {est.ipw.method}'s",
        "The estimate of {est.ipw.method}'s.",
        "The method is '{est.ipw.method}'.",
        "The point for {est.ipw.method} is ({est.ipw.point}).",
        # Read in one pass: a long run of punctuation is not a recursion limit.
        "The point for {est.ipw.method} is {est.ipw.point}" + "." * 5000,
        "For {est.ipw.method}, {est.ipw.point}, with a wide interval.",
        "The point for {est.ipw.method} is {est.ipw.point}.",
        "The point for {est.ipw.method} is {est.ipw.point};",
        "Is the point for {est.ipw.method} {est.ipw.point}?",
        "The point for {est.ipw.method} is ({est.ipw.point}.) here.",
        "The point for {est.ipw.method} lies from {est.ipw.ci_lower} to {est.ipw.ci_upper}",
        "The point for {est.ipw.method} lies between {est.ipw.ci_lower} and {est.ipw.ci_upper}.",
    ],
)
def test_ordinary_punctuation_around_a_placeholder_is_allowed(statement, facts):
    assert _delimiter_reasons(statement, facts) == []


def test_the_whole_good_narrative_has_no_delimiter_violation(good, facts):
    assert [v for v in verify(good, facts) if v.rule == "delimiters"] == []


@pytest.mark.parametrize("sign", ["-", "\u2212", "\u2013", "+", "\u00b1", "\u00d7", "*", "/", "^",
                                  "~", "\u2248", "<", ">", "="])
def test_a_sign_or_operator_before_a_number_is_refused_even_after_a_space(sign, facts):
    statement = f"The point for {{est.ipw.method}} is {sign} {{est.ipw.point}} here."

    reasons = _delimiter_reasons(statement, facts)

    assert (
        "a sign or operator before {est.ipw.point}; the fact carries its own sign, "
        "and a range is written from one bound to the other"
    ) in reasons


def test_an_operator_before_a_word_valued_fact_is_not_this_rules_concern(facts):
    statement = "The method is = {est.ipw.method} here."

    assert [r for r in _delimiter_reasons(statement, facts) if r.startswith("a sign")] == []


def test_a_placeholder_naming_no_fact_gets_no_sign_complaint(facts):
    statement = "The method is - {est.ipw.nothing} here."

    assert [r for r in _delimiter_reasons(statement, facts) if r.startswith("a sign")] == []


def test_a_glued_placeholder_is_reported_with_its_reason(facts):
    reasons = _delimiter_reasons("The count is {est.ipw.n_treated}k for {est.ipw.method}.", facts)

    assert reasons == [
        "{est.ipw.n_treated} is glued to the text after it; put a space or sentence "
        "punctuation after a placeholder"
    ]
    reasons = _delimiter_reasons("The count is x{est.ipw.n_treated} for {est.ipw.method}.", facts)
    assert reasons == [
        "{est.ipw.n_treated} is glued to the text before it; put a space or an opening "
        "bracket before a placeholder"
    ]


@pytest.mark.parametrize(
    "number_word",
    [" It is minus {est.ipw.n_treated} units.", " It is {est.ipw.n_treated} times larger."],
)
def test_minus_and_times_are_number_words(number_word, facts):
    narrative = swap(
        good_narrative(), "validity", 1, good_narrative().validity[1] + number_word
    )

    assert "bare_value" in {v.rule for v in verify(narrative, facts)}


@pytest.mark.parametrize(
    ("character", "code"),
    [
        ("\n", "U+000A"),
        ("\r", "U+000D"),
        ("\t", "U+0009"),
        ("\u200b", "U+200B"),
        ("\u2028", "U+2028"),
        ("\u2029", "U+2029"),
        ("\u200d", "U+200D"),
    ],
)
def test_a_control_or_invisible_character_is_refused(character, code, facts):
    narrative = swap(
        good_narrative(),
        "validity",
        1,
        good_narrative().validity[1] + f" It is sig{character}nificant.",
    )

    violations = [v for v in verify(narrative, facts) if v.rule == "one_line"]

    assert [v.reason for v in violations] == [
        f"a control or invisible character ({code}); a statement is one line of ordinary text"
    ]


def test_an_injected_heading_is_caught_by_the_one_line_rule_alone(facts):
    assert {v.rule for v in verify(PLANTED["one_line"], facts)} == {"one_line"}


_IPW_ESTIMATE = (
    "Under its own assumptions, {est.ipw.method} estimates {est.ipw.point}, "
    "from {est.ipw.ci_lower} to {est.ipw.ci_upper}"
)


@pytest.mark.parametrize(
    "quoted",
    [
        _IPW_ESTIMATE + ".",
        "The standard error of {est.ipw.method} is {est.ipw.se}.",
        "The p-value for {est.ipw.method} is {est.ipw.p_value}.",
        "The interval for {est.ipw.method} reaches {est.ipw.ci_upper}.",
    ],
    ids=["point-and-interval", "se", "p_value", "a-bound-alone"],
)
def test_a_number_from_an_insufficient_estimate_needs_its_band(quoted, facts):
    insufficient = with_band(facts, "ipw", "INSUFFICIENT")

    reasons = list(verify_module._insufficient_caveat(quoted, insufficient))

    assert reasons == [
        "{est.ipw.method}'s estimate is quoted without {score.ipw.band}, which is INSUFFICIENT"
    ]


def test_citing_the_band_in_the_same_statement_is_the_caveat(facts):
    insufficient = with_band(facts, "ipw", "INSUFFICIENT")
    quoted = _IPW_ESTIMATE + ", although its validity band is {score.ipw.band}."

    assert list(verify_module._insufficient_caveat(quoted, insufficient)) == []


def test_the_band_cited_in_another_statement_is_not_the_caveat(good, facts):
    # The good narrative cites ipw's band in its validity section; the estimate
    # statement, read alone, still quotes a number without it.
    insufficient = with_band(facts, "ipw", "INSUFFICIENT")

    violations = [v for v in verify(good, insufficient) if v.rule == "insufficient_caveat"]

    assert [v.section for v in violations] == ["estimates"]


@pytest.mark.parametrize("band", ["WEAK", "MODERATE", "STRONG"])
def test_an_estimate_from_any_other_band_needs_no_caveat(band, facts):
    graded = with_band(facts, "ipw", band)

    assert list(verify_module._insufficient_caveat(_IPW_ESTIMATE + ".", graded)) == []


def test_a_fact_about_an_insufficient_method_that_is_no_number_needs_no_caveat(facts):
    # The method's name, estimand or direction quotes no estimate.
    insufficient = with_band(facts, "ipw", "INSUFFICIENT")
    statement = "{est.ipw.method} estimates the {est.ipw.estimand}, {est.ipw.direction}."

    assert list(verify_module._insufficient_caveat(statement, insufficient)) == []
