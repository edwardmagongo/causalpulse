"""What a refused interpretation tells its reader: each broken rule in plain words."""

from cp_api.app import RULE_DESCRIPTIONS, refusal_reason
from cp_llm.verify import NARRATIVE_RULES, STATEMENT_RULES


def test_every_verifier_rule_has_a_description():
    assert set(RULE_DESCRIPTIONS) == set(STATEMENT_RULES) | set(NARRATIVE_RULES)


def test_one_rule_reads_as_one_sentence():
    assert refusal_reason(["bare_point"]) == (
        "The last attempt was refused because it gave an estimate without its interval."
    )


def test_several_rules_are_listed_in_the_verifiers_order_once_each():
    assert refusal_reason(["completeness", "attribution", "attribution"]) == (
        "The last attempt was refused because it cited a method's result without naming the "
        "method; and it left out a result it must report: an estimate, a band, a comparison, "
        "a fit failure, or a failed or warned check."
    )


def test_a_rule_without_a_description_is_still_named():
    assert refusal_reason(["a_new_rule"]) == (
        "The last attempt was refused because it broke the verifier's rule a_new_rule."
    )
