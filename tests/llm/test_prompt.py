import json

from cp_core.language import causal_language_lint
from cp_llm import prompt as prompt_module
from cp_llm.prompt import facts_message, prompt_version, system_prompt, violations_message
from cp_llm.verify import Violation
from cp_llm.vocabulary import ALLOWED_PHRASES, CONTROLLED_WORDS, NUMBER_WORDS


def test_the_prompt_lists_every_word_the_verifier_refuses():
    prompt = system_prompt()

    for word in (*NUMBER_WORDS, *CONTROLLED_WORDS):
        assert f" {word}" in prompt, word
    for phrase in ALLOWED_PHRASES:
        assert f'"{phrase}"' in prompt


def test_the_assembled_prompt_passes_the_language_lint():
    # The corpus scan lints each string constant on its own; this lints the
    # prompt as the model receives it, word lists interpolated.
    assert causal_language_lint(system_prompt()) == ()


def test_the_prompt_version_is_stable_and_moves_with_the_vocabulary(monkeypatch):
    before = prompt_version()

    assert before == prompt_version()
    assert len(before) == 12

    monkeypatch.setattr(prompt_module, "NUMBER_WORDS", frozenset({*NUMBER_WORDS, "dozen"}))

    assert prompt_version() != before


def test_the_facts_message_is_the_tables_records_as_json(facts):
    records = json.loads(facts_message(facts))

    assert records == facts.to_records()
    assert set(records[0]) == {"id", "kind", "value", "subject"}


def test_a_rejection_names_each_rule_its_section_and_its_statement():
    message = violations_message(
        [
            Violation("bare_value", "validity", "It is 2.", "a digit outside a placeholder"),
            Violation("completeness", "", "", "the narrative never cites {est.ipw.point}"),
        ]
    )

    assert "- [bare_value] in validity: a digit outside a placeholder" in message
    assert "  statement: It is 2." in message
    assert "- [completeness] the narrative never cites {est.ipw.point}" in message
    assert "statement: \n" not in message


def test_the_prompt_states_the_rules_the_word_lists_cannot_carry():
    prompt = system_prompt()

    assert "no percent sign" in prompt
    assert "a zero result" in prompt
