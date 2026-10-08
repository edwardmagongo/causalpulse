import pytest
from cp_llm.prompt import prompt_version, system_prompt
from cp_llm_eval import variants as variants_module
from cp_llm_eval.variants import BASELINE, CITE_EACH_STATUS, CITE_EACH_STATUS_SENTENCE, VARIANTS


def test_the_baseline_is_phase_ones_prompt():
    assert BASELINE.system() == system_prompt()


def test_the_variant_adds_one_sentence_to_rule_8_and_nothing_else():
    base, variant = system_prompt(), CITE_EACH_STATUS.system()

    assert variant.replace(" " + CITE_EACH_STATUS_SENTENCE, "", 1) == base
    rule_8 = variant[variant.index("8. ") : variant.index("9. ")]
    assert CITE_EACH_STATUS_SENTENCE in rule_8


def test_the_two_prompts_record_two_versions():
    assert prompt_version(CITE_EACH_STATUS.system()) != prompt_version(BASELINE.system())


def test_a_variant_whose_anchor_has_gone_is_refused(monkeypatch):
    monkeypatch.setattr(variants_module, "system_prompt", lambda: "a prompt without rule 8")
    build = variants_module._with_sentence_after("rests on.", "More.")

    with pytest.raises(ValueError, match="no longer contains"):
        build()


def test_variants_are_registered_by_name():
    assert set(VARIANTS) == {"baseline", "cite-each-status"}
