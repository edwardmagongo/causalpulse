"""The three arms, and what the control prompt does and does not drop.

The study's claim rests on the control arm being a fair opponent, so the diff
against phase 1's prompt is asserted here rather than described: exactly
rules 1 to 3, replaced by one rule that says to state the values, and not a
word of anything else.
"""

from __future__ import annotations

import re

import pytest
from cp_llm.prompt import prompt_version, system_prompt
from cp_llm_eval import ungrounded as ungrounded_module
from cp_llm_eval.arms import ARMS, UNGROUNDED
from cp_llm_eval.ungrounded import ungrounded_prompt
from cp_llm_eval.variants import VARIANTS, Variant


def _rules(prompt: str) -> dict[str, str]:
    """Each numbered rule of `prompt`, by its number. One rule is one line."""
    return {m.group(1): m.group(0) for m in re.finditer(r"^(\d)\. .*$", prompt, re.MULTILINE)}


def _prose(prompt: str) -> list[str]:
    """Everything that is not a numbered rule: the task, the sections, the closing note."""
    return [line for line in prompt.splitlines() if not re.match(r"^\d\. ", line)]


def test_the_control_prompt_drops_rules_one_to_three_and_keeps_the_rest_word_for_word():
    base, control = _rules(system_prompt()), _rules(ungrounded_prompt())

    assert set(base) - set(control) == {"2", "3"}
    assert set(control) - set(base) == set()
    kept = {n: rule for n, rule in base.items() if n not in ("1", "2", "3")}
    assert {n: rule for n, rule in control.items() if n != "1"} == kept


def test_everything_outside_the_rules_is_phase_ones_prompt_word_for_word():
    """The sections, what each must cover, and the closing note on the limits."""
    assert _prose(ungrounded_prompt()) == _prose(system_prompt())


def test_rule_one_tells_the_control_arm_to_state_the_values():
    control = ungrounded_prompt()

    assert "Never write a value" not in control
    assert "write a placeholder" not in control
    assert _rules(control)["1"].startswith("1. State each value as the facts give it.")


def test_the_control_arm_is_told_no_less_than_the_grounded_arms_were():
    """Kept deliberately, and the one sentence worth saying so about.

    Nothing rejects an unverified reply, so this sentence promises the control
    arm a check it will not get. Dropping it would be the one place the
    control is told less than the grounded arms, and a weakened control is
    the failure mode that would make the whole comparison worthless.
    """
    assert "A reply that breaks any rule below is rejected." in ungrounded_prompt()


def test_the_numbering_keeps_its_gap():
    """Renumbering would change the text of rules kept word for word."""
    assert sorted(_rules(ungrounded_prompt())) == ["1", "4", "5", "6", "7", "8", "9"]


def test_a_control_prompt_whose_anchor_has_gone_is_refused(monkeypatch):
    monkeypatch.setattr(ungrounded_module, "system_prompt", lambda: "a prompt with no rules")

    with pytest.raises(ValueError, match="no longer contains"):
        ungrounded_prompt()


def test_a_control_prompt_whose_rules_have_moved_is_refused(monkeypatch):
    monkeypatch.setattr(
        ungrounded_module,
        "system_prompt",
        lambda: "4. When a statement cites a fact\n1. Never write a value.\n",
    )

    with pytest.raises(ValueError, match="no longer in the order"):
        ungrounded_prompt()


def test_the_three_arms_are_registered_by_name():
    assert set(ARMS) == {"baseline", "cite-each-status", "ungrounded"}
    # Derived from the variants, so a prompt added there is an arm here.
    assert set(VARIANTS) < set(ARMS)


def test_only_the_ungrounded_arm_runs_without_the_verifier():
    assert {name for name, arm in ARMS.items() if not arm.grounded} == {"ungrounded"}


def test_every_grounded_arm_still_forbids_writing_a_value():
    grounded = [arm for arm in ARMS.values() if arm.grounded]

    assert grounded
    assert all("Never write a value." in arm.system() for arm in grounded)


def test_the_ungrounded_arm_is_not_a_variant():
    """A variant's replies reach `interpret()`, which would refuse every one of these."""
    assert not isinstance(UNGROUNDED, Variant)
    assert isinstance(ARMS["baseline"], Variant)


def test_the_three_arms_record_three_prompt_versions():
    assert len({prompt_version(arm.system()) for arm in ARMS.values()}) == 3
