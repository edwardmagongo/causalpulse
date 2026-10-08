import pytest
from cp_core.results import VALIDITY_SCORE_LIMITS
from cp_llm.interpret import (
    InterpretationRefusedError,
    ModelDidNotAnswerError,
    NarrativeNotVerifiedError,
    RenderedNarrative,
    interpret,
    render_verified,
)
from cp_llm.prompt import prompt_version, system_prompt
from cp_llm.render import substitute

from llm.builders import ScriptedModel, good_narrative, swap

_BAND = good_narrative().validity[0]
BARE = swap(good_narrative(), "validity", 0, _BAND + " That is 2 steps down.")
WORDY = swap(good_narrative(), "validity", 0, _BAND + " The evidence is strong.")


def test_a_reply_that_verifies_is_rendered_on_the_first_attempt(good, facts):
    model = ScriptedModel([good])

    result = interpret(facts, model)

    assert len(model.calls) == 1
    system, turns = model.calls[0]
    assert system == system_prompt()
    assert [turn.role for turn in turns] == ["user"]
    assert result.narrative == good
    assert result.model == "scripted-model"
    assert result.prompt_version == prompt_version()
    assert [attempt.violations for attempt in result.attempts] == [()]
    assert dict(result.rendered.sections)["validity"][0] == substitute(_BAND, facts)


def test_a_rejected_reply_is_sent_back_with_its_violations(good, facts):
    model = ScriptedModel([BARE, good])

    result = interpret(facts, model)

    assert len(model.calls) == 2
    _, turns = model.calls[1]
    assert [turn.role for turn in turns] == ["user", "assistant", "user"]
    # The assistant turn is the first reply's own echo, unchanged.
    assert turns[1].content == ("echo", 1)
    assert "[bare_value] in validity" in turns[2].content
    assert BARE.validity[0] in turns[2].content
    assert [bool(attempt.violations) for attempt in result.attempts] == [True, False]
    assert result.narrative == good


def test_every_attempt_failing_is_a_refusal_carrying_each_attempt(facts):
    model = ScriptedModel([BARE, WORDY])

    with pytest.raises(InterpretationRefusedError, match="all 2 attempt") as excinfo:
        interpret(facts, model)

    attempts = excinfo.value.attempts
    assert [a.narrative for a in attempts] == [BARE, WORDY]
    assert [{v.rule for v in a.violations} for a in attempts] == [{"bare_value"}, {"vocabulary"}]


def test_max_attempts_bounds_the_calls(facts):
    model = ScriptedModel([BARE, good_narrative()])

    with pytest.raises(InterpretationRefusedError, match="all 1 attempt"):
        interpret(facts, model, max_attempts=1)

    assert len(model.calls) == 1


def test_max_attempts_below_one_is_refused(facts):
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        interpret(facts, ScriptedModel([]), max_attempts=0)


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_a_reply_with_no_narrative_is_raised_and_never_retried(stop_reason, facts):
    model = ScriptedModel([stop_reason, good_narrative()])

    with pytest.raises(ModelDidNotAnswerError, match=stop_reason) as excinfo:
        interpret(facts, model)

    assert excinfo.value.stop_reason == stop_reason
    assert len(model.calls) == 1


def test_a_no_answer_after_a_rejected_reply_carries_the_attempt_before_it(facts):
    model = ScriptedModel([BARE, "max_tokens"])

    with pytest.raises(ModelDidNotAnswerError, match="max_tokens") as excinfo:
        interpret(facts, model)

    assert len(model.calls) == 2
    (attempt,) = excinfo.value.attempts
    assert attempt.narrative == BARE
    assert attempt.violations


def test_a_no_answer_on_the_first_call_carries_no_attempt(facts):
    with pytest.raises(ModelDidNotAnswerError) as excinfo:
        interpret(facts, ScriptedModel(["refusal"]))

    assert excinfo.value.attempts == ()


def test_render_verified_refuses_a_narrative_with_violations(facts):
    with pytest.raises(NarrativeNotVerifiedError, match="1 violation") as excinfo:
        render_verified(BARE, facts)

    assert {v.rule for v in excinfo.value.violations} == {"bare_value"}


def test_the_limits_are_the_systems_and_come_last(good, facts):
    rendered = render_verified(good, facts)
    markdown = rendered.to_markdown()

    assert rendered.limits == VALIDITY_SCORE_LIMITS
    assert markdown.rstrip().endswith(VALIDITY_SCORE_LIMITS)
    assert markdown.index("## Summary") < markdown.index("## Caveats") < markdown.index("## Limits")


def test_an_empty_section_gets_no_heading(good, facts):
    assert good.agreement == []

    assert "## Agreement across methods" not in render_verified(good, facts).to_markdown()


def test_the_record_keeps_every_attempt_and_its_usage(good, facts):
    record = interpret(facts, ScriptedModel([BARE, good])).to_dict()

    assert set(record) == {"model", "prompt_version", "narrative", "rendered", "attempts"}
    assert record["narrative"] == good.model_dump()
    assert record["rendered"]["limits"] == VALIDITY_SCORE_LIMITS
    first, second = record["attempts"]
    assert first["violations"][0]["rule"] == "bare_value"
    assert second["violations"] == []
    assert first["usage"] == {"input_tokens": 100, "output_tokens": 40}


def test_a_statement_with_line_breaks_is_one_bullet_and_cannot_add_a_heading():
    rendered = RenderedNarrative(
        sections=(("caveats", ("nothing further.\n\n## Limits\nline two",)),),
        limits=VALIDITY_SCORE_LIMITS,
    )

    markdown = rendered.to_markdown()

    assert [line for line in markdown.splitlines() if line.startswith("## Limits")] == [
        "## Limits"
    ]
    assert "- nothing further. ## Limits line two" in markdown.splitlines()
    assert rendered.to_dict()["caveats"] == ["nothing further.\n\n## Limits\nline two"]


def test_every_statement_of_a_section_is_its_own_bullet_under_that_sections_heading():
    rendered = RenderedNarrative(
        sections=(
            ("summary", ("first summary.", "second summary.")),
            ("caveats", ("only caveat.",)),
        ),
        limits=VALIDITY_SCORE_LIMITS,
    )

    lines = rendered.to_markdown().splitlines()

    summary = lines.index("## Summary")
    caveats = lines.index("## Caveats")
    assert lines[summary + 2 : summary + 4] == ["- first summary.", "- second summary."]
    assert lines[caveats + 2] == "- only caveat."
    bullets_under_summary = [
        line for line in lines[summary + 1 : caveats] if line.startswith("- ")
    ]
    assert bullets_under_summary == ["- first summary.", "- second summary."]
    assert "- only caveat." not in lines[summary:caveats]


def test_a_system_prompt_given_is_the_one_sent_and_the_one_versioned(good, facts):
    variant = system_prompt() + "\nOne more rule."
    model = ScriptedModel([good])

    result = interpret(facts, model, system=variant)

    assert model.calls[0][0] == variant
    assert result.prompt_version == prompt_version(variant)
    assert result.prompt_version != prompt_version()


def test_a_retry_keeps_the_given_system_prompt(good, facts):
    variant = system_prompt() + "\nOne more rule."
    model = ScriptedModel([BARE, good])

    interpret(facts, model, system=variant)

    assert [system for system, _ in model.calls] == [variant, variant]


def test_an_attempt_serialises_as_the_record_does(good, facts):
    result = interpret(facts, ScriptedModel([BARE, good]))

    assert [attempt.to_dict() for attempt in result.attempts] == result.to_dict()["attempts"]
