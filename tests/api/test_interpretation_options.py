"""The operator's list of models and efforts a server offers for interpretations."""

from typing import get_args

import pytest
from cp_api.db import Interpretation
from cp_api.settings import (
    _DEFAULT_OPTIONS,
    EFFORTS,
    MODEL_ID_LIMIT,
    InterpretationOption,
    interpretation_options,
)
from cp_llm.interpret import MEASURED_CHOICE
from cp_llm.model import Effort

SETTING = "CP_INTERPRETATION_OPTIONS"
DEFAULT = (InterpretationOption("claude-opus-5-5", "medium"),)


def _read(value: str) -> tuple[InterpretationOption, ...]:
    return interpretation_options({SETTING: value})


def test_the_default_when_the_setting_is_absent_and_when_it_is_empty():
    assert interpretation_options({}) == DEFAULT
    assert _read("") == DEFAULT
    assert _read("  \t ") == DEFAULT


def test_without_a_mapping_it_reads_the_process_environment(monkeypatch):
    monkeypatch.setenv(SETTING, "claude-sonnet-5-5:low")
    assert interpretation_options() == (InterpretationOption("claude-sonnet-5-5", "low"),)
    monkeypatch.delenv(SETTING)
    assert interpretation_options() == DEFAULT


def test_one_entry():
    assert _read("claude-sonnet-5-5:high") == (InterpretationOption("claude-sonnet-5-5", "high"),)


def test_several_entries_with_spaces_keep_their_order():
    options = _read(" claude-sonnet-5-5 : low ,claude-opus-5-5:max,  claude-fable-5-1:xhigh ")
    assert options == (
        InterpretationOption("claude-sonnet-5-5", "low"),
        InterpretationOption("claude-opus-5-5", "max"),
        InterpretationOption("claude-fable-5-1", "xhigh"),
    )


def test_the_same_model_at_two_efforts_is_two_entries():
    assert len(_read("claude-opus-5-5:low,claude-opus-5-5:high")) == 2


def test_an_entry_has_a_label():
    assert InterpretationOption("claude-sonnet-5-5", "low").label == "claude-sonnet-5-5, low"


def test_an_option_is_frozen():
    with pytest.raises(AttributeError):
        DEFAULT[0].effort = "low"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("value", "entry", "wrong"),
    [
        ("claude-opus-5-5", "claude-opus-5-5", "not model:effort"),
        ("claude-opus-5-5:medium:low", "claude-opus-5-5:medium:low", "not model:effort"),
        (":medium", ":medium", "model"),
        ("claude-opus-5-5:", "claude-opus-5-5:", "effort"),
        ("Claude-Opus-5-5:medium", "Claude-Opus-5-5:medium", "model"),
        ("claude opus:medium", "claude opus:medium", "model"),
        ("-claude:medium", "-claude:medium", "model"),
        ("claude-opus-5-5:Medium", "claude-opus-5-5:Medium", "unknown effort"),
        ("claude-opus-5-5:extreme", "claude-opus-5-5:extreme", "unknown effort"),
        ("claude-opus-5-5:medium,", "", "empty"),
        ("claude-opus-5-5:medium,,claude-sonnet-5-5:low", "", "empty"),
    ],
)
def test_a_malformed_value_names_the_setting_and_the_entry(value, entry, wrong):
    with pytest.raises(ValueError) as caught:
        _read(value)
    message = str(caught.value)
    assert "\n" not in message
    assert SETTING in message
    assert wrong in message
    if entry:
        assert f"'{entry}'" in message


@pytest.mark.parametrize(
    ("entry", "escaped"),
    [
        ("a\nb:low", "a\\nb:low"),
        ("a\rb:low", "a\\rb:low"),
        ("a\x0bb:low", "a\\x0bb:low"),
        ("a\x85b:low", "a\\x85b:low"),
        ("a\u2028b:low", "a\\u2028b:low"),
        ("a\u2029b:low", "a\\u2029b:low"),
    ],
)
def test_an_entry_with_a_line_break_is_shown_escaped_on_one_line(entry, escaped):
    with pytest.raises(ValueError) as caught:
        _read(f"ok:low,{entry}")
    message = str(caught.value)
    assert len(message.splitlines()) == 1
    assert f"'{escaped}'" in message


def test_a_printable_look_alike_is_shown_as_written():
    with pytest.raises(ValueError) as caught:
        _read("\u0430:low")  # Cyrillic a
    assert "'\u0430:low'" in str(caught.value)


def test_a_huge_entry_is_cut_in_the_message():
    entry = "x" * 100_000
    with pytest.raises(ValueError) as caught:
        _read(entry)
    message = str(caught.value)
    assert len(message) < 300
    assert "'" + "x" * 80 + "...'" in message


def test_a_model_id_is_at_most_as_long_as_its_column():
    longest = "m" * MODEL_ID_LIMIT
    assert _read(f"{longest}:low") == (InterpretationOption(longest, "low"),)
    with pytest.raises(ValueError) as caught:
        _read(f"{longest}m:low")
    message = str(caught.value)
    assert "model id longer than 64 characters" in message
    assert SETTING in message


def test_a_repeated_entry_is_refused_naming_the_second_one():
    with pytest.raises(ValueError) as caught:
        _read("claude-opus-5-5:medium, claude-sonnet-5-5:low,claude-opus-5-5:medium")
    message = str(caught.value)
    assert SETTING in message
    assert "repeated" in message
    assert "'claude-opus-5-5:medium'" in message


def test_the_first_wrong_entry_is_the_one_named():
    with pytest.raises(ValueError) as caught:
        _read("claude-opus-5-5:medium,bad,worse:nope")
    assert "'bad'" in str(caught.value)
    assert "worse" not in str(caught.value)


def test_the_efforts_are_the_ones_the_llm_layer_knows():
    assert get_args(Effort) == EFFORTS


def test_the_model_id_limit_is_the_model_columns_width():
    assert Interpretation.__table__.c.model.type.length == MODEL_ID_LIMIT


def test_the_default_entry_is_the_one_the_misreading_study_measured():
    # A server without the setting writes without the unmeasured note.
    (parsed,) = _read(_DEFAULT_OPTIONS)
    assert interpretation_options({}) == (parsed,)
    assert (parsed.model, parsed.effort) == MEASURED_CHOICE
