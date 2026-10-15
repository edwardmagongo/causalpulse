"""The interpreter version: what an interpretation was written and verified under."""

import cp_llm.prompt as prompt
import cp_llm.verify as verify
from cp_llm.version import interpreter_version

# The rule set the version was last reviewed against. A change to it fails
# here: decide whether the change alters what "verified" means, and if it does
# not (a rule renamed with no change in behaviour, say), say so in the commit.
REVIEWED_RULES = (
    "attribution",
    "bare_point",
    "bare_value",
    "completeness",
    "delimiters",
    "insufficient_caveat",
    "language",
    "one_line",
    "resolves",
    "vocabulary",
)
REVIEWED_VERSION = "35d8275efecb"


def test_the_version_is_twelve_hex_characters():
    version = interpreter_version()
    assert len(version) == 12 and all(c in "0123456789abcdef" for c in version)


def test_the_rule_set_and_version_are_the_reviewed_ones():
    names = tuple(sorted([*verify.STATEMENT_RULES, *verify.NARRATIVE_RULES]))
    assert names == REVIEWED_RULES
    assert interpreter_version() == REVIEWED_VERSION


def test_a_new_prompt_moves_the_version(monkeypatch):
    before = interpreter_version()
    monkeypatch.setattr(prompt, "prompt_version", lambda system=None: "000000000000")
    assert interpreter_version() != before


def test_a_new_rule_moves_the_version(monkeypatch):
    before = interpreter_version()
    rule = verify.NARRATIVE_RULES["completeness"]
    monkeypatch.setitem(verify.NARRATIVE_RULES, "a_new_rule", rule)
    assert interpreter_version() != before
