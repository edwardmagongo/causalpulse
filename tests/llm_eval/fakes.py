"""Models for the runner and judge tests: thread-safe, and no network."""

from __future__ import annotations

import threading
from collections.abc import Sequence

from cp_llm.model import ModelCallError, ModelReply, Turn, Usage
from cp_llm.narrative import Narrative
from cp_llm.render import format_fact
from cp_llm_eval.ungrounded import UNVERIFIED

# An uncited status summary (D4) that phase 1's verifier accepts: true of the
# `entry` fit, whose difference_in_means checks are all NOT_TESTABLE.
D4_STATEMENT = "No check for {est.difference_in_means.method} raised a concern."


class ReplyModel:
    """Answers every call with the next reply from `script`, cycling, and counts calls.

    A `Narrative` is a finished reply; a string is a stop reason for a reply
    with no narrative; `None` raises a transport failure.
    """

    def __init__(self, script: Sequence[Narrative | str | None]) -> None:
        self._script = list(script)
        self._lock = threading.Lock()
        self.calls = 0
        self.systems: list[str] = []

    def complete(self, system: str, turns: Sequence[Turn]) -> ModelReply:
        with self._lock:
            item = self._script[self.calls % len(self._script)]
            self.calls += 1
            self.systems.append(system)
        if item is None:
            raise ModelCallError("scripted transport failure")
        if isinstance(item, str):
            return ModelReply(None, item, "scripted-model", Usage(1000, 0), None)
        return ModelReply(item, "end_turn", "scripted-model", Usage(1000, 100), ("echo",))


class ReplyJudge:
    """A judge that answers every call with `reply_for(n_statements)`, and counts calls.

    `reply_for` returning None stands for a reply with no verdict; raising
    stands for a transport failure.
    """

    def __init__(self, reply_for) -> None:
        self._reply_for = reply_for
        self._lock = threading.Lock()
        self.calls = 0
        self.messages: list[str] = []

    def judge(self, system: str, user: str):
        from cp_llm_eval.judge import JudgeCall

        with self._lock:
            self.calls += 1
            self.messages.append(user)
        n_statements = sum(1 for line in user.splitlines() if line[:1].isdigit())
        reply = self._reply_for(n_statements)
        stop_reason = "end_turn" if reply else "refusal"
        return JudgeCall(reply, stop_reason, "scripted-judge", Usage(2000, 200))


# A value no fact of the `entry` fit holds at any rounding, and no prose fact
# of it mentions. `test_a_fabricated_value...` fails loudly if that changes.
FABRICATED = "7.7321"


def _value(facts, fact_id):
    """The fact's value as the renderer writes it, so the prose below is attributable."""
    fact = facts.get(fact_id)
    assert fact is not None, fact_id
    return format_fact(fact)


def ungrounded_narrative(facts):
    """Prose with the table's own values in it, and two planted defects.

    The comparative is false of the values: ipw's point estimate is the lower
    of the two, so "higher than" contradicts the table. D1 can only see that
    once both numbers have been matched back to the facts they came from. The
    estimates statement carries a value no fact supports at any rounding.
    """
    ipw, dim = facts.get("est.ipw.point"), facts.get("est.difference_in_means.point")
    assert ipw is not None and dim is not None
    assert ipw.value < dim.value, "the planted comparative is only false while ipw is lower"
    return Narrative(
        summary=[
            f"ipw estimates the effect at {_value(facts, 'est.ipw.point')}, which is higher "
            f"than difference_in_means at {_value(facts, 'est.difference_in_means.point')}."
        ],
        estimates=[f"A further check put the effect at {FABRICATED} on the same scale."],
        validity=["The design is a cross-section and the usual caveats apply."],
        agreement=[],
        caveats=[],
    )


def ungrounded_record(entry, narrative, replicate=1):
    """What the ungrounded run path stores: one reply, unverified, values and all.

    Written by hand, so that `tests/llm_eval/test_ungrounded.py` can hold the
    real run path to this shape rather than to itself. The arm has no
    verifier, so there is no `rendered` section: the statement the model wrote
    is already the statement a reader is shown.
    """
    body = narrative.model_dump()
    return {
        "key": {
            "case_id": entry.case_id,
            "facts_sha256": entry.facts_sha256,
            "variant": "ungrounded",
            "prompt_version": "ungrounded-prompt",
            "model": "claude-opus-5-5",
            "effort": "medium",
            "replicate": replicate,
        },
        "outcome": UNVERIFIED,
        "interpretation": {
            "model": "scripted-model",
            "prompt_version": "ungrounded-prompt",
            "narrative": body,
        },
        "attempts": [{"narrative": body}],
        "calls": [],
    }
