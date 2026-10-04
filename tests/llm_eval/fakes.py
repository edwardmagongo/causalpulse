"""Models for the runner and judge tests: thread-safe, and no network."""

from __future__ import annotations

import threading
from collections.abc import Sequence

from cp_llm.model import ModelCallError, ModelReply, Turn, Usage
from cp_llm.narrative import Narrative

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
