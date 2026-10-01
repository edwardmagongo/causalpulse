"""The seam between the interpreter and whatever writes the narrative.

A test seam, not a provider abstraction: there is one real implementation,
`cp_llm.client.ClaudeModel`. Everything else in the package is written against
this protocol, so it runs with a scripted model and no network, and without
importing the SDK.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from cp_llm.narrative import Narrative

DEFAULT_MODEL = "claude-opus-5-5"
Effort = Literal["low", "medium", "high", "xhigh", "max"]
DEFAULT_EFFORT: Effort = "medium"


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Turn:
    """One message. An assistant turn's `content` is a `ModelReply.echo`, unchanged."""

    role: Literal["user", "assistant"]
    content: object


@dataclass(frozen=True)
class ModelReply:
    """`narrative` is `None` exactly when the model did not finish an answer."""

    narrative: Narrative | None
    stop_reason: str
    model: str
    usage: Usage
    # What to send back as the assistant turn to continue this conversation.
    # Opaque to every caller: only the adapter that produced it reads it.
    echo: object


class ModelCallError(Exception):
    """The model could not be reached or would not serve the request.

    Raised by an adapter in place of its SDK's own error, which stays chained
    as the cause, so that a caller can report it without importing the SDK.
    """


class NarrativeModel(Protocol):
    def complete(self, system: str, turns: Sequence[Turn]) -> ModelReply: ...
