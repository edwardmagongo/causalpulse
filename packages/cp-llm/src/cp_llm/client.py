"""Claude as a `NarrativeModel`, and the one module that imports the SDK."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import anthropic
from anthropic.types import MessageParam

from cp_llm.model import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    Effort,
    ModelCallError,
    ModelReply,
    Turn,
    Usage,
)
from cp_llm.narrative import Narrative

MAX_TOKENS = 16000
_SCHEMA = anthropic.transform_schema(Narrative)


class ClaudeModel:
    """Claude through the Messages API, constrained to the `Narrative` schema.

    Server-side fallback is deliberately not enabled: a reply always comes
    from the model named in it. A refusal is returned as a reply with no
    narrative, for the caller to raise on.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: Effort = DEFAULT_EFFORT,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._model = model
        self._effort = effort
        self._client = client if client is not None else anthropic.Anthropic()

    def complete(self, system: str, turns: Sequence[Turn]) -> ModelReply:
        messages = [
            cast(MessageParam, {"role": turn.role, "content": turn.content}) for turn in turns
        ]
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=messages,
                output_config={
                    "effort": self._effort,
                    "format": {"type": "json_schema", "schema": _SCHEMA},
                },
            )
        except anthropic.APIError as exc:
            raise ModelCallError(f"the model API call failed: {type(exc).__name__}: {exc}") from exc
        except TypeError as exc:
            # With no credentials at all the SDK raises a bare TypeError while
            # building the request headers, before anything is sent.
            if "authentication" not in str(exc):
                raise
            raise ModelCallError(
                "no API credentials were found; set ANTHROPIC_API_KEY or run `ant auth login`"
            ) from exc
        stop_reason = response.stop_reason or "unknown"
        narrative: Narrative | None = None
        # Read the stop reason before the text: a refusal or a reply cut off at
        # the token limit carries text that is not the JSON it was asked for.
        if stop_reason == "end_turn":
            text = "".join(block.text for block in response.content if block.type == "text")
            narrative = Narrative.model_validate_json(text)
        return ModelReply(
            narrative=narrative,
            stop_reason=stop_reason,
            model=response.model,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            echo=response.content,
        )
