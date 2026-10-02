"""From a fact table to a verified, rendered interpretation -- or a refusal.

`render_verified` is the only way to a rendered narrative, and it verifies
before it substitutes anything, so a caller cannot render first and check
later.
"""

from __future__ import annotations

from dataclasses import dataclass

from cp_core.results import VALIDITY_SCORE_LIMITS

from cp_llm.facts import FactTable
from cp_llm.model import ModelReply, NarrativeModel, Turn, Usage
from cp_llm.narrative import SECTIONS, Narrative
from cp_llm.prompt import facts_message, prompt_version, system_prompt, violations_message
from cp_llm.render import substitute
from cp_llm.verify import Violation, verify

_HEADINGS = {
    "summary": "Summary",
    "estimates": "Estimates",
    "validity": "Validity",
    "agreement": "Agreement across methods",
    "caveats": "Caveats",
}


@dataclass(frozen=True)
class Attempt:
    """One reply from the model, and what the verifier made of it."""

    narrative: Narrative
    violations: tuple[Violation, ...]
    usage: Usage


class InterpretationRefusedError(Exception):
    """Every attempt broke a rule. Nothing was rendered."""

    def __init__(self, attempts: tuple[Attempt, ...]) -> None:
        self.attempts = attempts
        last = attempts[-1].violations
        super().__init__(
            f"the narrative failed verification on all {len(attempts)} attempt(s); "
            f"the last had {len(last)} violation(s)"
        )


class ModelDidNotAnswerError(Exception):
    """The model stopped without a complete reply: a refusal, or a token limit."""

    def __init__(self, stop_reason: str) -> None:
        self.stop_reason = stop_reason
        super().__init__(f"the model returned no narrative (stop reason: {stop_reason})")


class NarrativeNotVerifiedError(ValueError):
    def __init__(self, violations: tuple[Violation, ...]) -> None:
        self.violations = violations
        super().__init__(f"refusing to render a narrative with {len(violations)} violation(s)")


@dataclass(frozen=True)
class RenderedNarrative:
    """The narrative with every placeholder replaced, section by section."""

    sections: tuple[tuple[str, tuple[str, ...]], ...]
    limits: str

    def to_markdown(self) -> str:
        lines = ["# Interpretation", ""]
        for section, statements in self.sections:
            if not statements:
                continue
            lines.extend([f"## {_HEADINGS[section]}", ""])
            lines.extend(f"- {statement}" for statement in statements)
            lines.append("")
        lines.extend(["## Limits", "", self.limits, ""])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            **{section: list(statements) for section, statements in self.sections},
            "limits": self.limits,
        }


@dataclass(frozen=True)
class Interpretation:
    """`attempts` holds every reply in order; the last one is the one rendered."""

    narrative: Narrative
    rendered: RenderedNarrative
    model: str
    prompt_version: str
    attempts: tuple[Attempt, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "narrative": self.narrative.model_dump(),
            "rendered": self.rendered.to_dict(),
            "attempts": [
                {
                    "narrative": attempt.narrative.model_dump(),
                    "violations": [
                        {
                            "rule": v.rule,
                            "section": v.section,
                            "statement": v.statement,
                            "reason": v.reason,
                        }
                        for v in attempt.violations
                    ],
                    "usage": {
                        "input_tokens": attempt.usage.input_tokens,
                        "output_tokens": attempt.usage.output_tokens,
                    },
                }
                for attempt in self.attempts
            ],
        }


def render_verified(narrative: Narrative, facts: FactTable) -> RenderedNarrative:
    """Render `narrative`, or raise if the verifier finds anything."""
    violations = verify(narrative, facts)
    if violations:
        raise NarrativeNotVerifiedError(violations)
    return RenderedNarrative(
        sections=tuple(
            (section, tuple(substitute(s, facts) for s in getattr(narrative, section)))
            for section in SECTIONS
        ),
        # Appended here and never written by the model, which has no
        # placeholder for it: the statement cannot be reworded or dropped.
        limits=VALIDITY_SCORE_LIMITS,
    )


def interpret(facts: FactTable, model: NarrativeModel, max_attempts: int = 2) -> Interpretation:
    """Ask `model` for a narrative of `facts` until one verifies, or refuse.

    A reply that breaks a rule is sent back with its violations, in the same
    conversation, at most `max_attempts - 1` times. A refusal or a truncated
    reply is not retried: asking again fixes neither.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
    system = system_prompt()
    turns: list[Turn] = [Turn("user", facts_message(facts))]
    attempts: list[Attempt] = []
    reply: ModelReply
    for _ in range(max_attempts):
        reply = model.complete(system, turns)
        if reply.narrative is None:
            raise ModelDidNotAnswerError(reply.stop_reason)
        violations = verify(reply.narrative, facts)
        attempts.append(Attempt(reply.narrative, violations, reply.usage))
        if not violations:
            return Interpretation(
                narrative=reply.narrative,
                rendered=render_verified(reply.narrative, facts),
                model=reply.model,
                prompt_version=prompt_version(),
                attempts=tuple(attempts),
            )
        turns.extend(
            [Turn("assistant", reply.echo), Turn("user", violations_message(violations))]
        )
    raise InterpretationRefusedError(tuple(attempts))
