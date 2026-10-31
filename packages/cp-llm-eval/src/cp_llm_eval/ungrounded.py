"""The run path for the arm with no verifier, and the prompt it is given.

`cp_llm.interpret` is not reusable here and should not be: it verifies before
it renders, so a caller cannot render first and check later. That property is
worth keeping, so this module makes its own call -- one call, no verifier, no
substitution -- and stores the reply as the model wrote it.

The prompt is phase 1's with the placeholder machinery taken out and nothing
else weakened, built from `system_prompt()` by anchor so that a change to
phase 1's prompt either reaches this arm or fails loudly. A crippled control
would make the study's central comparison meaningless, and a reviewer will
read the diff.
"""

from __future__ import annotations

from dataclasses import dataclass

from cp_llm.facts import FactTable
from cp_llm.interpret import Attempt, ModelDidNotAnswerError
from cp_llm.model import NarrativeModel, Turn
from cp_llm.narrative import Narrative
from cp_llm.prompt import facts_message, prompt_version, system_prompt

# The outcome an unverified reply is stored under. The arm has no verifier, so
# it can neither verify nor refuse: the model's one reply is the narrative,
# and the values in it are the model's own.
UNVERIFIED = "unverified"

# The most model calls one of this arm's narratives can cost. There is no
# verifier to send a rejected reply back to, so there is nothing to retry
# against.
CALLS = 1

# Rule 1 and the two vocabulary rules, replaced by the one rule below: the
# only part of phase 1's prompt this arm does not get.
_RULE_1 = "1. Never write a value."
_RULE_4 = "4. When a statement cites"

# Rule 1 for an arm that writes its own values. Its last sentence is phase
# 1's, word for word: how to write a number is formatting, not grounding.
_STATE_VALUES = """\
1. State each value as the facts give it. Wherever a number, a band, a status, \
a method name or any other fact value belongs, write the value, accurately, as \
the fact holds it. Where a rule below says a statement cites a fact, state that \
fact's value in the statement; an id in braces, such as {est.aipw.point}, names \
the fact whose value belongs there, and you write the value and never the id. \
Write no sign or operator before a number, and write a range as from one bound \
to the other.
"""


def ungrounded_prompt() -> str:
    """Phase 1's prompt with rules 1 to 3 replaced by one rule: state the values.

    Rules 1 (never write a value), 2 (no digits outside a placeholder) and 3
    (controlled vocabulary) are the three the verifier enforces on grounded
    replies, and they go together with the placeholder schema. Every other
    instruction stays word for word -- the sections and what each must cover,
    the structural rules 4 to 9, the caution against overclaiming, rule 8's
    assumption clause -- including the sentence that says a reply breaking a
    rule is rejected, which is kept so that this arm is told no less than the
    grounded arms were.

    The numbering keeps its gap: the next rule is 4. Renumbering would change
    the text of rules this arm is meant to be given unchanged.
    """
    base = system_prompt()
    for anchor in (_RULE_1, _RULE_4):
        if base.count(anchor) != 1:
            raise ValueError(f"the base prompt no longer contains {anchor!r} exactly once")
    start, end = base.index(_RULE_1), base.index(_RULE_4)
    if start > end:
        raise ValueError("the base prompt's rules are no longer in the order this arm assumes")
    return base[:start] + _STATE_VALUES + base[end:]


@dataclass(frozen=True)
class Unverified:
    """One reply from the arm with no verifier, stored as the model wrote it.

    There is no `rendered` field, and that is the point: nothing was
    substituted because nothing was a placeholder, so the narrative is
    already the text a reader of this arm would be shown. `attempts` holds
    the one reply, in the shape a grounded attempt is stored in, with no
    violations -- not because none were found, but because nothing looked.
    """

    narrative: Narrative
    model: str
    prompt_version: str
    attempts: tuple[Attempt, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "narrative": self.narrative.model_dump(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


def write_unverified(
    facts: FactTable, model: NarrativeModel, system: str | None = None
) -> Unverified:
    """Ask `model` for one narrative of `facts` and keep it, whatever it says.

    Nothing is verified and nothing is substituted. A reply with no narrative
    -- a refusal, a reply cut off at the token limit, or one that did not fit
    the schema -- raises `ModelDidNotAnswerError`, the same as on a grounded
    arm, so both arms store a no-answer the same way. It is not retried:
    there is no verifier whose complaint a second attempt could answer.
    """
    if system is None:
        system = ungrounded_prompt()
    reply = model.complete(system, [Turn("user", facts_message(facts))])
    if reply.narrative is None:
        raise ModelDidNotAnswerError(reply.stop_reason)
    return Unverified(
        narrative=reply.narrative,
        model=reply.model,
        prompt_version=prompt_version(system),
        attempts=(Attempt(reply.narrative, (), reply.usage),),
    )
