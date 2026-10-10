"""A narrative that verifies against any fact table, and a model that returns it.

The interpretation job is tested without the network: `ScriptedReplies` is
handed to it in place of the model. `narrative_for` writes, from the facts
alone, the plainest narrative phase 1's verifier accepts -- every statement
cites what completeness demands, and nothing is written as free text that a
rule refuses -- so a test can build it for whatever analysis it ran.
"""

from __future__ import annotations

from collections.abc import Sequence

from cp_llm.facts import FactTable
from cp_llm.model import ModelReply, Turn, Usage
from cp_llm.narrative import Narrative


def _methods(facts: FactTable) -> list[str]:
    return [
        f.id.split(".")[1] for f in facts if f.id.startswith("est.") and f.id.endswith(".point")
    ]


def narrative_for(facts: FactTable) -> Narrative:
    estimates, validity, agreement, caveats = [], [], [], []
    for method in _methods(facts):
        estimates.append(
            f"Under its own assumptions, {{est.{method}.method}} estimates the "
            f"{{est.{method}.estimand}} at {{est.{method}.point}}, with a "
            f"{{est.{method}.ci_level}} interval from {{est.{method}.ci_lower}} to "
            f"{{est.{method}.ci_upper}}."
        )
        validity.append(
            f"The validity band for {{est.{method}.method}} is {{score.{method}.band}}, "
            f"under a ceiling of {{score.{method}.ceiling}}."
        )
    for fact in facts:
        if fact.id.startswith("cmp.") and fact.id.endswith(".agreement"):
            agreement.append(f"The comparison across methods reads {{{fact.id}}}.")
        elif fact.id.startswith("cmp.") and fact.id.endswith(".not_compared_reason"):
            agreement.append(f"No comparison was made: {{{fact.id}}}")
        elif fact.id.startswith("check.") and fact.id.endswith(".status") and fact.value in (
            "FAIL",
            "WARN",
        ):
            method, check = fact.id.split(".")[1:3]
            caveats.append(
                f"For {{est.{method}.method}}, the {{check.{method}.{check}.name}} check has "
                f"status {{{fact.id}}}."
            )
        elif fact.id.startswith("fail.") and fact.id.endswith(".error"):
            method = fact.id.split(".")[1]
            caveats.append(f"{{fail.{method}.method}} reported: {{{fact.id}}}")
    return Narrative(
        summary=[], estimates=estimates, validity=validity, agreement=agreement, caveats=caveats
    )


class ScriptedReplies:
    """Answers each call with the next item: a `Narrative`, or a stop reason (no narrative).

    An exception in the script is raised instead of answered.
    """

    def __init__(self, script: Sequence[Narrative | str | BaseException]) -> None:
        self._script = list(script)
        self.calls = 0

    def complete(self, system: str, turns: Sequence[Turn]) -> ModelReply:
        item = self._script[self.calls]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return ModelReply(None, item, "scripted-model", Usage(10, 0), None)
        return ModelReply(item, "end_turn", "scripted-model", Usage(100, 40), ("echo",))
