"""A narrative that verifies against the `facts` fixture, and ways to break it."""

from __future__ import annotations

from collections.abc import Sequence

from cp_llm.model import ModelReply, Turn, Usage
from cp_llm.narrative import Narrative


def _estimate(method: str) -> str:
    return (
        f"Under its own assumptions, {{est.{method}.method}} estimates the "
        f"{{est.{method}.estimand}} at {{est.{method}.point}}, with a "
        f"{{est.{method}.ci_level}} interval from {{est.{method}.ci_lower}} to "
        f"{{est.{method}.ci_upper}} that {{est.{method}.interval_excludes_zero}}."
    )


def _band(method: str) -> str:
    return (
        f"The validity band for {{est.{method}.method}} is {{score.{method}.band}}, "
        f"under a ceiling of {{score.{method}.ceiling}}."
    )


def good_narrative() -> Narrative:
    """Cites everything completeness requires of the `facts` fixture, and nothing else."""
    return Narrative(
        summary=[
            "The comparison across methods reads {cmp.ate.percentage_point.agreement}: "
            "{cmp.ate.percentage_point.verdict}"
        ],
        estimates=[_estimate("difference_in_means"), _estimate("ipw")],
        validity=[_band("difference_in_means"), _band("ipw")],
        agreement=[],
        caveats=[
            "For {est.ipw.method}, the {check.ipw.positivity.name} check has status "
            "{check.ipw.positivity.status}: {check.ipw.positivity.interpretation}"
        ],
    )


def swap(narrative: Narrative, section: str, index: int, statement: str) -> Narrative:
    data = narrative.model_dump()
    data[section][index] = statement
    return Narrative(**data)


def drop(narrative: Narrative, section: str, index: int) -> Narrative:
    data = narrative.model_dump()
    del data[section][index]
    return Narrative(**data)


class ScriptedModel:
    """Replays a fixed list of replies and records every call it was given.

    A `Narrative` in the script is a finished reply; a string is a stop reason
    for a reply that carries no narrative.
    """

    def __init__(self, script: Sequence[Narrative | str]) -> None:
        self._script = list(script)
        self.calls: list[tuple[str, list[Turn]]] = []

    def complete(self, system: str, turns: Sequence[Turn]) -> ModelReply:
        self.calls.append((system, list(turns)))
        item = self._script.pop(0)
        if isinstance(item, str):
            return ModelReply(None, item, "scripted-model", Usage(3, 0), None)
        return ModelReply(
            item, "end_turn", "scripted-model", Usage(100, 40), ("echo", len(self.calls))
        )
