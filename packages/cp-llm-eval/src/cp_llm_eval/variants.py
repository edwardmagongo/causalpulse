"""The system prompts the study compares.

The baseline is phase 1's prompt, unchanged. A variant is the baseline with
one stated change, built from the baseline each time, so a change to phase
1's prompt reaches every variant, and a variant whose anchor text has gone is
refused rather than silently sent without its change.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from cp_llm.prompt import system_prompt


@dataclass(frozen=True)
class Variant:
    name: str
    change: str
    _build: Callable[[], str]

    def system(self) -> str:
        return self._build()


def _with_sentence_after(anchor: str, sentence: str) -> Callable[[], str]:
    def build() -> str:
        base = system_prompt()
        if base.count(anchor) != 1:
            raise ValueError(f"the base prompt no longer contains {anchor!r} exactly once")
        return base.replace(anchor, f"{anchor} {sentence}")

    return build


# Rule 8's last sentence. The variant's sentence joins rule 8, as the design
# states (§6.3).
_RULE_8_END = "unless the same sentence states the assumption the claim rests on."
CITE_EACH_STATUS_SENTENCE = "Do not summarise check statuses; cite each status you mention."

BASELINE = Variant("baseline", "phase 1's prompt, unchanged", system_prompt)
CITE_EACH_STATUS = Variant(
    "cite-each-status",
    f'rule 8 gains: "{CITE_EACH_STATUS_SENTENCE}"',
    _with_sentence_after(_RULE_8_END, CITE_EACH_STATUS_SENTENCE),
)

VARIANTS: Mapping[str, Variant] = MappingProxyType(
    {variant.name: variant for variant in (BASELINE, CITE_EACH_STATUS)}
)
