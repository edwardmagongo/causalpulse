"""The arms the study compares, and the run path each one takes.

A grounded arm is a `Variant`: phase 1's prompt with one stated change, so
its replies still carry placeholders and still go through
`cp_llm.interpret`, which verifies before it renders and refuses what will
not verify. The ungrounded arm cannot take that path -- there is no
placeholder to resolve, so there is nothing for the verifier to check and
nothing it could refuse -- and it runs `cp_llm_eval.ungrounded` instead.

`Arm` is what the runner needs in order to tell them apart: a name, the
change it makes, the prompt to send, and whether the verifier runs on what
comes back. `grounded` is the only discriminator, and `runner` is the only
module that reads it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from cp_llm_eval.ungrounded import ungrounded_prompt
from cp_llm_eval.variants import VARIANTS


class Arm(Protocol):
    """One column of the study: a prompt, and the run path its replies take."""

    @property
    def name(self) -> str: ...

    @property
    def change(self) -> str: ...

    @property
    def grounded(self) -> bool:
        """Whether a reply of this arm's is verified before it is stored."""
        ...

    def system(self) -> str: ...


@dataclass(frozen=True)
class UngroundedArm:
    """The arm that states its own values, and so has no verifier.

    Deliberately not a `Variant`. A variant is the baseline prompt plus one
    sentence and keeps everything the verifier needs; this arm takes three
    rules and the placeholder schema out, and its replies are stored
    unverified. Giving it a `Variant`'s type would let it reach
    `interpret()`, which would refuse every reply it ever made.
    """

    name: str
    change: str

    @property
    def grounded(self) -> bool:
        return False

    def system(self) -> str:
        return ungrounded_prompt()


UNGROUNDED = UngroundedArm(
    "ungrounded",
    "rules 1 to 3 and the placeholder schema are replaced by: state the values",
)

# Every arm of the study, grounded ones first, derived from `VARIANTS` so that
# a prompt added there is an arm here without being named twice.
ARMS: Mapping[str, Arm] = MappingProxyType({**VARIANTS, UNGROUNDED.name: UNGROUNDED})
