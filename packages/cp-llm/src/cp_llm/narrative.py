"""The shape a model's reply must take.

Five fixed sections, so that two interpretations of two analyses can be laid
side by side. Each statement is one sentence of prose in which every value is
a `{fact.id}` placeholder; the model never writes a value itself.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

SECTIONS: tuple[str, ...] = ("summary", "estimates", "validity", "agreement", "caveats")


class Narrative(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: list[str]
    estimates: list[str]
    validity: list[str]
    agreement: list[str]
    caveats: list[str]

    def statements(self) -> tuple[tuple[str, str], ...]:
        """Every `(section, statement)` pair, in reading order."""
        return tuple(
            (section, statement)
            for section in SECTIONS
            for statement in getattr(self, section)
        )
