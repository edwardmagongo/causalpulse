"""A flat table of citable facts, and its JSON form.

The table is the only thing the interpreter sees and the only thing a
narrative may cite. This module holds the types alone and imports nothing
from the result packages, so interpretation runs from a stored `facts.json`
with no fit in memory. `cp_llm.extract` builds a table from an analysis.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, cast, get_args

FactKind = Literal["number", "count", "level", "p_value", "label", "name", "text"]

FACTS_SCHEMA_VERSION = 1
# The file `analyze --report` writes and `interpret` reads.
FACTS_FILENAME = "facts.json"

_KINDS: frozenset[str] = frozenset(get_args(FactKind))
_FLOAT_KINDS = frozenset({"number", "level", "p_value"})
_ID = re.compile(r"[a-z][a-z0-9_]*(\.[a-z0-9_]+)+")


@dataclass(frozen=True)
class Fact:
    """One citable value.

    `subject` is the id of the `name` fact that must accompany this one in a
    statement -- `est.aipw.method` for every fact about `aipw`'s estimate,
    score and checks -- or `None` for a fact about the analysis as a whole.
    """

    id: str
    kind: FactKind
    value: float | int | str
    subject: str | None

    def __post_init__(self) -> None:
        if _ID.fullmatch(self.id) is None:
            raise ValueError(f"not a fact id: {self.id!r}")
        if self.kind not in _KINDS:
            raise ValueError(f"{self.id}: unknown kind {self.kind!r}")
        value = self.value
        if self.kind == "count":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{self.id}: a count must be an int, got {value!r}")
        elif self.kind in _FLOAT_KINDS:
            if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{self.id}: a {self.kind} must be a finite float, got {value!r}")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{self.id}: a {self.kind} must be a non-empty string, got {value!r}")


@dataclass(frozen=True)
class FactTable:
    facts: tuple[Fact, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for fact in self.facts:
            if fact.id in seen:
                raise ValueError(f"duplicate fact id: {fact.id}")
            seen.add(fact.id)
        for fact in self.facts:
            if fact.subject is None:
                continue
            subject = self.get(fact.subject)
            if subject is None or subject.kind != "name":
                raise ValueError(f"{fact.id}: subject {fact.subject!r} is not a name fact")

    @cached_property
    def _by_id(self) -> Mapping[str, Fact]:
        return {fact.id: fact for fact in self.facts}

    def get(self, fact_id: str) -> Fact | None:
        return self._by_id.get(fact_id)

    def __iter__(self) -> Iterator[Fact]:
        return iter(self.facts)

    def to_records(self) -> list[dict[str, object]]:
        return [
            {"id": f.id, "kind": f.kind, "value": f.value, "subject": f.subject}
            for f in self.facts
        ]

    def to_json(self) -> str:
        payload = {"schema_version": FACTS_SCHEMA_VERSION, "facts": self.to_records()}
        return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, raw: str) -> FactTable:
        payload = json.loads(raw)
        version = payload.get("schema_version")
        if version != FACTS_SCHEMA_VERSION:
            raise ValueError(
                f"facts schema_version {version!r} is not {FACTS_SCHEMA_VERSION}; "
                "rerun `causalpulse analyze --report`"
            )
        facts = []
        for record in payload["facts"]:
            kind = cast(FactKind, record["kind"])
            value = record["value"]
            # JSON has one number type: 2.0 is written "2.0" and read back as a
            # float, but a float fact whose value is integral must not come
            # back as an int.
            if kind in _FLOAT_KINDS and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            facts.append(Fact(id=record["id"], kind=kind, value=value, subject=record["subject"]))
        return cls(tuple(facts))
