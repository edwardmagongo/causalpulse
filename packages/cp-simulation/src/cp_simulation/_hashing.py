"""Stable configuration hashing, shared by both data-generating processes.

sha256 over sorted JSON rather than `hash()`, which is salted per process and
would make the recorded value meaningless across runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def _describe(component: object) -> dict[str, Any]:
    """Name a component by its class and its field values."""
    if is_dataclass(component) and not isinstance(component, type):
        return {type(component).__name__: asdict(component)}
    return {type(component).__name__: repr(component)}


def component_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def describe_components(**components: object) -> dict[str, Any]:
    return {name: _describe(value) for name, value in components.items()}
