"""Stable configuration hashing, shared by both data-generating processes.

sha256 over sorted JSON rather than `hash()`, which is salted per process and
would make the recorded value meaningless across runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields, is_dataclass
from typing import Any


def _describe(component: object) -> dict[str, Any]:
    """Name a component by its class and its field values.

    A field tagged `metadata={"hash_omit_at_default": True}` is dropped from
    the description when it holds its declared default. That lets a stable
    component gain a new field -- defaulting to a no-op -- without perturbing
    the hash of every configuration that never mentions it: without the
    filter, the field's mere presence in `asdict()` output changes the
    recorded fingerprint even though nothing about the configuration changed.
    A field away from its default is never omitted, so a real configuration
    change still changes the hash.
    """
    if is_dataclass(component) and not isinstance(component, type):
        values = asdict(component)
        for component_field in fields(component):
            if (
                component_field.metadata.get("hash_omit_at_default")
                and values.get(component_field.name) == component_field.default
            ):
                del values[component_field.name]
        return {type(component).__name__: values}
    return {type(component).__name__: repr(component)}


def component_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def describe_components(**components: object) -> dict[str, Any]:
    return {name: _describe(value) for name, value in components.items()}
