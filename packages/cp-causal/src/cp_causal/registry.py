"""Estimator registry.

Registration by decorator so that adding a method is one class and one line,
and so the router, comparison layer, and benchmark harness all discover new
estimators without being edited.
"""

from __future__ import annotations

from cp_causal.base import Requirements

_REGISTRY: dict[str, type] = {}


def register[T: type](cls: T) -> T:
    """Register an estimator class under its declared `name`."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"{cls.__name__} must declare a non-empty string `name` to be registered"
        )
    if name in _REGISTRY:
        raise ValueError(
            f"estimator name {name!r} is already registered to "
            f"{_REGISTRY[name].__name__}"
        )
    # Validated here rather than left to fail inside the router: without this, a
    # class missing `requirements` registers cleanly and then dies in
    # `route()` at `pool[name].requirements.unmet(...)`, far from the mistake.
    if not isinstance(getattr(cls, "requirements", None), Requirements):
        raise ValueError(
            f"{cls.__name__} must declare a `requirements` of type Requirements "
            f"to be registered"
        )
    _REGISTRY[name] = cls
    return cls


def get_estimator(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"unknown estimator {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_estimators() -> dict[str, type]:
    """A copy of the registry, so callers cannot corrupt it process-wide."""
    return dict(_REGISTRY)
