"""Fixtures shared across the whole test suite.

The registry is process-wide module state. Without isolation, a dummy estimator
registered in one test remains visible to every later test in the session,
corrupting test isolation.

The snapshot/restore logic lives in `isolated_registry` rather than only inside
the fixture so it can be tested directly, in one self-contained test, instead of
relying on two top-level tests running in file order — an order-dependent pair
passes vacuously when run alone or reordered.

Restoring in place (`clear()` + `update()`) rather than rebinding
`registry._REGISTRY` is deliberate: rebinding would leave any module that did
`from cp_causal.registry import _REGISTRY` pointing at the stale dict. Nothing
imports it that way today, so the two are currently equivalent — this is the
form that stays correct if one ever does.

Defined here, at the top of `tests/`, rather than in `tests/causal/conftest.py`,
so that every directory under `tests/` — not just `tests/causal/` — gets the
autouse isolation. `tests/evaluation/` in particular registers seeded-defect
estimators for the benchmark-instrument tests and needs the same cleanup.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from cp_causal import registry


@contextmanager
def isolated_registry() -> Iterator[None]:
    """Restore the registry's contents on exit."""
    saved = dict(registry._REGISTRY)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    with isolated_registry():
        yield
