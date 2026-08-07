"""Fixtures for tests in this directory.

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
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

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


@pytest.fixture
def registry_isolation() -> Callable[[], AbstractContextManager[None]]:
    """Hands `isolated_registry` to a test that wants to assert on it directly.

    Exposed as a fixture rather than imported across modules because `tests/` is
    not a package, so `from tests.causal.conftest import ...` does not resolve.
    """
    return isolated_registry
