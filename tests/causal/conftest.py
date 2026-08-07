"""Fixtures for tests in this directory.

The registry is process-wide module state. Without this fixture, a dummy
estimator registered in one test remains visible to every later test in the
session, corrupting test isolation. This autouse fixture snapshots and restores
the registry around every test to prevent leakage.
"""

import pytest
from cp_causal import registry


@pytest.fixture(autouse=True)
def _isolate_registry():
    saved = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)
