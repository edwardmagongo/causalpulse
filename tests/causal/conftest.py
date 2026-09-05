"""Fixtures for tests in this directory.

`isolated_registry` and its autouse `_isolate_registry` fixture live in
`tests/conftest.py` now, so every directory under `tests/` gets registry
isolation, not just this one. `registry_isolation` stays here because nothing
outside `tests/causal/` currently wants to assert on the context manager
directly.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest
from conftest import isolated_registry


@pytest.fixture
def registry_isolation() -> Callable[[], AbstractContextManager[None]]:
    """Hands `isolated_registry` to a test that wants to assert on it directly.

    Exposed as a fixture rather than imported across modules because `tests/` is
    not a package, so `from tests.causal.conftest import ...` does not resolve.
    """
    return isolated_registry
