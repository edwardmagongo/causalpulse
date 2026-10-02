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

`_no_real_model_call` is a net under every test of the model layer. Each of
those tests replaces the model by hand, but nothing else stops a future one
building a real `anthropic.Anthropic()` and spending money. Unless
`CP_LLM_LIVE=1` (set only by `make check-llm`), the key and token variables
are removed and the base URL is the discard port, where nothing listens: a
client built by mistake finds no credentials, and one given a key finds no
server. It sets environment variables only. Importing the SDK here would load
it in every test session.
"""

import os
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


@pytest.fixture(autouse=True)
def _no_real_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("CP_LLM_LIVE") == "1":
        return
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
