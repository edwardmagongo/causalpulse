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

`--shard I/N` keeps the tests of one shard of the test files, so that CI can
run the slow tier as N parallel jobs. It splits by file, not by test, so a
file's tests and any module-scoped fixture they share stay in one job. The
files are the ones still selected after `-m` and `-k` (the hook runs last for
that reason), taken in sorted order of their path and dealt round-robin.
Dealing the whole suite's files instead would leave the handful of slow files
spread unevenly, and some shards empty. Sorted order, and
nothing else, decides the split: under `pytest-xdist` every worker collects for
itself and all of them must arrive at the same set, so the split cannot depend
on the worker, on collection order or on a salted hash.
"""

import argparse
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


@pytest.fixture(autouse=True)
def _no_ambient_server_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's shell must not change how `create_app()` behaves in a test."""
    for name in (
        "CP_INTERPRETATION",
        "CP_LIMIT_SIGNIN_FAILURES",
        "CP_LIMIT_UPLOADS_PER_DAY",
        "CP_LIMIT_ANALYSES_PER_DAY",
        "CP_LIMIT_ANALYSES_ACTIVE",
        "CP_LIMIT_INTERPRETATIONS_PER_DAY",
    ):
        monkeypatch.delenv(name, raising=False)


def _shard(value: str) -> tuple[int, int]:
    """Parse `I/N` into (I, N), 1-based; anything else is a usage error."""
    index, slash, count = value.partition("/")
    if not (slash and index.isdecimal() and count.isdecimal()):
        raise argparse.ArgumentTypeError(f"{value!r} is not I/N, such as 3/8")
    if not 1 <= int(index) <= int(count):
        raise argparse.ArgumentTypeError(f"{value!r} is not a shard: I/N needs 1 <= I <= N")
    return int(index), int(count)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        type=_shard,
        default=None,
        metavar="I/N",
        help="run only shard I of N (1-based) of the selected test files",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    shard = config.getoption("--shard")
    if shard is None:
        return
    index, count = shard
    files = sorted({item.nodeid.split("::")[0] for item in items})
    mine = {file for k, file in enumerate(files) if k % count + 1 == index}
    kept = [item for item in items if item.nodeid.split("::")[0] in mine]
    dropped = [item for item in items if item.nodeid.split("::")[0] not in mine]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
