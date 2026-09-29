"""PostgreSQL- and Redis-backed fixtures. Every test here skips without them.

`make check-services` starts both against their own compose project
(`causalpulse-test`, ports 5434/6381) and sets TEST_DATABASE_URL and
TEST_REDIS_URL — deliberately not DATABASE_URL/REDIS_URL, the variables the
API and worker read. If these tests read the app's own variables, exporting
DATABASE_URL for local development would make `make test-fast` migrate
(downgrade included) and TRUNCATE whatever real database that points at, and
`make check-services` would share the demo stack's compose project and ports
and `down -v` its volumes. Separate variables and a separate compose project
make that impossible by construction, not by convention.

No SQLite stand-in: JSONB and bytea behave differently there, and a stand-in
would hide exactly the bugs these tests exist to find.
"""

import os

import pytest
from sqlalchemy import text

TABLES = "analysis_summaries, estimates, routing_decisions, analyses, datasets"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test under this directory `services`, so none can silently
    fall outside `-m services` and run against a database it shouldn't."""
    for item in items:
        if "tests/services/" in str(item.path).replace(os.sep, "/"):
            item.add_marker(pytest.mark.services)


@pytest.fixture(scope="session")
def database_url():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set; run `make check-services`")
    return url


@pytest.fixture(scope="session")
def migrated(database_url):
    from cp_api.migrate import upgrade

    upgrade(database_url)
    return database_url


@pytest.fixture
def sessions(migrated):
    from cp_api.db import make_engine, make_sessionmaker

    engine = make_engine(migrated)
    maker = make_sessionmaker(engine)
    yield maker
    with maker() as session:
        session.execute(text(f"TRUNCATE {TABLES}"))
        session.commit()
    engine.dispose()


@pytest.fixture
def redis_conn():
    url = os.environ.get("TEST_REDIS_URL")
    if not url:
        pytest.skip("TEST_REDIS_URL is not set; run `make check-services`")
    from redis import Redis

    connection = Redis.from_url(url)
    connection.flushdb()
    yield connection
    connection.flushdb()
    connection.close()


@pytest.fixture
def worker_env(monkeypatch, migrated):
    """Point the app's own DATABASE_URL/REDIS_URL at the test stack for the
    duration of one test.

    Code under test that reads the app's own variables directly — the worker
    job, `make_engine()` called with no argument — sees the test database
    only for that one test, via `monkeypatch`, never a process-wide export.
    `test_jobs.py` runs the job directly with it; `test_api.py`'s `client`
    fixture takes it so the jobs its `_drain` runs see the test stack.
    """
    monkeypatch.setenv("DATABASE_URL", migrated)
    redis_url = os.environ.get("TEST_REDIS_URL")
    if redis_url:
        monkeypatch.setenv("REDIS_URL", redis_url)
