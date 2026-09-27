"""PostgreSQL- and Redis-backed fixtures. Every test here skips without them.

`make check-services` starts both and sets DATABASE_URL and REDIS_URL. No
SQLite stand-in: JSONB and bytea behave differently there, and a stand-in
would hide exactly the bugs these tests exist to find.
"""

import os

import pytest
from sqlalchemy import text

TABLES = "analysis_summaries, estimates, routing_decisions, analyses, datasets"


@pytest.fixture(scope="session")
def database_url():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL is not set; run `make check-services`")
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
    url = os.environ.get("REDIS_URL")
    if not url:
        pytest.skip("REDIS_URL is not set; run `make check-services`")
    from redis import Redis

    connection = Redis.from_url(url)
    connection.flushdb()
    yield connection
    connection.flushdb()
    connection.close()
