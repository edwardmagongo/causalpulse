import hashlib
import uuid

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from cp_api.db import Analysis, Base, make_engine
from cp_api.migrate import downgrade, upgrade
from sqlalchemy import inspect, text

pytestmark = pytest.mark.services


def test_the_migration_builds_exactly_the_models_tables(migrated):
    engine = make_engine(migrated)
    try:
        with engine.connect() as connection:
            diffs = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    finally:
        engine.dispose()

    assert diffs == []


def test_downgrade_removes_every_table_and_upgrade_restores_them(migrated):
    engine = make_engine(migrated)
    try:
        try:
            downgrade(migrated)
            assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
        finally:
            # Always restore, in the `finally`, so a failed assertion above
            # can't leave the database at base for later tests.
            upgrade(migrated)
        assert set(inspect(engine).get_table_names()) >= set(Base.metadata.tables)
    finally:
        engine.dispose()


# The newest revision: what `upgrade` leaves the database at.
HEAD = "0003"

ALL_TABLES = (
    "interpretations, analysis_summaries, estimates, routing_decisions, analyses, datasets, "
    "api_keys, users"
)


def _truncate(engine):
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {ALL_TABLES}"))


def test_upgrading_a_phase_1_database_gives_every_row_to_legacy(migrated):
    engine = make_engine(migrated)
    try:
        _truncate(engine)
        downgrade(migrated, "0001")
        dataset_id, analysis_id = uuid.uuid4(), uuid.uuid4()
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO datasets (id, filename, sha256, n_bytes, csv) "
                    "VALUES (:id, 'd.csv', :sha, 3, :csv)"
                ),
                {"id": dataset_id, "sha": hashlib.sha256(b"a,b").hexdigest(), "csv": b"a,b"},
            )
            connection.execute(
                text(
                    "INSERT INTO analyses (id, dataset_id, schema, random_seed, status, partial) "
                    "VALUES (:id, :ds, '{}'::jsonb, 0, 'succeeded', false)"
                ),
                {"id": analysis_id, "ds": dataset_id},
            )

        upgrade(migrated)

        with engine.connect() as connection:
            for table in ("datasets", "analyses"):
                owners = connection.execute(
                    text(f"SELECT u.name FROM {table} t JOIN users u ON u.id = t.owner_id")
                ).scalars().all()
                assert owners == ["legacy"], table
            keys = connection.execute(text("SELECT count(*) FROM api_keys")).scalar_one()
            assert keys == 0  # legacy has no key until an admin issues one
    finally:
        upgrade(migrated)
        _truncate(engine)
        engine.dispose()


def test_downgrade_refuses_when_two_owners_hold_the_same_bytes(migrated):
    from cp_api.accounts import create_user
    from cp_api.db import Dataset, make_sessionmaker

    engine = make_engine(migrated)
    sessions = make_sessionmaker(engine)
    try:
        _truncate(engine)
        sha = hashlib.sha256(b"a,b").hexdigest()
        with sessions() as session:
            for name in ("alice", "bob"):
                user = create_user(session, name)
                session.add(
                    Dataset(owner_id=user.id, filename="d.csv", sha256=sha, n_bytes=3, csv=b"a,b")
                )
            session.commit()

        with pytest.raises(RuntimeError, match="more than one owner"):
            downgrade(migrated, "0001")

        # The refusal rolled the whole downgrade back, so the schema is still at the head.
        with engine.connect() as connection:
            version = connection.execute(text("SELECT version_num FROM alembic_version"))
            assert version.scalar_one() == HEAD
    finally:
        upgrade(migrated)
        _truncate(engine)
        engine.dispose()


def test_downgrade_succeeds_when_no_two_owners_share_bytes_and_upgrade_restores_them(migrated):
    from cp_api.accounts import create_user
    from cp_api.db import Dataset, make_sessionmaker

    engine = make_engine(migrated)
    sessions = make_sessionmaker(engine)
    try:
        _truncate(engine)
        with sessions() as session:
            alice = create_user(session, "alice")
            bob = create_user(session, "bob")
            alices = Dataset(
                owner_id=alice.id, filename="d.csv", n_bytes=3, csv=b"a,b",
                sha256=hashlib.sha256(b"a,b").hexdigest(),
            )
            bobs = Dataset(
                owner_id=bob.id, filename="d.csv", n_bytes=3, csv=b"c,d",
                sha256=hashlib.sha256(b"c,d").hexdigest(),
            )
            session.add_all([alices, bobs])
            session.flush()
            session.add(
                Analysis(
                    owner_id=alice.id, dataset_id=alices.id, causal_schema={},
                    random_seed=0, status="succeeded",
                )
            )
            session.commit()

        downgrade(migrated, "0001")

        with engine.connect() as connection:
            assert connection.execute(text("SELECT count(*) FROM datasets")).scalar_one() == 2
            assert connection.execute(text("SELECT count(*) FROM analyses")).scalar_one() == 1
            for table in ("datasets", "analyses"):
                columns = {c["name"] for c in inspect(connection).get_columns(table)}
                assert "owner_id" not in columns, table
            version = connection.execute(text("SELECT version_num FROM alembic_version"))
            assert version.scalar_one() == "0001"

        upgrade(migrated)

        with engine.connect() as connection:
            for table, expected in (("datasets", 2), ("analyses", 1)):
                owners = connection.execute(
                    text(f"SELECT u.name FROM {table} t JOIN users u ON u.id = t.owner_id")
                ).scalars().all()
                assert owners == ["legacy"] * expected, table
    finally:
        upgrade(migrated)
        _truncate(engine)
        engine.dispose()


def test_the_head_is_the_newest_revision(migrated):
    engine = make_engine(migrated)
    try:
        with engine.connect() as connection:
            version = connection.execute(text("SELECT version_num FROM alembic_version"))
            assert version.scalar_one() == HEAD
    finally:
        engine.dispose()


def test_downgrading_to_0002_drops_the_facts_and_the_interpretations(migrated):
    engine = make_engine(migrated)
    try:
        _truncate(engine)
        downgrade(migrated, "0002")
        with engine.connect() as connection:
            assert "interpretations" not in inspect(connection).get_table_names()
            columns = {c["name"] for c in inspect(connection).get_columns("analysis_summaries")}
            assert "facts" not in columns
    finally:
        upgrade(migrated)
        engine.dispose()
