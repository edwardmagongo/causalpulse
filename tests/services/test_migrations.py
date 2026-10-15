import hashlib
import json
import uuid

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from cp_api.db import Analysis, Base, make_engine, utcnow
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
HEAD = "0004"

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


# Every column `interpretations` had at 0003, in table order.
COLUMNS_0003 = (
    "analysis_id, status, job_id, model, effort, returned_model, prompt_version, facts_sha256, "
    "rendered, notice, attempts, error, created_at, started_at, finished_at"
)


def _analyses(sessions, n):
    """`n` succeeded analyses of one dataset owned by a new user; their ids.

    Only `interpretations` differs between 0003 and the head, so these rows
    can be written through the models at either revision.
    """
    from cp_api.accounts import create_user
    from cp_api.db import Dataset

    with sessions() as session:
        owner = create_user(session, f"owner-{uuid.uuid4().hex[:8]}")
        dataset = Dataset(
            owner_id=owner.id, filename="d.csv", sha256=hashlib.sha256(b"a,b").hexdigest(),
            n_bytes=3, csv=b"a,b",
        )
        session.add(dataset)
        session.flush()
        analyses = [
            Analysis(
                owner_id=owner.id, dataset_id=dataset.id, causal_schema={},
                random_seed=0, status="succeeded",
            )
            for _ in range(n)
        ]
        session.add_all(analyses)
        session.commit()
        return [a.id for a in analyses]


def _insert_interpretation(connection, analysis_id, status, **extra):
    """One row by SQL, naming only the columns 0003 had, so it runs at either revision."""
    fields = {
        "analysis_id": analysis_id, "status": status, "job_id": f"interpretation-{status}",
        "model": "claude-opus-5-5", "effort": "medium", **extra,
    }
    names = ", ".join(fields)
    values = ", ".join(
        f"CAST(:{name} AS jsonb)" if name in ("rendered", "attempts") else f":{name}"
        for name in fields
    )
    connection.execute(text(f"INSERT INTO interpretations ({names}) VALUES ({values})"), fields)


def _version(connection):
    return connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()


def test_upgrading_from_0003_keeps_every_interpretation_and_gives_each_an_id(migrated):
    from cp_api.db import INTERPRETATION_STATUSES, make_sessionmaker

    engine = make_engine(migrated)
    try:
        _truncate(engine)
        downgrade(migrated, "0003")
        analysis_ids = _analyses(make_sessionmaker(engine), len(INTERPRETATION_STATUSES))
        with engine.begin() as connection:
            for analysis_id, status in zip(analysis_ids, INTERPRETATION_STATUSES, strict=True):
                _insert_interpretation(
                    connection, analysis_id, status,
                    returned_model="claude-opus-5-5-20261001", prompt_version="5b08e57994d4",
                    facts_sha256="f" * 64, notice=f"notice {status}", error=f"error {status}",
                    rendered=json.dumps({"sections": [], "limits": status}),
                    attempts=json.dumps([{"status": status}]),
                    started_at=utcnow(), finished_at=utcnow(),
                )
        select_old = f"SELECT {COLUMNS_0003} FROM interpretations ORDER BY analysis_id"
        with engine.connect() as connection:
            before = [dict(r) for r in connection.execute(text(select_old)).mappings()]

        upgrade(migrated)

        with engine.connect() as connection:
            assert _version(connection) == HEAD
            after = [dict(r) for r in connection.execute(text(select_old)).mappings()]
            added = connection.execute(
                text("SELECT id, interpreter_version FROM interpretations")
            ).all()
        assert sorted(row["status"] for row in before) == sorted(INTERPRETATION_STATUSES)
        assert after == before
        ids = [row.id for row in added]
        assert len(ids) == len(before) and None not in ids and len(set(ids)) == len(ids)
        assert [row.interpreter_version for row in added] == [None] * len(before)
    finally:
        upgrade(migrated)
        _truncate(engine)
        engine.dispose()


def test_after_the_upgrade_one_analysis_can_hold_two_interpretations(migrated):
    from cp_api.db import Interpretation, make_sessionmaker

    engine = make_engine(migrated)
    sessions = make_sessionmaker(engine)
    try:
        _truncate(engine)
        (analysis_id,) = _analyses(sessions, 1)
        with sessions() as session:
            for status in ("failed", "verified"):
                session.add(
                    Interpretation(
                        analysis_id=analysis_id, status=status, job_id=f"interpretation-{status}",
                        model="claude-opus-5-5", effort="medium",
                        interpreter_version="35d8275efecb",
                    )
                )
            session.commit()

        with engine.connect() as connection:
            rows = connection.execute(
                text("SELECT id, status, interpreter_version FROM interpretations "
                     "WHERE analysis_id = :a ORDER BY status"),
                {"a": analysis_id},
            ).all()
            primary_key = inspect(connection).get_pk_constraint("interpretations")
            indexes = inspect(connection).get_indexes("interpretations")
        assert [(r.status, r.interpreter_version) for r in rows] == [
            ("failed", "35d8275efecb"), ("verified", "35d8275efecb"),
        ]
        assert rows[0].id != rows[1].id
        assert (primary_key["name"], primary_key["constrained_columns"]) == (
            "interpretations_pkey", ["id"],
        )
        assert [(i["name"], i["column_names"], i["unique"]) for i in indexes] == [
            ("ix_interpretations_analysis_id", ["analysis_id"], False),
        ]
    finally:
        _truncate(engine)
        engine.dispose()


def test_downgrading_to_0003_restores_the_analysis_key_when_each_has_one_interpretation(migrated):
    from cp_api.db import make_sessionmaker

    engine = make_engine(migrated)
    try:
        _truncate(engine)
        analysis_ids = _analyses(make_sessionmaker(engine), 2)
        with engine.begin() as connection:
            for analysis_id, status in zip(analysis_ids, ("verified", "failed"), strict=True):
                _insert_interpretation(connection, analysis_id, status)
        select_old = f"SELECT {COLUMNS_0003} FROM interpretations ORDER BY analysis_id"
        with engine.connect() as connection:
            before = [dict(r) for r in connection.execute(text(select_old)).mappings()]

        downgrade(migrated, "0003")

        with engine.connect() as connection:
            assert _version(connection) == "0003"
            after = [dict(r) for r in connection.execute(text(select_old)).mappings()]
            columns = {c["name"] for c in inspect(connection).get_columns("interpretations")}
            primary_key = inspect(connection).get_pk_constraint("interpretations")
            indexes = inspect(connection).get_indexes("interpretations")
        assert after == before
        assert columns == {name.strip() for name in COLUMNS_0003.split(",")}
        assert (primary_key["name"], primary_key["constrained_columns"]) == (
            "interpretations_pkey", ["analysis_id"],
        )
        assert indexes == []
    finally:
        upgrade(migrated)
        _truncate(engine)
        engine.dispose()


def test_downgrading_to_0003_refuses_when_an_analysis_has_two_interpretations(migrated):
    from cp_api.db import make_sessionmaker

    engine = make_engine(migrated)
    try:
        _truncate(engine)
        (analysis_id,) = _analyses(make_sessionmaker(engine), 1)
        with engine.begin() as connection:
            _insert_interpretation(connection, analysis_id, "failed")
            _insert_interpretation(connection, analysis_id, "verified")
        select_all = "SELECT * FROM interpretations ORDER BY status"
        with engine.connect() as connection:
            before = [dict(r) for r in connection.execute(text(select_all)).mappings()]

        with pytest.raises(RuntimeError, match="the earlier ones would be lost"):
            downgrade(migrated, "0003")

        with engine.connect() as connection:
            assert _version(connection) == HEAD
            after = [dict(r) for r in connection.execute(text(select_all)).mappings()]
            primary_key = inspect(connection).get_pk_constraint("interpretations")
        assert len(before) == 2
        assert after == before
        assert (primary_key["name"], primary_key["constrained_columns"]) == (
            "interpretations_pkey", ["id"],
        )
    finally:
        upgrade(migrated)
        _truncate(engine)
        engine.dispose()
