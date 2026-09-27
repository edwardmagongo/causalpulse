import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from cp_api.db import Base, make_engine
from cp_api.migrate import downgrade, upgrade
from sqlalchemy import inspect

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
