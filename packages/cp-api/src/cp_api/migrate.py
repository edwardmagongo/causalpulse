"""Alembic, driven from code rather than an alembic.ini."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from cp_api.db import database_url
from cp_api.settings import require_production_settings


def alembic_config(url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    # ConfigParser interpolation treats `%` specially.
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def upgrade(url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str, revision: str = "base") -> None:
    command.downgrade(alembic_config(url), revision)


def main() -> None:
    require_production_settings()
    upgrade(database_url())
