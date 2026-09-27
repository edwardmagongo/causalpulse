"""Alembic, driven from code rather than an alembic.ini."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from cp_api.db import database_url


def alembic_config(url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    # ConfigParser interpolation treats `%` specially.
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def upgrade(url: str) -> None:
    command.upgrade(alembic_config(url), "head")


def downgrade(url: str) -> None:
    command.downgrade(alembic_config(url), "base")


def main() -> None:
    upgrade(database_url())
