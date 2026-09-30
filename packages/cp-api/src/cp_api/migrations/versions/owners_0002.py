"""Users, API keys, and an owner on every dataset and analysis."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"

_OWNED = ("datasets", "analyses")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(12), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("secret_sha256", sa.String(64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Every phase 1 row goes to `legacy`, which has no key until an admin
    # issues one: nothing is lost, and nothing is visible by accident.
    op.execute("INSERT INTO users (id, name) VALUES (gen_random_uuid(), 'legacy')")
    for table in _OWNED:
        op.add_column(table, sa.Column("owner_id", sa.Uuid(), nullable=True))
        op.execute(
            f"UPDATE {table} SET owner_id = (SELECT id FROM users WHERE name = 'legacy')"
        )
        op.alter_column(table, "owner_id", nullable=False)
        op.create_foreign_key(f"{table}_owner_id_fkey", table, "users", ["owner_id"], ["id"])
    op.drop_constraint("datasets_sha256_key", "datasets", type_="unique")
    op.create_unique_constraint(
        "datasets_owner_id_sha256_key", "datasets", ["owner_id", "sha256"]
    )


def downgrade() -> None:
    clashes: Sequence[str] = (
        op.get_bind()
        .execute(sa.text("SELECT sha256 FROM datasets GROUP BY sha256 HAVING count(*) > 1"))
        .scalars()
        .all()
    )
    if clashes:
        raise RuntimeError(
            f"cannot downgrade: {len(clashes)} sha256 value(s) are held by more than one "
            f"owner ({sorted(clashes)[:5]}), and the phase 1 schema allows each only once"
        )
    op.drop_constraint("datasets_owner_id_sha256_key", "datasets", type_="unique")
    op.create_unique_constraint("datasets_sha256_key", "datasets", ["sha256"])
    for table in reversed(_OWNED):
        op.drop_constraint(f"{table}_owner_id_fkey", table, type_="foreignkey")
        op.drop_column(table, "owner_id")
    op.drop_table("api_keys")
    op.drop_table("users")
