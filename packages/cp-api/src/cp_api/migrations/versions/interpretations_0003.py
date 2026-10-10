"""An analysis's fact table, and its interpretation."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"


def upgrade() -> None:
    # Nullable: analyses stored before this revision have no fact table, and
    # they are not refitted to get one.
    op.add_column(
        "analysis_summaries", sa.Column("facts", postgresql.JSONB(), nullable=True)
    )
    op.create_table(
        "interpretations",
        sa.Column(
            "analysis_id", sa.Uuid(), sa.ForeignKey("analyses.id"), primary_key=True
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("job_id", sa.String(64), nullable=False),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("effort", sa.String(16), nullable=False),
        sa.Column("returned_model", sa.String(64), nullable=True),
        sa.Column("prompt_version", sa.String(32), nullable=True),
        sa.Column("facts_sha256", sa.String(64), nullable=True),
        sa.Column("rendered", postgresql.JSONB(), nullable=True),
        sa.Column("notice", sa.Text(), nullable=True),
        sa.Column("attempts", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("interpretations")
    op.drop_column("analysis_summaries", "facts")
