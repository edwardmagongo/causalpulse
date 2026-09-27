"""Initial tables."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("n_bytes", sa.Integer(), nullable=False),
        sa.Column("csv", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("sha256"),
    )
    op.create_table(
        "analyses",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("dataset_id", sa.Uuid(), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("schema", postgresql.JSONB(), nullable=False),
        sa.Column("requested_estimators", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("random_seed", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("partial", sa.Boolean(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "routing_decisions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("analysis_id", sa.Uuid(), sa.ForeignKey("analyses.id"), nullable=False),
        sa.Column("estimator_name", sa.String(64), nullable=False),
        sa.Column("eligible", sa.Boolean(), nullable=False),
        sa.Column("reasons", postgresql.ARRAY(sa.Text()), nullable=False),
    )
    op.create_table(
        "estimates",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("analysis_id", sa.Uuid(), sa.ForeignKey("analyses.id"), nullable=False),
        sa.Column("method", sa.String(64), nullable=False),
        sa.Column("summary", postgresql.JSONB(), nullable=True),
        sa.Column("band", sa.String(16), nullable=True),
        sa.Column("report_html", sa.Text(), nullable=True),
        sa.Column("report_markdown", sa.Text(), nullable=True),
        sa.Column("report_assets", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.UniqueConstraint("analysis_id", "method"),
    )
    op.create_table(
        "analysis_summaries",
        sa.Column("analysis_id", sa.Uuid(), sa.ForeignKey("analyses.id"), primary_key=True),
        sa.Column("manifest", postgresql.JSONB(), nullable=False),
        sa.Column("comparisons", postgresql.JSONB(), nullable=False),
    )


def downgrade() -> None:
    for table in ("analysis_summaries", "estimates", "routing_decisions", "analyses", "datasets"):
        op.drop_table(table)
