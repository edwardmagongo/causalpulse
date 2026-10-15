"""One row per interpretation, so an analysis keeps every interpretation it is given."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"

_TABLE = "interpretations"
_PRIMARY_KEY = "interpretations_pkey"
_INDEX = "ix_interpretations_analysis_id"


def upgrade() -> None:
    # The server default fills every existing row, so the column can be NOT
    # NULL from the start.
    op.add_column(
        _TABLE,
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
    )
    op.drop_constraint(_PRIMARY_KEY, _TABLE, type_="primary")
    op.create_primary_key(_PRIMARY_KEY, _TABLE, ["id"])
    # Stated, whatever dropping the old primary key did to the column.
    op.alter_column(_TABLE, "analysis_id", existing_type=sa.Uuid(), nullable=False)
    op.create_index(_INDEX, _TABLE, ["analysis_id"])
    # Null on every existing row: what they were written under was not recorded.
    op.add_column(_TABLE, sa.Column("interpreter_version", sa.String(32), nullable=True))


def downgrade() -> None:
    # Checked before anything changes: the 0003 key holds one row per
    # analysis, so going back would have to delete the others.
    clash = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT analysis_id FROM interpretations "
                "GROUP BY analysis_id HAVING count(*) > 1 LIMIT 1"
            )
        )
        .first()
    )
    if clash is not None:
        raise RuntimeError(
            "cannot downgrade: some analyses have more than one interpretation, "
            "and the earlier ones would be lost"
        )
    op.drop_column(_TABLE, "interpreter_version")
    op.drop_index(_INDEX, table_name=_TABLE)
    op.drop_constraint(_PRIMARY_KEY, _TABLE, type_="primary")
    op.drop_column(_TABLE, "id")
    op.create_primary_key(_PRIMARY_KEY, _TABLE, ["analysis_id"])
