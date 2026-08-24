"""Episodic summary blocks + memory TTL/confidence safeguards

Revision ID: 003
Revises: 002
Create Date: 2026-08-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    op.create_table(
        "session_summaries",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "person_id",
            sa.Text(),
            sa.ForeignKey("persons.person_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("window_index", sa.Integer(), nullable=False),
        sa.Column("start_seq", sa.BigInteger(), nullable=True),
        sa.Column("end_seq", sa.BigInteger(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "person_id", "session_id", "window_index", name="uq_session_summary_window"
        ),
    )
    op.create_index(
        "ix_session_summaries_lookup",
        "session_summaries",
        ["person_id", "session_id", "window_index"],
    )

    op.add_column(
        "memories",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "memories",
        sa.Column(
            "confidence",
            sa.Numeric(3, 2),
            server_default=sa.text("1.0"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_memories_expires_at",
        "memories",
        ["expires_at"],
        postgresql_where=sa.text("expires_at IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_memories_expires_at", table_name="memories")
    op.drop_column("memories", "confidence")
    op.drop_column("memories", "expires_at")
    op.drop_index("ix_session_summaries_lookup", table_name="session_summaries")
    op.drop_table("session_summaries")
