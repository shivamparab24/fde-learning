"""persons anchor table + FK cascades + lookup indexes

Revision ID: 002
Revises: 001
Create Date: 2026-08-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = "002"
down_revision: str = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "persons",
        sa.Column("person_id", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("person_id"),
    )

    # Backfill: every person already referenced anywhere becomes an anchor row
    op.execute("""
        INSERT INTO persons (person_id)
        SELECT DISTINCT person_id FROM profiles
        UNION
        SELECT DISTINCT person_id FROM chat_history
        UNION
        SELECT DISTINCT person_id FROM memories
        ON CONFLICT DO NOTHING;
    """)

    op.create_foreign_key(
        "fk_profiles_person", "profiles", "persons",
        ["person_id"], ["person_id"], ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_chat_history_person", "chat_history", "persons",
        ["person_id"], ["person_id"], ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_memories_person", "memories", "persons",
        ["person_id"], ["person_id"], ondelete="CASCADE",
    )

    # Indexes: cascaded deletes and hot read paths would seq-scan otherwise
    op.create_index(
        "ix_chat_history_person_session", "chat_history",
        ["person_id", "session_id"],
    )
    op.create_index("ix_memories_person_id", "memories", ["person_id"])


def downgrade() -> None:
    op.drop_index("ix_memories_person_id", table_name="memories")
    op.drop_index("ix_chat_history_person_session", table_name="chat_history")
    op.drop_constraint("fk_memories_person", "memories", type_="foreignkey")
    op.drop_constraint("fk_chat_history_person", "chat_history", type_="foreignkey")
    op.drop_constraint("fk_profiles_person", "profiles", type_="foreignkey")
    op.drop_table("persons")
