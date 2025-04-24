"""Make article timestamps timezone aware

Revision ID: 378cfd606144
Revises: 8f66566842de
Create Date: 2025-04-XX 12:34:56.789012

"""

# revision identifiers, used by Alembic.
revision = '378cfd606144'
down_revision = '8f66566842de'
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    # scheduled_at カラムを timezone-aware に変更
    with op.batch_alter_table("article") as batch_op:
        batch_op.alter_column(
            "scheduled_at",
            existing_type=sa.DateTime(),
            type_=sa.DateTime(timezone=True),
            existing_nullable=True
        )


def downgrade() -> None:
    # 戻す：timezone-aware → 通常の DateTime
    with op.batch_alter_table("article") as batch_op:
        batch_op.alter_column(
            "scheduled_at",
            existing_type=sa.DateTime(timezone=True),
            type_=sa.DateTime(),
            existing_nullable=True
        )
