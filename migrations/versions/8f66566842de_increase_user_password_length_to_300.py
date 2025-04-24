"""Increase User.password length to 300

Revision ID: 8f66566842de
Revises: 8fb6e61024db
Create Date: 2025-04-24 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '8f66566842de'
down_revision = '8fb6e61024db'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ユーザーパスワード長を 128 → 300 に拡張
    with op.batch_alter_table('user') as batch_op:
        batch_op.alter_column(
            'password',
            existing_type=sa.String(length=128),
            type_=sa.String(length=300),
            existing_nullable=False
        )


def downgrade() -> None:
    # 戻す：300 → 128
    with op.batch_alter_table('user') as batch_op:
        batch_op.alter_column(
            'password',
            existing_type=sa.String(length=300),
            type_=sa.String(length=128),
            existing_nullable=False
        )
