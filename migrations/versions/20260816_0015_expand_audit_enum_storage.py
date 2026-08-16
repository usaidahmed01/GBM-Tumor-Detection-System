"""expand audit enum storage for current action/entity labels

Revision ID: 20260816_0015
Revises: 20260816_0014
Create Date: 2026-08-16

The original audit table used VARCHAR widths inferred from the enum labels that
existed in Phase 4. Later phases added longer labels such as
SEGMENTATION_PREPARATION_COMPLETED (34 chars) and SEGMENTATION_JOB (16 chars).
PostgreSQL therefore rejected valid current audit events before the surrounding
segmentation transaction could commit.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260816_0015"
down_revision: Union[str, Sequence[str], None] = "20260816_0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "audit_logs",
        "action",
        existing_type=sa.String(length=24),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        "audit_logs",
        "entity_type",
        existing_type=sa.String(length=14),
        type_=sa.String(length=64),
        existing_nullable=False,
    )


def downgrade() -> None:
    # Downgrade is only safe when no newer long enum labels have been stored.
    # Alembic/PostgreSQL will reject the narrowing operation rather than
    # silently truncate audit data.
    op.alter_column(
        "audit_logs",
        "entity_type",
        existing_type=sa.String(length=64),
        type_=sa.String(length=14),
        existing_nullable=False,
    )
    op.alter_column(
        "audit_logs",
        "action",
        existing_type=sa.String(length=64),
        type_=sa.String(length=24),
        existing_nullable=False,
    )
