"""add study MRI QC state

Revision ID: 20260814_0005
Revises: 20260814_0004
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260814_0005"
down_revision: Union[str, Sequence[str], None] = "20260814_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "studies",
        sa.Column(
            "qc_status",
            sa.Enum(
                "PENDING",
                "PASS",
                "PARTIAL",
                "FAIL",
                name="study_qc_status",
                native_enum=False,
            ),
            server_default="PENDING",
            nullable=False,
        ),
    )
    op.add_column(
        "studies",
        sa.Column(
            "qc_summary",
            sa.JSON(),
            server_default=sa.text("'{}'::json"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("studies", "qc_summary")
    op.drop_column("studies", "qc_status")
