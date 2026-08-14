"""add phase 6 segmentation preparation state

Revision ID: 20260814_0007
Revises: 20260814_0006
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260814_0007"
down_revision: Union[str, Sequence[str], None] = "20260814_0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "studies",
        sa.Column(
            "segmentation_preparation_status",
            sa.Enum(
                "PENDING",
                "READY",
                "REGISTRATION_REQUIRED",
                "FAILED",
                name="study_segmentation_preparation_status",
                native_enum=False,
            ),
            server_default="PENDING",
            nullable=False,
        ),
    )
    op.add_column(
        "studies",
        sa.Column(
            "segmentation_preparation_summary",
            sa.JSON(),
            server_default=sa.text("'{}'::json"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("studies", "segmentation_preparation_summary")
    op.drop_column("studies", "segmentation_preparation_status")
