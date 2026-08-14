"""add capability routing state

Revision ID: 20260814_0006
Revises: 20260814_0005
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260814_0006"
down_revision: Union[str, Sequence[str], None] = "20260814_0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "studies",
        sa.Column(
            "brain_scope_status",
            sa.Enum(
                "PENDING",
                "SUPPORTED_BY_METADATA",
                "CLINICIAN_CONFIRMED",
                "OUT_OF_SCOPE",
                name="study_brain_scope_status",
                native_enum=False,
            ),
            server_default="PENDING",
            nullable=False,
        ),
    )
    op.add_column(
        "studies",
        sa.Column(
            "nifti_sequence_mapping",
            sa.JSON(),
            server_default=sa.text("'{}'::json"),
            nullable=False,
        ),
    )
    op.add_column(
        "studies",
        sa.Column(
            "capability_routing_status",
            sa.Enum(
                "PENDING",
                "READY",
                "REVIEW_REQUIRED",
                "NO_SUPPORTED_ANALYSIS",
                name="study_capability_routing_status",
                native_enum=False,
            ),
            server_default="PENDING",
            nullable=False,
        ),
    )
    op.add_column(
        "studies",
        sa.Column(
            "capability_summary",
            sa.JSON(),
            server_default=sa.text("'{}'::json"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("studies", "capability_summary")
    op.drop_column("studies", "capability_routing_status")
    op.drop_column("studies", "nifti_sequence_mapping")
    op.drop_column("studies", "brain_scope_status")
