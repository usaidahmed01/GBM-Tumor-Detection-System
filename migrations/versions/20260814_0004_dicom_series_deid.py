"""add DICOM deidentified working copy and series table

Revision ID: 20260814_0004
Revises: 20260814_0003
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260814_0004"
down_revision: Union[str, Sequence[str], None] = "20260814_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "studies",
        sa.Column(
            "deidentified_storage_key",
            sa.String(length=500),
            nullable=True,
        ),
    )
    op.add_column(
        "studies",
        sa.Column(
            "deidentified_checksum_sha256",
            sa.String(length=64),
            nullable=True,
        ),
    )
    op.add_column(
        "studies",
        sa.Column(
            "deidentification_status",
            sa.Enum(
                "PENDING",
                "NOT_APPLICABLE",
                "METADATA_DEIDENTIFIED",
                "BLOCKED_PIXEL_PHI_RISK",
                "FAILED",
                name="study_deidentification_status",
                native_enum=False,
            ),
            server_default="PENDING",
            nullable=False,
        ),
    )

    op.create_table(
        "series",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("series_uid", sa.String(length=128), nullable=False),
        sa.Column("series_number", sa.Integer(), nullable=True),
        sa.Column("detected_sequence", sa.String(length=32), nullable=True),
        sa.Column("confirmed_sequence", sa.String(length=32), nullable=True),
        sa.Column("sequence_confidence", sa.Float(), nullable=True),
        sa.Column("sequence_metadata", sa.JSON(), nullable=False),
        sa.Column("slice_count", sa.Integer(), nullable=False),
        sa.Column(
            "spacing_orientation_metadata",
            sa.JSON(),
            nullable=False,
        ),
        sa.Column(
            "working_member_prefix",
            sa.String(length=200),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["study_id"],
            ["studies.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_series_study_id",
        "series",
        ["study_id"],
        unique=False,
    )
    op.create_index(
        "ix_series_study_created",
        "series",
        ["study_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ux_series_study_uid",
        "series",
        ["study_id", "series_uid"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ux_series_study_uid", table_name="series")
    op.drop_index("ix_series_study_created", table_name="series")
    op.drop_index("ix_series_study_id", table_name="series")
    op.drop_table("series")

    op.drop_column("studies", "deidentification_status")
    op.drop_column("studies", "deidentified_checksum_sha256")
    op.drop_column("studies", "deidentified_storage_key")
