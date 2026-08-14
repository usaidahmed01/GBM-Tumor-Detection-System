"""add persistent phase 6 segmentation result records

Revision ID: 20260814_0008
Revises: 20260814_0007
Create Date: 2026-08-15
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260814_0008"
down_revision: Union[str, Sequence[str], None] = "20260814_0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "segmentations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("analysis_run_id", sa.Uuid(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("GENERATED", name="segmentation_status", native_enum=False),
            nullable=False,
        ),
        sa.Column("model_input_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("inference_version", sa.String(length=128), nullable=False),
        sa.Column("preprocessing_version", sa.String(length=128), nullable=False),
        sa.Column("bundle_name", sa.String(length=128), nullable=False),
        sa.Column("bundle_version", sa.String(length=64), nullable=False),
        sa.Column("weights_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("device", sa.String(length=32), nullable=False),
        sa.Column("amp_enabled", sa.Boolean(), nullable=False),
        sa.Column("roi_size", sa.JSON(), nullable=False),
        sa.Column("overlap", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("spatial_shape", sa.JSON(), nullable=False),
        sa.Column("affine_ras", sa.JSON(), nullable=False),
        sa.Column("tc_storage_key", sa.String(length=500), nullable=False),
        sa.Column("tc_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("tc_size_bytes", sa.Integer(), nullable=False),
        sa.Column("wt_storage_key", sa.String(length=500), nullable=False),
        sa.Column("wt_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("wt_size_bytes", sa.Integer(), nullable=False),
        sa.Column("et_storage_key", sa.String(length=500), nullable=False),
        sa.Column("et_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("et_size_bytes", sa.Integer(), nullable=False),
        sa.Column("labelmap_storage_key", sa.String(length=500), nullable=False),
        sa.Column("labelmap_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("labelmap_size_bytes", sa.Integer(), nullable=False),
        sa.Column("voxel_counts", sa.JSON(), nullable=False),
        sa.Column("runtime_seconds", sa.Float(), nullable=True),
        sa.Column(
            "review_status",
            sa.Enum(
                "UNREVIEWED",
                "ACCEPTED",
                "EDITED",
                "REJECTED",
                name="segmentation_review_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("clinician_modified", sa.Boolean(), nullable=False),
        sa.Column("physical_volume_generated", sa.Boolean(), nullable=False),
        sa.Column("anatomical_localization_generated", sa.Boolean(), nullable=False),
        sa.Column("clinical_validation_claimed", sa.Boolean(), nullable=False),
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
        sa.ForeignKeyConstraint(["analysis_run_id"], ["analysis_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_segmentations_analysis_run_id",
        "segmentations",
        ["analysis_run_id"],
        unique=False,
    )
    op.create_index(
        "ux_segmentations_analysis_run",
        "segmentations",
        ["analysis_run_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ux_segmentations_analysis_run", table_name="segmentations")
    op.drop_index("ix_segmentations_analysis_run_id", table_name="segmentations")
    op.drop_table("segmentations")
