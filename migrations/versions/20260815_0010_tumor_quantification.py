"""add phase 7 physical tumor quantification records

Revision ID: 20260815_0010
Revises: 20260814_0009
Create Date: 2026-08-15
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260815_0010"
down_revision: Union[str, Sequence[str], None] = "20260814_0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tumor_quantifications",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("segmentation_id", sa.Uuid(), nullable=False),
        sa.Column("quantification_version", sa.String(length=128), nullable=False),
        sa.Column("source_fingerprint_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_review_status", sa.String(length=32), nullable=False),
        sa.Column("source_clinician_modified", sa.Boolean(), nullable=False),
        sa.Column("source_mask_checksums", sa.JSON(), nullable=False),
        sa.Column("spatial_shape", sa.JSON(), nullable=False),
        sa.Column("affine_ras", sa.JSON(), nullable=False),
        sa.Column("voxel_spacing_mm", sa.JSON(), nullable=False),
        sa.Column("voxel_volume_mm3", sa.Float(), nullable=False),
        sa.Column("axial_pixel_area_mm2", sa.Float(), nullable=False),
        sa.Column("wt_voxel_count", sa.Integer(), nullable=False),
        sa.Column("wt_volume_mm3", sa.Float(), nullable=False),
        sa.Column("wt_volume_cm3", sa.Float(), nullable=False),
        sa.Column("wt_max_axial_area_mm2", sa.Float(), nullable=False),
        sa.Column("wt_max_axial_slice_index", sa.Integer(), nullable=True),
        sa.Column("wt_axial_nonzero_slice_count", sa.Integer(), nullable=False),
        sa.Column("tc_voxel_count", sa.Integer(), nullable=False),
        sa.Column("tc_volume_mm3", sa.Float(), nullable=False),
        sa.Column("tc_volume_cm3", sa.Float(), nullable=False),
        sa.Column("tc_max_axial_area_mm2", sa.Float(), nullable=False),
        sa.Column("tc_max_axial_slice_index", sa.Integer(), nullable=True),
        sa.Column("tc_axial_nonzero_slice_count", sa.Integer(), nullable=False),
        sa.Column("et_voxel_count", sa.Integer(), nullable=False),
        sa.Column("et_volume_mm3", sa.Float(), nullable=False),
        sa.Column("et_volume_cm3", sa.Float(), nullable=False),
        sa.Column("et_max_axial_area_mm2", sa.Float(), nullable=False),
        sa.Column("et_max_axial_slice_index", sa.Integer(), nullable=True),
        sa.Column("et_axial_nonzero_slice_count", sa.Integer(), nullable=False),
        sa.Column("per_slice_area_storage_key", sa.String(length=500), nullable=False),
        sa.Column("per_slice_area_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("per_slice_area_size_bytes", sa.Integer(), nullable=False),
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
        sa.ForeignKeyConstraint(["segmentation_id"], ["segmentations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_tumor_quantifications_segmentation_id",
        "tumor_quantifications",
        ["segmentation_id"],
        unique=False,
    )
    op.create_index(
        "ux_tumor_quantifications_source_fingerprint",
        "tumor_quantifications",
        ["source_fingerprint_sha256"],
        unique=True,
    )
    op.create_index(
        "ix_tumor_quantifications_segmentation_created",
        "tumor_quantifications",
        ["segmentation_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_tumor_quantifications_segmentation_created", table_name="tumor_quantifications")
    op.drop_index("ux_tumor_quantifications_source_fingerprint", table_name="tumor_quantifications")
    op.drop_index("ix_tumor_quantifications_segmentation_id", table_name="tumor_quantifications")
    op.drop_table("tumor_quantifications")
