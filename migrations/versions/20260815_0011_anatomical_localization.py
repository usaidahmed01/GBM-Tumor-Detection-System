"""add phase 7 anatomical localization records

Revision ID: 20260815_0011
Revises: 20260815_0010
Create Date: 2026-08-15
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260815_0011"
down_revision: Union[str, Sequence[str], None] = "20260815_0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "anatomical_localizations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("segmentation_id", sa.Uuid(), nullable=False),
        sa.Column("quantification_id", sa.Uuid(), nullable=False),
        sa.Column("localization_version", sa.String(length=128), nullable=False),
        sa.Column("source_fingerprint_sha256", sa.String(length=64), nullable=False),
        sa.Column("standard_space", sa.String(length=128), nullable=False),
        sa.Column("template_name", sa.String(length=128), nullable=False),
        sa.Column("template_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("atlas_name", sa.String(length=256), nullable=False),
        sa.Column("atlas_version", sa.String(length=64), nullable=False),
        sa.Column("atlas_license", sa.String(length=128), nullable=False),
        sa.Column("atlas_manifest_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("registration_method", sa.String(length=128), nullable=False),
        sa.Column("registration_metric", sa.String(length=128), nullable=False),
        sa.Column("registration_metric_value", sa.Float(), nullable=False),
        sa.Column("registration_support_dice", sa.Float(), nullable=False),
        sa.Column("registration_qc_passed", sa.Boolean(), nullable=False),
        sa.Column("hemisphere", sa.String(length=32), nullable=False),
        sa.Column("centroid_mni_mm", sa.JSON(), nullable=False),
        sa.Column("primary_region", sa.String(length=256), nullable=False),
        sa.Column("primary_region_overlap_voxels", sa.Integer(), nullable=False),
        sa.Column("primary_region_overlap_fraction_of_wt", sa.Float(), nullable=False),
        sa.Column("secondary_regions", sa.JSON(), nullable=False),
        sa.Column("transformed_wt_storage_key", sa.String(length=500), nullable=False),
        sa.Column("transformed_wt_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("transformed_wt_size_bytes", sa.Integer(), nullable=False),
        sa.Column("transform_storage_key", sa.String(length=500), nullable=False),
        sa.Column("transform_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("transform_size_bytes", sa.Integer(), nullable=False),
        sa.Column("overlap_details_storage_key", sa.String(length=500), nullable=False),
        sa.Column("overlap_details_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("overlap_details_size_bytes", sa.Integer(), nullable=False),
        sa.Column("clinician_verification_required", sa.Boolean(), nullable=False),
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
        sa.ForeignKeyConstraint(["quantification_id"], ["tumor_quantifications.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_anatomical_localizations_segmentation_id",
        "anatomical_localizations",
        ["segmentation_id"],
        unique=False,
    )
    op.create_index(
        "ix_anatomical_localizations_quantification_id",
        "anatomical_localizations",
        ["quantification_id"],
        unique=False,
    )
    op.create_index(
        "ux_anatomical_localizations_source_fingerprint",
        "anatomical_localizations",
        ["source_fingerprint_sha256"],
        unique=True,
    )
    op.create_index(
        "ix_anatomical_localizations_segmentation_created",
        "anatomical_localizations",
        ["segmentation_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_anatomical_localizations_segmentation_created", table_name="anatomical_localizations")
    op.drop_index("ux_anatomical_localizations_source_fingerprint", table_name="anatomical_localizations")
    op.drop_index("ix_anatomical_localizations_quantification_id", table_name="anatomical_localizations")
    op.drop_index("ix_anatomical_localizations_segmentation_id", table_name="anatomical_localizations")
    op.drop_table("anatomical_localizations")
