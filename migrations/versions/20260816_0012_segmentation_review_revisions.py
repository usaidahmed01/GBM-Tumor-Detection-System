"""add phase 8 clinician segmentation review revisions

Revision ID: 20260816_0012
Revises: 20260815_0011
Create Date: 2026-08-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260816_0012"
down_revision: Union[str, Sequence[str], None] = "20260815_0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "segmentation_review_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("segmentation_id", sa.Uuid(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("review_version", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("source_review_status", sa.String(length=32), nullable=False),
        sa.Column("result_review_status", sa.String(length=32), nullable=False),
        sa.Column("source_artifacts", sa.JSON(), nullable=False),
        sa.Column("result_artifacts", sa.JSON(), nullable=False),
        sa.Column("modified_voxel_count", sa.Integer(), nullable=False),
        sa.Column("note", sa.String(length=1000), nullable=True),
        sa.Column("downstream_quantification_policy", sa.String(length=128), nullable=False),
        sa.Column("downstream_localization_policy", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["segmentation_id"], ["segmentations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_segmentation_review_revisions_segmentation_id",
        "segmentation_review_revisions",
        ["segmentation_id"],
        unique=False,
    )
    op.create_index(
        "ux_segmentation_review_revision_number",
        "segmentation_review_revisions",
        ["segmentation_id", "revision_number"],
        unique=True,
    )
    op.create_index(
        "ix_segmentation_review_revision_created",
        "segmentation_review_revisions",
        ["segmentation_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_segmentation_review_revision_created",
        table_name="segmentation_review_revisions",
    )
    op.drop_index(
        "ux_segmentation_review_revision_number",
        table_name="segmentation_review_revisions",
    )
    op.drop_index(
        "ix_segmentation_review_revisions_segmentation_id",
        table_name="segmentation_review_revisions",
    )
    op.drop_table("segmentation_review_revisions")
