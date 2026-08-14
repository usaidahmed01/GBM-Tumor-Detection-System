"""add durable phase 6 segmentation background jobs

Revision ID: 20260814_0009
Revises: 20260814_0008
Create Date: 2026-08-15
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260814_0009"
down_revision: Union[str, Sequence[str], None] = "20260814_0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "segmentation_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("deduplication_key", sa.String(length=128), nullable=False),
        sa.Column("model_input_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "QUEUED",
                "RUNNING",
                "COMPLETE",
                "FAILED",
                name="segmentation_job_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("worker_id", sa.String(length=128), nullable=True),
        sa.Column("last_error_code", sa.String(length=128), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("analysis_run_id", sa.Uuid(), nullable=True),
        sa.Column("segmentation_id", sa.Uuid(), nullable=True),
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
        sa.ForeignKeyConstraint(["study_id"], ["studies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["analysis_run_id"], ["analysis_runs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["segmentation_id"], ["segmentations.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_segmentation_jobs_study_id", "segmentation_jobs", ["study_id"], unique=False)
    op.create_index("ix_segmentation_jobs_analysis_run_id", "segmentation_jobs", ["analysis_run_id"], unique=False)
    op.create_index("ix_segmentation_jobs_segmentation_id", "segmentation_jobs", ["segmentation_id"], unique=False)
    op.create_index(
        "ux_segmentation_jobs_deduplication_key",
        "segmentation_jobs",
        ["deduplication_key"],
        unique=True,
    )
    op.create_index(
        "ix_segmentation_jobs_status_available",
        "segmentation_jobs",
        ["status", "available_at"],
        unique=False,
    )
    op.create_index(
        "ix_segmentation_jobs_study_created",
        "segmentation_jobs",
        ["study_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_segmentation_jobs_study_created", table_name="segmentation_jobs")
    op.drop_index("ix_segmentation_jobs_status_available", table_name="segmentation_jobs")
    op.drop_index("ux_segmentation_jobs_deduplication_key", table_name="segmentation_jobs")
    op.drop_index("ix_segmentation_jobs_segmentation_id", table_name="segmentation_jobs")
    op.drop_index("ix_segmentation_jobs_analysis_run_id", table_name="segmentation_jobs")
    op.drop_index("ix_segmentation_jobs_study_id", table_name="segmentation_jobs")
    op.drop_table("segmentation_jobs")
