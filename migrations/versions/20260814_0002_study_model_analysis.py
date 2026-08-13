"""create study model version and analysis run tables

Revision ID: 20260814_0002
Revises: 20260814_0001
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260814_0002"
down_revision: Union[str, Sequence[str], None] = "20260814_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "studies",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("assessment_id", sa.Uuid(), nullable=False),
        sa.Column(
            "source_format",
            sa.Enum(
                "PENDING", "IMAGE", "DICOM", "NIFTI",
                name="study_source_format",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("modality", sa.String(length=32), nullable=False),
        sa.Column("study_instance_uid", sa.String(length=128), nullable=True),
        sa.Column("deidentified_metadata", sa.JSON(), nullable=False),
        sa.Column("storage_key", sa.String(length=500), nullable=True),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "AWAITING_UPLOAD",
                "UPLOADED",
                "READY_FOR_ANALYSIS",
                "FAILED",
                name="study_status",
                native_enum=False,
            ),
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
            ["assessment_id"],
            ["assessments.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_studies_assessment_id",
        "studies",
        ["assessment_id"],
        unique=False,
    )
    op.create_index(
        "ix_studies_study_instance_uid",
        "studies",
        ["study_instance_uid"],
        unique=False,
    )
    op.create_index(
        "ix_studies_assessment_created",
        "studies",
        ["assessment_id", "created_at"],
        unique=False,
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column(
            "role",
            sa.Enum(
                "CLASSIFIER",
                "SEGMENTATION",
                name="model_role",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("architecture", sa.String(length=128), nullable=False),
        sa.Column("weights_checksum_sha256", sa.String(length=64), nullable=True),
        sa.Column("code_version", sa.String(length=128), nullable=False),
        sa.Column("preprocessing_version", sa.String(length=128), nullable=False),
        sa.Column("threshold_version", sa.String(length=128), nullable=True),
        sa.Column("calibration_version", sa.String(length=128), nullable=True),
        sa.Column("license_source_notes", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ux_model_versions_name_version",
        "model_versions",
        ["model_name", "version"],
        unique=True,
    )

    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("classifier_model_version_id", sa.Uuid(), nullable=True),
        sa.Column("segmentation_model_version_id", sa.Uuid(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING", "RUNNING", "COMPLETE", "FAILED",
                name="analysis_status",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "qc_state",
            sa.Enum(
                "PENDING", "PASS", "REVIEW", "FAIL",
                name="analysis_qc_state",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("ood_score", sa.Float(), nullable=True),
        sa.Column("ood_likeness_candidate", sa.Boolean(), nullable=True),
        sa.Column("raw_probability_gbm", sa.Float(), nullable=True),
        sa.Column("calibrated_probability_gbm", sa.Float(), nullable=True),
        sa.Column(
            "decision_state",
            sa.Enum(
                "PENDING",
                "GBM_SUSPECTED",
                "GBM_NOT_SUSPECTED",
                "INDETERMINATE",
                name="analysis_decision_state",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("safety_reason_codes", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
            ["classifier_model_version_id"],
            ["model_versions.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["segmentation_model_version_id"],
            ["model_versions.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["study_id"],
            ["studies.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_analysis_runs_study_id",
        "analysis_runs",
        ["study_id"],
        unique=False,
    )
    op.create_index(
        "ix_analysis_runs_classifier_model_version_id",
        "analysis_runs",
        ["classifier_model_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_analysis_runs_segmentation_model_version_id",
        "analysis_runs",
        ["segmentation_model_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_analysis_runs_study_created",
        "analysis_runs",
        ["study_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_analysis_runs_study_created",
        table_name="analysis_runs",
    )
    op.drop_index(
        "ix_analysis_runs_segmentation_model_version_id",
        table_name="analysis_runs",
    )
    op.drop_index(
        "ix_analysis_runs_classifier_model_version_id",
        table_name="analysis_runs",
    )
    op.drop_index(
        "ix_analysis_runs_study_id",
        table_name="analysis_runs",
    )
    op.drop_table("analysis_runs")

    op.drop_index(
        "ux_model_versions_name_version",
        table_name="model_versions",
    )
    op.drop_table("model_versions")

    op.drop_index(
        "ix_studies_assessment_created",
        table_name="studies",
    )
    op.drop_index(
        "ix_studies_study_instance_uid",
        table_name="studies",
    )
    op.drop_index(
        "ix_studies_assessment_id",
        table_name="studies",
    )
    op.drop_table("studies")
