"""add immutable structured clinical reports and sign-off

Revision ID: 20260816_0014
Revises: 20260816_0013
Create Date: 2026-08-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260816_0014"
down_revision: Union[str, Sequence[str], None] = "20260816_0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "clinical_reports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("decision_analysis_run_id", sa.Uuid(), nullable=False),
        sa.Column("report_version", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("report_payload", sa.JSON(), nullable=False),
        sa.Column("report_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("clinician_name", sa.String(length=200), nullable=False),
        sa.Column("clinician_comment", sa.Text(), nullable=True),
        sa.Column("signed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("signoff_identity_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["decision_analysis_run_id"], ["analysis_runs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["study_id"], ["studies.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_clinical_reports_study_created", "clinical_reports", ["study_id", "created_at"], unique=False)
    op.create_index("ix_clinical_reports_study_id", "clinical_reports", ["study_id"], unique=False)
    op.create_index("ix_clinical_reports_decision_analysis_run_id", "clinical_reports", ["decision_analysis_run_id"], unique=False)
    op.create_index("ux_clinical_reports_report_checksum", "clinical_reports", ["report_checksum_sha256"], unique=True)


def downgrade() -> None:
    op.drop_index("ux_clinical_reports_report_checksum", table_name="clinical_reports")
    op.drop_index("ix_clinical_reports_decision_analysis_run_id", table_name="clinical_reports")
    op.drop_index("ix_clinical_reports_study_id", table_name="clinical_reports")
    op.drop_index("ix_clinical_reports_study_created", table_name="clinical_reports")
    op.drop_table("clinical_reports")
