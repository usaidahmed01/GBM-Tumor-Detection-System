"""create patient and assessment tables

Revision ID: 20260814_0001
Revises:
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260814_0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "patients",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.String(length=64), nullable=False),
        sa.Column("patient_name", sa.String(length=200), nullable=True),
        sa.Column("age_years", sa.Integer(), nullable=True),
        sa.Column(
            "sex",
            sa.Enum("MALE", "FEMALE", "OTHER", "UNKNOWN",
                    name="patient_sex", native_enum=False),
            nullable=False,
        ),
        sa.Column("privacy_flags", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_patients_patient_id", "patients", ["patient_id"], unique=True)

    op.create_table(
        "assessments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("mri_date", sa.Date(), nullable=False),
        sa.Column("symptoms", sa.JSON(), nullable=False),
        sa.Column("symptom_duration", sa.String(length=100), nullable=True),
        sa.Column("prior_treatment", sa.Boolean(), nullable=False),
        sa.Column("clinical_notes", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("DRAFT", "READY_FOR_UPLOAD", "PROCESSING", "COMPLETE", "FAILED",
                    name="assessment_status", native_enum=False),
            nullable=False,
        ),
        sa.Column(
            "scope_status",
            sa.Enum("IN_SCOPE", "OUT_OF_SCOPE_PRIOR_TREATMENT",
                    name="assessment_scope_status", native_enum=False),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_assessments_patient_id", "assessments", ["patient_id"], unique=False)
    op.create_index(
        "ix_assessments_patient_created",
        "assessments",
        ["patient_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_assessments_patient_created", table_name="assessments")
    op.drop_index("ix_assessments_patient_id", table_name="assessments")
    op.drop_table("assessments")
    op.drop_index("ix_patients_patient_id", table_name="patients")
    op.drop_table("patients")
