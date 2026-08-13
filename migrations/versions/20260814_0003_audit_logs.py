"""create append-only audit log table

Revision ID: 20260814_0003
Revises: 20260814_0002
Create Date: 2026-08-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260814_0003"
down_revision: Union[str, Sequence[str], None] = "20260814_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "actor_type",
            sa.Enum(
                "SYSTEM",
                "DEMO_USER",
                "AUTHENTICATED_USER",
                name="audit_actor_type",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("actor_id", sa.String(length=128), nullable=True),
        sa.Column(
            "action",
            sa.Enum(
                "PATIENT_CREATED",
                "ASSESSMENT_CREATED",
                "STUDY_CREATED",
                "STUDY_SOURCE_STORED",
                "STUDY_VIEWED",
                "ANALYSIS_CREATED",
                "ANALYSIS_STARTED",
                "ANALYSIS_COMPLETED",
                "ANALYSIS_FAILED",
                "RESULT_VIEWED",
                "MODEL_VERSION_REGISTERED",
                "OBJECT_DOWNLOADED",
                "SEGMENTATION_EDITED",
                "REPORT_FINALIZED",
                name="audit_action",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "entity_type",
            sa.Enum(
                "PATIENT",
                "ASSESSMENT",
                "STUDY",
                "ANALYSIS_RUN",
                "MODEL_VERSION",
                "SEGMENTATION",
                "REPORT",
                "STORAGE_OBJECT",
                name="audit_entity_type",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("entity_uuid", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("technical_context", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_audit_logs_entity",
        "audit_logs",
        ["entity_type", "entity_uuid"],
        unique=False,
    )
    op.create_index(
        "ix_audit_logs_created_at",
        "audit_logs",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_logs_request_id",
        "audit_logs",
        ["request_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_audit_logs_request_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_entity", table_name="audit_logs")
    op.drop_table("audit_logs")
