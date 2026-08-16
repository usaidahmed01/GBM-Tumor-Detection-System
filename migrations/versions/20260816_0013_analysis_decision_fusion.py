"""add phase 9 decision-fusion provenance to analysis runs

Revision ID: 20260816_0013
Revises: 20260816_0012
Create Date: 2026-08-16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260816_0013"
down_revision: Union[str, Sequence[str], None] = "20260816_0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "analysis_runs",
        sa.Column("decision_fusion_version", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "analysis_runs",
        sa.Column(
            "decision_evidence_summary",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
    )
    op.add_column(
        "analysis_runs",
        sa.Column("decision_fused_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("analysis_runs", "decision_fused_at")
    op.drop_column("analysis_runs", "decision_evidence_summary")
    op.drop_column("analysis_runs", "decision_fusion_version")
