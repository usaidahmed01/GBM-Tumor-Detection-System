from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Index, JSON, String, Text, Uuid, event
from sqlalchemy.orm import Mapped, mapped_column

from gbm_ai.api.models.base import Base, TimestampMixin


class ReportStatus(str, enum.Enum):
    FINALIZED = "finalized"


class ClinicalReport(TimestampMixin, Base):
    __tablename__ = "clinical_reports"
    __table_args__ = (
        Index("ix_clinical_reports_study_created", "study_id", "created_at"),
        Index("ux_clinical_reports_report_checksum", "report_checksum_sha256", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("studies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    decision_analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("analysis_runs.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    report_version: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[ReportStatus] = mapped_column(
        Enum(ReportStatus, name="clinical_report_status", native_enum=False, validate_strings=True),
        nullable=False,
        default=ReportStatus.FINALIZED,
    )
    report_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    report_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    clinician_name: Mapped[str] = mapped_column(String(200), nullable=False)
    clinician_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    signed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    signoff_identity_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


@event.listens_for(ClinicalReport, "before_update")
def _prevent_final_report_update(mapper, connection, target) -> None:
    raise RuntimeError("Finalized ClinicalReport records are immutable.")


@event.listens_for(ClinicalReport, "before_delete")
def _prevent_final_report_delete(mapper, connection, target) -> None:
    raise RuntimeError("Finalized ClinicalReport records are immutable.")
