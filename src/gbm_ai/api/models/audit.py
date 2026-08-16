from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Index, JSON, String, Uuid, event, func
from sqlalchemy.orm import Mapped, mapped_column

from gbm_ai.api.models.base import Base


class AuditActorType(str, enum.Enum):
    SYSTEM = "system"
    DEMO_USER = "demo_user"
    AUTHENTICATED_USER = "authenticated_user"


class AuditAction(str, enum.Enum):
    PATIENT_CREATED = "patient_created"
    ASSESSMENT_CREATED = "assessment_created"
    STUDY_CREATED = "study_created"
    STUDY_SOURCE_STORED = "study_source_stored"
    STUDY_VIEWED = "study_viewed"
    STUDY_QC_COMPLETED = "study_qc_completed"
    SERIES_SEQ_CONFIRMED = "series_sequence_confirmed"
    STUDY_SCOPE_CONFIRMED = "study_scope_confirmed"
    NIFTI_SEQUENCE_MAPPED = "nifti_sequence_mapped"
    STUDY_CAPABILITY_ROUTED = "study_capability_routed"
    SEGMENTATION_PREPARATION_COMPLETED = "segmentation_preparation_completed"
    ANALYSIS_CREATED = "analysis_created"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    RESULT_VIEWED = "result_viewed"
    MODEL_VERSION_REGISTERED = "model_version_registered"
    OBJECT_DOWNLOADED = "object_downloaded"
    SEGMENTATION_EDITED = "segmentation_edited"
    SEGMENTATION_JOB_ENQUEUED = "segmentation_job_enqueued"
    SEGMENTATION_JOB_CLAIMED = "segmentation_job_claimed"
    SEGMENTATION_JOB_REQUEUED = "segmentation_job_requeued"
    SEGMENTATION_JOB_COMPLETED = "segmentation_job_completed"
    SEGMENTATION_JOB_FAILED = "segmentation_job_failed"
    QUANTIFICATION_COMPLETED = "quantification_completed"
    LOCALIZATION_COMPLETED = "localization_completed"
    DECISION_FUSED = "decision_fused"
    REPORT_FINALIZED = "report_finalized"
    CLINICAL_CONTEXT_UPDATED = "clinical_context_updated"


class AuditEntityType(str, enum.Enum):
    PATIENT = "patient"
    ASSESSMENT = "assessment"
    STUDY = "study"
    SERIES = "series"
    ANALYSIS_RUN = "analysis_run"
    MODEL_VERSION = "model_version"
    SEGMENTATION = "segmentation"
    SEGMENTATION_JOB = "segmentation_job"
    QUANTIFICATION = "quantification"
    LOCALIZATION = "localization"
    REPORT = "report"
    STORAGE_OBJECT = "storage_object"


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_entity", "entity_type", "entity_uuid"),
        Index("ix_audit_logs_created_at", "created_at"),
        Index("ix_audit_logs_request_id", "request_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    actor_type: Mapped[AuditActorType] = mapped_column(
        Enum(
            AuditActorType,
            name="audit_actor_type",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
        default=AuditActorType.SYSTEM,
    )
    # Optional future authenticated user UUID/string.
    # In the current no-auth prototype, DEMO_USER or SYSTEM is used.
    actor_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    action: Mapped[AuditAction] = mapped_column(
        Enum(
            AuditAction,
            name="audit_action",
            native_enum=False,
            validate_strings=True,
            length=64,
        ),
        nullable=False,
    )
    entity_type: Mapped[AuditEntityType] = mapped_column(
        Enum(
            AuditEntityType,
            name="audit_entity_type",
            native_enum=False,
            validate_strings=True,
            length=64,
        ),
        nullable=False,
    )
    entity_uuid: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    request_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    technical_context: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )



@event.listens_for(AuditLog, "before_update")
def _prevent_audit_update(mapper, connection, target) -> None:
    raise RuntimeError("AuditLog records are append-only and cannot be updated.")


@event.listens_for(AuditLog, "before_delete")
def _prevent_audit_delete(mapper, connection, target) -> None:
    raise RuntimeError("AuditLog records are append-only and cannot be deleted.")
