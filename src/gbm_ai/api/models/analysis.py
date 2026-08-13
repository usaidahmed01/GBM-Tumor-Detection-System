from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gbm_ai.api.models.base import Base, TimestampMixin


class SourceFormat(str, enum.Enum):
    PENDING = "pending"
    IMAGE = "image"
    DICOM = "dicom"
    NIFTI = "nifti"


class StudyStatus(str, enum.Enum):
    AWAITING_UPLOAD = "awaiting_upload"
    UPLOADED = "uploaded"
    READY_FOR_ANALYSIS = "ready_for_analysis"
    FAILED = "failed"


class ModelRole(str, enum.Enum):
    CLASSIFIER = "classifier"
    SEGMENTATION = "segmentation"


class AnalysisStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class QCState(str, enum.Enum):
    PENDING = "pending"
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"


class DecisionState(str, enum.Enum):
    PENDING = "pending"
    GBM_SUSPECTED = "gbm_suspected"
    GBM_NOT_SUSPECTED = "gbm_not_suspected"
    INDETERMINATE = "indeterminate"


class Study(TimestampMixin, Base):
    __tablename__ = "studies"
    __table_args__ = (
        Index("ix_studies_assessment_created", "assessment_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_format: Mapped[SourceFormat] = mapped_column(
        Enum(
            SourceFormat,
            name="study_source_format",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
        default=SourceFormat.PENDING,
    )
    modality: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="MRI",
    )
    study_instance_uid: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )
    deidentified_metadata: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    storage_key: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )
    checksum_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    status: Mapped[StudyStatus] = mapped_column(
        Enum(
            StudyStatus,
            name="study_status",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
        default=StudyStatus.AWAITING_UPLOAD,
    )

    analysis_runs: Mapped[list["AnalysisRun"]] = relationship(
        back_populates="study",
        cascade="all, delete-orphan",
    )


class ModelVersion(TimestampMixin, Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        Index(
            "ux_model_versions_name_version",
            "model_name",
            "version",
            unique=True,
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[ModelRole] = mapped_column(
        Enum(
            ModelRole,
            name="model_role",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
    )
    architecture: Mapped[str] = mapped_column(String(128), nullable=False)
    weights_checksum_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    code_version: Mapped[str] = mapped_column(String(128), nullable=False)
    preprocessing_version: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    threshold_version: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    calibration_version: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    license_source_notes: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )


class AnalysisRun(TimestampMixin, Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (
        Index("ix_analysis_runs_study_created", "study_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    study_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("studies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    classifier_model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("model_versions.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    segmentation_model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("model_versions.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(
            AnalysisStatus,
            name="analysis_status",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
        default=AnalysisStatus.PENDING,
    )
    qc_state: Mapped[QCState] = mapped_column(
        Enum(
            QCState,
            name="analysis_qc_state",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
        default=QCState.PENDING,
    )
    ood_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ood_likeness_candidate: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )
    raw_probability_gbm: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    calibrated_probability_gbm: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    decision_state: Mapped[DecisionState] = mapped_column(
        Enum(
            DecisionState,
            name="analysis_decision_state",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
        default=DecisionState.PENDING,
    )
    safety_reason_codes: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    study: Mapped[Study] = relationship(back_populates="analysis_runs")
