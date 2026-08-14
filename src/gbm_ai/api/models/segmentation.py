from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, JSON, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from gbm_ai.api.models.base import Base, TimestampMixin


class SegmentationStatus(str, enum.Enum):
    GENERATED = "generated"


class SegmentationReviewStatus(str, enum.Enum):
    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"


class SegmentationJobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class SegmentationJob(TimestampMixin, Base):
    """Durable background-execution record for one prepared 3D model input.

    A job is bound to the immutable Step 4 model-input checksum.  It never
    stores patient identifiers, clinical text, MRI pixels, or masks.
    """

    __tablename__ = "segmentation_jobs"
    __table_args__ = (
        Index("ux_segmentation_jobs_deduplication_key", "deduplication_key", unique=True),
        Index("ix_segmentation_jobs_status_available", "status", "available_at"),
        Index("ix_segmentation_jobs_study_created", "study_id", "created_at"),
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
    deduplication_key: Mapped[str] = mapped_column(String(128), nullable=False)
    model_input_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[SegmentationJobStatus] = mapped_column(
        Enum(
            SegmentationJobStatus,
            name="segmentation_job_status",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
        default=SegmentationJobStatus.QUEUED,
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    analysis_run_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("analysis_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    segmentation_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("segmentations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )


class Segmentation(TimestampMixin, Base):
    """Persistent technical record for one generated 3D segmentation result.

    Physical volumes and anatomical localization intentionally do not live in
    this Phase 6 Step 5 record. Those are later validated derivations.
    """

    __tablename__ = "segmentations"
    __table_args__ = (
        Index("ux_segmentations_analysis_run", "analysis_run_id", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[SegmentationStatus] = mapped_column(
        Enum(
            SegmentationStatus,
            name="segmentation_status",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
        default=SegmentationStatus.GENERATED,
    )

    model_input_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    inference_version: Mapped[str] = mapped_column(String(128), nullable=False)
    preprocessing_version: Mapped[str] = mapped_column(String(128), nullable=False)
    bundle_name: Mapped[str] = mapped_column(String(128), nullable=False)
    bundle_version: Mapped[str] = mapped_column(String(64), nullable=False)
    weights_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    device: Mapped[str] = mapped_column(String(32), nullable=False)
    amp_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    roi_size: Mapped[list[int]] = mapped_column(JSON, nullable=False, default=list)
    overlap: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    spatial_shape: Mapped[list[int]] = mapped_column(JSON, nullable=False, default=list)
    affine_ras: Mapped[list[list[float]]] = mapped_column(JSON, nullable=False, default=list)

    tc_storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    tc_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    tc_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    wt_storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    wt_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    wt_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    et_storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    et_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    et_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    labelmap_storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    labelmap_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    labelmap_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    voxel_counts: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    runtime_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    review_status: Mapped[SegmentationReviewStatus] = mapped_column(
        Enum(
            SegmentationReviewStatus,
            name="segmentation_review_status",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
        default=SegmentationReviewStatus.UNREVIEWED,
    )
    clinician_modified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    physical_volume_generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    anatomical_localization_generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    clinical_validation_claimed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
