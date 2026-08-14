from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, JSON, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from gbm_ai.api.models.base import Base, TimestampMixin


class AnatomicalLocalization(TimestampMixin, Base):
    """Atlas-based anatomical localization derived from a validated WT mask.

    Large spatial artifacts (warped mask, transform, overlap details) remain in
    protected object storage. This row stores traceability and concise result
    metadata only.
    """

    __tablename__ = "anatomical_localizations"
    __table_args__ = (
        Index("ux_anatomical_localizations_source_fingerprint", "source_fingerprint_sha256", unique=True),
        Index("ix_anatomical_localizations_segmentation_created", "segmentation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    segmentation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("segmentations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    quantification_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("tumor_quantifications.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    localization_version: Mapped[str] = mapped_column(String(128), nullable=False)
    source_fingerprint_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    standard_space: Mapped[str] = mapped_column(String(128), nullable=False)
    template_name: Mapped[str] = mapped_column(String(128), nullable=False)
    template_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    atlas_name: Mapped[str] = mapped_column(String(256), nullable=False)
    atlas_version: Mapped[str] = mapped_column(String(64), nullable=False)
    atlas_license: Mapped[str] = mapped_column(String(128), nullable=False)
    atlas_manifest_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    registration_method: Mapped[str] = mapped_column(String(128), nullable=False)
    registration_metric: Mapped[str] = mapped_column(String(128), nullable=False)
    registration_metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    registration_support_dice: Mapped[float] = mapped_column(Float, nullable=False)
    registration_qc_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)

    hemisphere: Mapped[str] = mapped_column(String(32), nullable=False)
    centroid_mni_mm: Mapped[list[float]] = mapped_column(JSON, nullable=False, default=list)
    primary_region: Mapped[str] = mapped_column(String(256), nullable=False)
    primary_region_overlap_voxels: Mapped[int] = mapped_column(Integer, nullable=False)
    primary_region_overlap_fraction_of_wt: Mapped[float] = mapped_column(Float, nullable=False)
    secondary_regions: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)

    transformed_wt_storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    transformed_wt_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    transformed_wt_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    transform_storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    transform_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    transform_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    overlap_details_storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    overlap_details_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    overlap_details_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    clinician_verification_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    anatomical_localization_generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    clinical_validation_claimed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
