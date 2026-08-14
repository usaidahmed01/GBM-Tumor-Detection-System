from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, JSON, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from gbm_ai.api.models.base import Base, TimestampMixin


class TumorQuantification(TimestampMixin, Base):
    """Physical measurements derived from one persisted 3D segmentation.

    This record stores measurement/provenance metadata only. The detailed
    per-slice area series is kept in protected object storage so the database
    does not become a bulk imaging store.
    """

    __tablename__ = "tumor_quantifications"
    __table_args__ = (
        Index("ux_tumor_quantifications_source_fingerprint", "source_fingerprint_sha256", unique=True),
        Index("ix_tumor_quantifications_segmentation_created", "segmentation_id", "created_at"),
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
    quantification_version: Mapped[str] = mapped_column(String(128), nullable=False)
    source_fingerprint_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_review_status: Mapped[str] = mapped_column(String(32), nullable=False)
    source_clinician_modified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_mask_checksums: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    spatial_shape: Mapped[list[int]] = mapped_column(JSON, nullable=False, default=list)
    affine_ras: Mapped[list[list[float]]] = mapped_column(JSON, nullable=False, default=list)
    voxel_spacing_mm: Mapped[list[float]] = mapped_column(JSON, nullable=False, default=list)
    voxel_volume_mm3: Mapped[float] = mapped_column(Float, nullable=False)
    axial_pixel_area_mm2: Mapped[float] = mapped_column(Float, nullable=False)

    wt_voxel_count: Mapped[int] = mapped_column(Integer, nullable=False)
    wt_volume_mm3: Mapped[float] = mapped_column(Float, nullable=False)
    wt_volume_cm3: Mapped[float] = mapped_column(Float, nullable=False)
    wt_max_axial_area_mm2: Mapped[float] = mapped_column(Float, nullable=False)
    wt_max_axial_slice_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    wt_axial_nonzero_slice_count: Mapped[int] = mapped_column(Integer, nullable=False)

    tc_voxel_count: Mapped[int] = mapped_column(Integer, nullable=False)
    tc_volume_mm3: Mapped[float] = mapped_column(Float, nullable=False)
    tc_volume_cm3: Mapped[float] = mapped_column(Float, nullable=False)
    tc_max_axial_area_mm2: Mapped[float] = mapped_column(Float, nullable=False)
    tc_max_axial_slice_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tc_axial_nonzero_slice_count: Mapped[int] = mapped_column(Integer, nullable=False)

    et_voxel_count: Mapped[int] = mapped_column(Integer, nullable=False)
    et_volume_mm3: Mapped[float] = mapped_column(Float, nullable=False)
    et_volume_cm3: Mapped[float] = mapped_column(Float, nullable=False)
    et_max_axial_area_mm2: Mapped[float] = mapped_column(Float, nullable=False)
    et_max_axial_slice_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    et_axial_nonzero_slice_count: Mapped[int] = mapped_column(Integer, nullable=False)

    per_slice_area_storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    per_slice_area_checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    per_slice_area_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    physical_volume_generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    anatomical_localization_generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    clinical_validation_claimed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
