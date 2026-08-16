from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ViewerAssetKind = Literal[
    "mri_volume",
    "segmentation_mask",
    "segmentation_labelmap",
    "standard_space_mask",
    "derived_metadata",
]
ViewerCoordinateSpace = Literal[
    "patient_model_space_ras",
    "standard_space_mni152nlin6asym",
    "metadata",
]


class ViewerAssetResponse(BaseModel):
    alias: str
    kind: ViewerAssetKind
    format: Literal["nifti_gzip", "json"]
    media_type: str
    checksum_sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    coordinate_space: ViewerCoordinateSpace
    sequence: Literal["T1C", "T1", "T2", "FLAIR"] | None = None
    region: Literal["WT", "TC", "ET", "BRATS_LABELMAP"] | None = None
    download_url: str
    loader_url: str


class ViewerRegionMeasurementResponse(BaseModel):
    region: Literal["WT", "TC", "ET"]
    voxel_count: int = Field(ge=0)
    volume_mm3: float = Field(ge=0)
    volume_cm3: float = Field(ge=0)
    max_axial_area_mm2: float = Field(ge=0)
    max_axial_slice_index: int | None = None


class ViewerQuantificationSummaryResponse(BaseModel):
    available: bool
    stale: bool
    primary_region: Literal["WT"] = "WT"
    voxel_spacing_mm: list[float] = Field(default_factory=list)
    measurements: list[ViewerRegionMeasurementResponse] = Field(default_factory=list)


class ViewerLocalizationSummaryResponse(BaseModel):
    available: bool
    stale: bool
    standard_space: str | None = None
    hemisphere: Literal["left", "right", "bilateral", "midline"] | None = None
    centroid_mni_mm: list[float] = Field(default_factory=list)
    primary_region: str | None = None
    secondary_regions: list[dict] = Field(default_factory=list)
    registration_qc_passed: bool | None = None
    clinician_verification_required: bool = True


class ClinicalViewerManifestResponse(BaseModel):
    version: Literal["phase8_step1_clinical_viewer_backend_v1"]
    ui_version: Literal["phase8_step3_clinician_mask_review_v1"]
    status: Literal["ready"]
    study_uuid: uuid.UUID
    source_format: Literal["dicom", "nifti"]
    segmentation_uuid: uuid.UUID
    segmentation_review_status: Literal["unreviewed", "accepted", "edited", "rejected"]
    clinician_modified: bool
    primary_reference_sequence: Literal["T1C"] = "T1C"
    canonical_orientation: Literal["RAS"] = "RAS"
    viewer_planes: list[Literal["axial", "coronal", "sagittal"]]
    assets: list[ViewerAssetResponse]
    quantification: ViewerQuantificationSummaryResponse
    localization: ViewerLocalizationSummaryResponse
    raw_storage_keys_exposed: Literal[False] = False
    patient_identifiers_in_manifest: Literal[False] = False
    checksum_validation_required_before_streaming: Literal[True] = True
    overlays_available: Literal[True] = True
    three_dimensional_asset_basis_ready: Literal[True] = True
    cornerstone_or_ohif_frontend_implemented: Literal[True] = True
    manual_mask_editing_implemented: Literal[True] = True
    clinician_accept_reject_implemented: Literal[True] = True
    immutable_review_history_implemented: Literal[True] = True
    downstream_recalculation_after_edit: Literal[True] = True
    clinician_verification_required: Literal[True] = True
    segmentation_is_gbm_diagnosis: Literal[False] = False
    clinical_validation_claimed: Literal[False] = False
    next_step: Literal["phase8_step5_3d_volume_and_surface_review"]


class SegmentationReviewRequest(BaseModel):
    action: Literal["accept", "reject"]
    note: str | None = Field(default=None, max_length=1000)


class SegmentationReviewRevisionResponse(BaseModel):
    revision_uuid: uuid.UUID
    revision_number: int = Field(ge=1)
    action: Literal["accept", "reject", "edit"]
    source_review_status: Literal["unreviewed", "accepted", "edited", "rejected"]
    result_review_status: Literal["unreviewed", "accepted", "edited", "rejected"]
    source_mask_checksums: dict[str, str]
    result_mask_checksums: dict[str, str]
    modified_voxel_count: int = Field(ge=0)
    note: str | None = None
    downstream_quantification_policy: str
    downstream_localization_policy: str
    created_at: datetime


class SegmentationReviewResponse(BaseModel):
    version: Literal["phase8_step3_clinician_mask_review_v1"]
    status: Literal["complete"]
    segmentation_uuid: uuid.UUID
    review_status: Literal["unreviewed", "accepted", "edited", "rejected"]
    clinician_modified: bool
    revision: SegmentationReviewRevisionResponse
    current_mask_checksums: dict[str, str]
    downstream: dict[str, str | None]
    segmentation_is_gbm_diagnosis: Literal[False] = False
    clinical_validation_claimed: Literal[False] = False


class SegmentationReviewHistoryResponse(BaseModel):
    version: Literal["phase8_step3_clinician_mask_review_v1"]
    segmentation_uuid: uuid.UUID
    current_review_status: Literal["unreviewed", "accepted", "edited", "rejected"]
    clinician_modified: bool
    revisions: list[SegmentationReviewRevisionResponse]
    immutable_history: Literal[True] = True
    clinical_validation_claimed: Literal[False] = False
