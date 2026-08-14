from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


TumorRegion = Literal["WT", "TC", "ET"]


class TumorRegionQuantificationResponse(BaseModel):
    region: TumorRegion
    voxel_count: int
    volume_mm3: float
    volume_cm3: float
    max_axial_area_mm2: float
    max_axial_slice_index: int | None = None
    axial_nonzero_slice_count: int


class PerSliceAreaArtifactResponse(BaseModel):
    storage_key: str
    checksum_sha256: str
    size_bytes: int
    format: Literal["json"] = "json"
    plane: Literal["axial_ras"] = "axial_ras"


class TumorQuantificationResponse(BaseModel):
    version: Literal["phase7_step1_physical_quantification_v1"]
    status: Literal["complete"]
    study_uuid: uuid.UUID
    analysis_run_uuid: uuid.UUID
    segmentation_uuid: uuid.UUID
    quantification_uuid: uuid.UUID
    source_format: Literal["dicom", "nifti"]
    source_review_status: Literal["unreviewed", "accepted", "edited"]
    clinician_modified: bool
    spatial_shape: list[int]
    affine_ras: list[list[float]]
    voxel_spacing_mm: list[float]
    voxel_volume_mm3: float
    axial_pixel_area_mm2: float
    primary_quantitative_region: Literal["WT"] = "WT"
    regions: list[TumorRegionQuantificationResponse] = Field(default_factory=list)
    per_slice_area_artifact: PerSliceAreaArtifactResponse
    measurement_basis: Literal["3d_segmentation_plus_valid_spatial_metadata"] = (
        "3d_segmentation_plus_valid_spatial_metadata"
    )
    segmentation_is_gbm_diagnosis: Literal[False] = False
    physical_volume_generated: Literal[True] = True
    anatomical_localization_generated: Literal[False] = False
    clinical_validation_claimed: Literal[False] = False
    next_step: Literal["phase7_step2_anatomical_localization"] = (
        "phase7_step2_anatomical_localization"
    )
