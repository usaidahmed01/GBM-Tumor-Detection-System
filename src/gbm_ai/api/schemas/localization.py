from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class RegistrationQCResponse(BaseModel):
    method: Literal["rigid_plus_affine_mattes_mutual_information"]
    metric: Literal["MattesMutualInformation"]
    metric_value: float
    support_dice: float
    qc_passed: Literal[True] = True
    engineering_qc_threshold: float


class SecondaryRegionResponse(BaseModel):
    atlas: str
    label_index: int
    label: str
    overlap_voxels: int
    fraction_of_wt: float


class AnatomicalLocalizationResponse(BaseModel):
    version: Literal["phase7_step2_anatomical_localization_v1"]
    status: Literal["complete"]
    study_uuid: uuid.UUID
    segmentation_uuid: uuid.UUID
    quantification_uuid: uuid.UUID
    localization_uuid: uuid.UUID
    standard_space: Literal["MNI152NLin6Asym"]
    template_name: str
    atlas_name: str
    atlas_version: str
    atlas_license: Literal["CC BY-SA 4.0"]
    registration: RegistrationQCResponse
    hemisphere: Literal["left", "right", "bilateral", "midline"]
    centroid_mni_mm: list[float] = Field(min_length=3, max_length=3)
    primary_region: str
    primary_region_overlap_voxels: int
    primary_region_overlap_fraction_of_wt: float
    secondary_regions: list[SecondaryRegionResponse] = Field(default_factory=list)
    clinician_verification_required: Literal[True] = True
    anatomical_localization_generated: Literal[True] = True
    localization_is_functional_deficit_prediction: Literal[False] = False
    segmentation_is_gbm_diagnosis: Literal[False] = False
    clinical_validation_claimed: Literal[False] = False
    next_step: Literal["phase8_clinical_viewer"]
