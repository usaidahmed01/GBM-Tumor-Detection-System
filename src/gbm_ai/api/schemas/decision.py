from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gbm_ai.api.models.analysis import DecisionState


class DecisionClassifierEvidence(BaseModel):
    available: bool
    validated_for_current_input_domain: bool
    analysis_run_uuid: uuid.UUID | None = None
    calibrated_probability_gbm: float | None = None
    classifier_state: DecisionState | None = None
    safety_reason_codes: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None


class DecisionSegmentationEvidence(BaseModel):
    available: bool
    segmentation_uuid: uuid.UUID | None = None
    review_status: str | None = None
    clinician_modified: bool = False
    lesion_evidence_present: bool | None = None
    wt_voxel_count: int | None = None
    quantification_available: bool = False
    localization_available: bool = False


class DecisionFusionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    version: str
    analysis_run_uuid: uuid.UUID
    study_uuid: uuid.UUID
    source_format: str
    decision_state: DecisionState
    calibrated_probability_gbm: float | None = None
    classifier: DecisionClassifierEvidence
    segmentation: DecisionSegmentationEvidence
    safety_reason_codes: list[str]
    other_intracranial_abnormality_not_excluded: bool
    report_ready: bool
    report_blockers: list[str]
    user_facing_summary: str
    clinical_notice: str
    segmentation_is_gbm_diagnosis: Literal[False]
    volumetric_classifier_bridge_validated: bool
    clinical_validation_claimed: Literal[False]
    fused_at: datetime


class DecisionFusionCurrentResponse(DecisionFusionResponse):
    pass
