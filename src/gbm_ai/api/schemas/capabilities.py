from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from gbm_ai.api.models.analysis import (
    BrainScopeStatus,
    CapabilityRoutingStatus,
    StudyStatus,
)


CapabilityStateLiteral = Literal[
    "eligible",
    "review_required",
    "blocked",
    "deferred",
]


class CapabilityDecision(BaseModel):
    state: CapabilityStateLiteral
    reasons: list[str]
    user_message: str | None = None
    prerequisites: list[str]
    input_eligible: bool
    execution_started: Literal[False] = False


class CapabilityMatrix(BaseModel):
    two_d_classification: CapabilityDecision
    gradcam_2d: CapabilityDecision
    three_d_segmentation: CapabilityDecision
    physical_volume: CapabilityDecision
    anatomical_localization: CapabilityDecision


class CapabilityRoutingResponse(BaseModel):
    study_uuid: uuid.UUID
    study_status: StudyStatus
    routing_status: CapabilityRoutingStatus
    brain_scope_status: BrainScopeStatus
    assessment_scope_status: str
    age_scope_status: str
    manual_review_required: bool
    global_block_reasons: list[str]
    global_review_reasons: list[str]
    capabilities: CapabilityMatrix
    eligible_capability_count: int
    review_capability_count: int
    model_execution_started: Literal[False] = False
    classifier_deployment_strategy_frozen: Literal[False] = False
    volumetric_to_2d_classifier_bridge_validated: Literal[False] = False
    clinical_validation_claimed: Literal[False] = False
    next_step: str


class BrainScopeConfirmationRequest(BaseModel):
    is_brain_mri: bool = Field(
        description=(
            "Clinician/researcher confirms whether the uploaded study is "
            "a brain MRI. This is a scope assertion only, not a diagnosis."
        )
    )


class BrainScopeConfirmationResponse(BaseModel):
    study_uuid: uuid.UUID
    brain_scope_status: BrainScopeStatus
    capability_routing_status: CapabilityRoutingStatus
    study_status: StudyStatus
    capability_routing_invalidated: Literal[True] = True
    next_step: Literal["rerun_capability_routing"] = (
        "rerun_capability_routing"
    )


class NiftiSequenceMappingRequest(BaseModel):
    t1: int = Field(ge=0)
    t1c: int = Field(ge=0)
    t2: int = Field(ge=0)
    flair: int = Field(ge=0)

    def normalized_mapping(self) -> dict[str, int]:
        return {
            "T1": self.t1,
            "T1C": self.t1c,
            "T2": self.t2,
            "FLAIR": self.flair,
        }


class NiftiSequenceMappingResponse(BaseModel):
    study_uuid: uuid.UUID
    mapping: dict[str, int]
    capability_routing_status: CapabilityRoutingStatus
    capability_routing_invalidated: Literal[True] = True
    next_step: Literal["rerun_capability_routing"] = (
        "rerun_capability_routing"
    )
