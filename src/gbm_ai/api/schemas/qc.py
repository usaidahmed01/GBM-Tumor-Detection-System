from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from gbm_ai.api.models.analysis import StudyQCStatus
from gbm_ai.api.schemas.analysis import SeriesRead


class StudyQCResponse(BaseModel):
    study_uuid: uuid.UUID
    qc_status: StudyQCStatus
    manual_review_required: bool
    fail_reasons: list[str]
    partial_reasons: list[str]
    warnings: list[str]
    checks: dict
    inference_started: bool
    capability_routing_completed: bool
    next_step: str


class SequenceConfirmationRequest(BaseModel):
    sequence: str = Field(
        min_length=2,
        max_length=32,
        examples=["FLAIR"],
        description=(
            "Clinician-confirmed label: T1, T1C, T2, FLAIR, OTHER, NOT_USABLE."
        ),
    )


class SequenceConfirmationResponse(BaseModel):
    series: SeriesRead
    study_qc_invalidated: bool = True
    next_step: str = "rerun_mri_qc"
