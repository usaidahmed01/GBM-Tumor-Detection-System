from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from gbm_ai.api.models.analysis import (
    AnalysisStatus,
    DecisionState,
    ModelRole,
    QCState,
    SourceFormat,
    StudyStatus,
)


class StudyCreate(BaseModel):
    assessment_uuid: uuid.UUID


class StudyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    assessment_id: uuid.UUID
    source_format: SourceFormat
    modality: str
    study_instance_uid: str | None
    deidentified_metadata: dict
    storage_key: str | None
    checksum_sha256: str | None
    status: StudyStatus
    created_at: datetime
    updated_at: datetime


class ModelVersionCreate(BaseModel):
    model_name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    role: ModelRole
    architecture: str = Field(min_length=1, max_length=128)
    weights_checksum_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    code_version: str = Field(min_length=1, max_length=128)
    preprocessing_version: str = Field(min_length=1, max_length=128)
    threshold_version: str | None = Field(default=None, max_length=128)
    calibration_version: str | None = Field(default=None, max_length=128)
    license_source_notes: str | None = Field(default=None, max_length=5000)
    is_active: bool = False


class ModelVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    model_name: str
    version: str
    role: ModelRole
    architecture: str
    weights_checksum_sha256: str | None
    code_version: str
    preprocessing_version: str
    threshold_version: str | None
    calibration_version: str | None
    license_source_notes: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class AnalysisRunCreate(BaseModel):
    study_uuid: uuid.UUID
    classifier_model_version_uuid: uuid.UUID | None = None
    segmentation_model_version_uuid: uuid.UUID | None = None


class AnalysisRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    study_id: uuid.UUID
    classifier_model_version_id: uuid.UUID | None
    segmentation_model_version_id: uuid.UUID | None
    status: AnalysisStatus
    qc_state: QCState
    ood_score: float | None
    ood_likeness_candidate: bool | None
    raw_probability_gbm: float | None
    calibrated_probability_gbm: float | None
    decision_state: DecisionState
    safety_reason_codes: list[str]
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime
