from __future__ import annotations

import uuid
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from gbm_ai.api.models.clinical import Sex
from gbm_ai.api.schemas.clinical import ALLOWED_SYMPTOMS


class UnifiedIntakeCreate(BaseModel):
    case_reference: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Optional human-facing local case reference. When omitted, NeuroGlioma AI "
            "generates one. This is not an ML feature."
        ),
    )
    patient_name: str | None = Field(default=None, max_length=200)
    age_years: int | None = Field(default=None, ge=18, le=100)
    sex: Sex = Sex.UNKNOWN
    mri_date: date
    symptoms: list[str] = Field(default_factory=list, max_length=20)
    symptom_duration: str | None = Field(default=None, max_length=100)
    prior_treatment: bool = False
    clinical_notes: str | None = Field(default=None, max_length=5000)

    @field_validator("case_reference")
    @classmethod
    def normalize_case_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().upper()
        return value or None


    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            item = value.strip().lower()
            if item not in ALLOWED_SYMPTOMS:
                raise ValueError(f"unsupported symptom {value!r}")
            if item not in normalized:
                normalized.append(item)
        return normalized

    @field_validator("patient_name")
    @classmethod
    def normalize_patient_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class StudyClinicalContextUpdate(BaseModel):
    age_years: int = Field(ge=18, le=100)


class StudyClinicalContextUpdateResponse(BaseModel):
    study_uuid: uuid.UUID
    case_reference: str
    age_years: int
    age_scope_status: Literal["adult"] = "adult"
    patient_context_used_as_ml_features: Literal[False] = False


class UnifiedIntakeResponse(BaseModel):
    version: Literal["phase8_step4_unified_intake_v1"] = (
        "phase8_step4_unified_intake_v1"
    )
    case_reference: str
    patient_uuid: uuid.UUID
    assessment_uuid: uuid.UUID
    study_uuid: uuid.UUID
    patient_reused: bool
    assessment_scope_status: str
    study_status: str
    internal_identifiers_managed_by_system: Literal[True] = True
    patient_context_used_as_ml_features: Literal[False] = False
    clinical_validation_claimed: Literal[False] = False
    next_step: Literal["upload_mri"] = "upload_mri"
