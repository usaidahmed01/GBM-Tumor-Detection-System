from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gbm_ai.api.models.clinical import AssessmentStatus, ScopeStatus, Sex


ALLOWED_SYMPTOMS = {
    "headache",
    "seizure",
    "weakness",
    "vision_change",
    "speech_change",
    "cognitive_change",
    "other",
}


class PatientCreate(BaseModel):
    patient_id: str = Field(min_length=1, max_length=64, examples=["GBM-2026-0001"])
    patient_name: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional context field. Use synthetic/de-identified values in the "
            "unauthenticated research prototype."
        ),
    )
    age_years: int | None = Field(default=None, ge=18, le=100)
    sex: Sex = Sex.UNKNOWN
    privacy_flags: dict = Field(default_factory=dict)

    @field_validator("patient_id")
    @classmethod
    def normalize_patient_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("patient_id cannot be blank")
        return value

    @field_validator("patient_name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class PatientRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: str
    patient_name: str | None
    age_years: int | None
    sex: Sex
    privacy_flags: dict
    created_at: datetime
    updated_at: datetime


class AssessmentCreate(BaseModel):
    patient_uuid: uuid.UUID
    mri_date: date
    symptoms: list[str] = Field(default_factory=list, max_length=20)
    symptom_duration: str | None = Field(default=None, max_length=100)
    prior_treatment: bool
    clinical_notes: str | None = Field(default=None, max_length=5000)

    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            item = value.strip().lower()
            if item not in ALLOWED_SYMPTOMS:
                raise ValueError(
                    f"Unsupported symptom {value!r}. Allowed: {sorted(ALLOWED_SYMPTOMS)}"
                )
            if item not in normalized:
                normalized.append(item)
        return normalized


class AssessmentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    mri_date: date
    symptoms: list[str]
    symptom_duration: str | None
    prior_treatment: bool
    clinical_notes: str | None
    status: AssessmentStatus
    scope_status: ScopeStatus
    created_at: datetime
    updated_at: datetime


class AssessmentWithPatient(BaseModel):
    assessment: AssessmentRead
    patient: PatientRead


class ClinicalContextPolicy(BaseModel):
    ml_input_policy: Literal["MRI_ONLY_V1"] = "MRI_ONLY_V1"
    patient_id_used_as_ml_feature: Literal[False] = False
    patient_name_used_as_ml_feature: Literal[False] = False
    age_used_as_ml_feature: Literal[False] = False
    sex_used_as_ml_feature: Literal[False] = False
    symptoms_used_as_ml_feature: Literal[False] = False
    symptom_duration_used_as_ml_feature: Literal[False] = False
    prior_treatment_used_as_ml_feature: Literal[False] = False
    clinical_notes_used_as_ml_feature: Literal[False] = False
