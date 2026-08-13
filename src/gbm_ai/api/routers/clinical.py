from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from gbm_ai.api.dependencies import get_db_session
from gbm_ai.api.schemas.clinical import (
    AssessmentCreate,
    AssessmentRead,
    AssessmentWithPatient,
    ClinicalContextPolicy,
    PatientCreate,
    PatientRead,
)
from gbm_ai.api.services.clinical_records import (
    AssessmentNotFoundError,
    DuplicatePatientIdentifierError,
    PatientNotFoundError,
    create_assessment,
    create_patient,
    get_assessment,
    get_patient_by_public_id,
    get_patient_by_uuid,
)

router = APIRouter(tags=["clinical-records"])


@router.post("/patients", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
def patient_create(payload: PatientCreate, db: Session = Depends(get_db_session)):
    try:
        return create_patient(db, payload)
    except DuplicatePatientIdentifierError:
        raise HTTPException(status_code=409, detail="patient_id already exists")


@router.get("/patients/by-id/{patient_id}", response_model=PatientRead)
def patient_get_by_public_id(patient_id: str, db: Session = Depends(get_db_session)):
    try:
        return get_patient_by_public_id(db, patient_id)
    except PatientNotFoundError:
        raise HTTPException(status_code=404, detail="patient not found")


@router.get("/patients/{patient_uuid}", response_model=PatientRead)
def patient_get(patient_uuid: uuid.UUID, db: Session = Depends(get_db_session)):
    try:
        return get_patient_by_uuid(db, patient_uuid)
    except PatientNotFoundError:
        raise HTTPException(status_code=404, detail="patient not found")


@router.post("/assessments", response_model=AssessmentRead, status_code=status.HTTP_201_CREATED)
def assessment_create(payload: AssessmentCreate, db: Session = Depends(get_db_session)):
    try:
        return create_assessment(db, payload)
    except PatientNotFoundError:
        raise HTTPException(status_code=404, detail="patient not found")


@router.get("/assessments/{assessment_id}", response_model=AssessmentWithPatient)
def assessment_get(assessment_id: uuid.UUID, db: Session = Depends(get_db_session)):
    try:
        assessment = get_assessment(db, assessment_id)
        patient = get_patient_by_uuid(db, assessment.patient_id)
    except (AssessmentNotFoundError, PatientNotFoundError):
        raise HTTPException(status_code=404, detail="assessment not found")

    return AssessmentWithPatient(
        assessment=AssessmentRead.model_validate(assessment),
        patient=PatientRead.model_validate(patient),
    )


@router.get("/clinical-context-policy", response_model=ClinicalContextPolicy)
def clinical_context_policy():
    return ClinicalContextPolicy()
