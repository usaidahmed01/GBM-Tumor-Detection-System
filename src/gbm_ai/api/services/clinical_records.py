from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from gbm_ai.api.models.clinical import Assessment, AssessmentStatus, Patient, ScopeStatus
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate


class DuplicatePatientIdentifierError(Exception):
    pass


class PatientNotFoundError(Exception):
    pass


class AssessmentNotFoundError(Exception):
    pass


def create_patient(db: Session, payload: PatientCreate) -> Patient:
    patient = Patient(
        patient_id=payload.patient_id,
        patient_name=payload.patient_name,
        age_years=payload.age_years,
        sex=payload.sex,
        privacy_flags=payload.privacy_flags,
    )
    db.add(patient)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise DuplicatePatientIdentifierError(payload.patient_id) from exc
    db.refresh(patient)
    return patient


def get_patient_by_uuid(db: Session, patient_uuid: uuid.UUID) -> Patient:
    patient = db.get(Patient, patient_uuid)
    if patient is None:
        raise PatientNotFoundError(str(patient_uuid))
    return patient


def get_patient_by_public_id(db: Session, patient_id: str) -> Patient:
    patient = db.scalar(select(Patient).where(Patient.patient_id == patient_id))
    if patient is None:
        raise PatientNotFoundError(patient_id)
    return patient


def create_assessment(db: Session, payload: AssessmentCreate) -> Assessment:
    patient = get_patient_by_uuid(db, payload.patient_uuid)

    scope = (
        ScopeStatus.OUT_OF_SCOPE_PRIOR_TREATMENT
        if payload.prior_treatment
        else ScopeStatus.IN_SCOPE
    )

    assessment = Assessment(
        patient_id=patient.id,
        mri_date=payload.mri_date,
        symptoms=payload.symptoms,
        symptom_duration=payload.symptom_duration,
        prior_treatment=payload.prior_treatment,
        clinical_notes=payload.clinical_notes,
        status=AssessmentStatus.READY_FOR_UPLOAD,
        scope_status=scope,
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    return assessment


def get_assessment(db: Session, assessment_id: uuid.UUID) -> Assessment:
    assessment = db.get(Assessment, assessment_id)
    if assessment is None:
        raise AssessmentNotFoundError(str(assessment_id))
    return assessment
