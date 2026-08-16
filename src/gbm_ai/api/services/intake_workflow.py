from __future__ import annotations

import secrets
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import SourceFormat, Study, StudyStatus
from gbm_ai.api.models.clinical import (
    Assessment,
    AssessmentStatus,
    Patient,
    ScopeStatus,
)
from gbm_ai.api.schemas.intake import UnifiedIntakeCreate


UNIFIED_INTAKE_VERSION = "phase8_step4_unified_intake_v1"


class UnifiedIntakeConflictError(RuntimeError):
    pass


def _generated_case_reference() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"NGAI-{stamp}-{secrets.token_hex(3).upper()}"


def _find_patient(db: Session, case_reference: str) -> Patient | None:
    return db.scalar(select(Patient).where(Patient.patient_id == case_reference))


def _resolve_case_reference(db: Session, requested: str | None) -> tuple[str, Patient | None]:
    if requested:
        return requested, _find_patient(db, requested)

    # Extremely unlikely collision protection without exposing UUIDs to users.
    for _ in range(10):
        candidate = _generated_case_reference()
        if _find_patient(db, candidate) is None:
            return candidate, None
    raise RuntimeError("unable to allocate a unique NeuroGlioma AI case reference")


def create_unified_intake(db: Session, payload: UnifiedIntakeCreate) -> dict:
    """Create Patient -> Assessment -> Study as one committed intake transaction.

    UUIDs remain the durable relational/API keys, but the normal UI only exposes
    the human-facing case reference. Clinical context is persisted for workflow
    documentation and is not introduced as a V1 classifier feature.
    """

    case_reference, patient = _resolve_case_reference(db, payload.case_reference)
    patient_reused = patient is not None

    try:
        if patient is None:
            patient = Patient(
                patient_id=case_reference,
                patient_name=payload.patient_name,
                age_years=payload.age_years,
                sex=payload.sex,
                privacy_flags={
                    "neuroglioma_ai_intake": True,
                    "human_facing_case_reference": True,
                },
            )
            db.add(patient)
            db.flush()

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
        db.flush()

        study = Study(
            assessment_id=assessment.id,
            source_format=SourceFormat.PENDING,
            modality="MRI",
            deidentified_metadata={},
            status=StudyStatus.AWAITING_UPLOAD,
        )
        db.add(study)
        db.commit()
        db.refresh(patient)
        db.refresh(assessment)
        db.refresh(study)
    except IntegrityError as exc:
        db.rollback()
        raise UnifiedIntakeConflictError(
            "case reference already exists or intake could not be committed"
        ) from exc
    except Exception:
        db.rollback()
        raise

    return {
        "version": UNIFIED_INTAKE_VERSION,
        "case_reference": patient.patient_id,
        "patient_uuid": patient.id,
        "assessment_uuid": assessment.id,
        "study_uuid": study.id,
        "patient_reused": patient_reused,
        "assessment_scope_status": assessment.scope_status.value,
        "study_status": study.status.value,
        "internal_identifiers_managed_by_system": True,
        "patient_context_used_as_ml_features": False,
        "clinical_validation_claimed": False,
        "next_step": "upload_mri",
    }
