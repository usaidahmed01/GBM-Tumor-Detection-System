from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from gbm_ai.api.dependencies import get_db_session
from gbm_ai.api.models.analysis import Study
from gbm_ai.api.schemas.intake import (
    StudyClinicalContextUpdate,
    StudyClinicalContextUpdateResponse,
    UnifiedIntakeCreate,
    UnifiedIntakeResponse,
)
from gbm_ai.api.services.intake_workflow import (
    UnifiedIntakeConflictError,
    create_unified_intake,
    update_study_patient_age,
)


router = APIRouter(tags=["unified-intake"])


@router.post(
    "/intake/studies",
    response_model=UnifiedIntakeResponse,
    status_code=status.HTTP_201_CREATED,
    summary=(
        "Create the patient/assessment/study records for one NeuroGlioma AI "
        "analysis without asking the user to manage UUIDs"
    ),
)
def unified_intake_create(
    payload: UnifiedIntakeCreate,
    db: Session = Depends(get_db_session),
):
    try:
        return create_unified_intake(db, payload)
    except UnifiedIntakeConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.patch(
    "/studies/{study_uuid}/intake-context",
    response_model=StudyClinicalContextUpdateResponse,
    summary="Update adult-scope context for the current MRI assessment",
)
def study_intake_context_update(
    study_uuid: uuid.UUID,
    payload: StudyClinicalContextUpdate,
    request: Request,
    db: Session = Depends(get_db_session),
):
    study = db.get(Study, study_uuid)
    if study is None:
        raise HTTPException(status_code=404, detail="study not found")
    try:
        return update_study_patient_age(
            db,
            study,
            payload.age_years,
            request_id=getattr(request.state, "request_id", None),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
