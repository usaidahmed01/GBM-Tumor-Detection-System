from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from gbm_ai.api.dependencies import get_db_session
from gbm_ai.api.schemas.intake import UnifiedIntakeCreate, UnifiedIntakeResponse
from gbm_ai.api.services.intake_workflow import UnifiedIntakeConflictError, create_unified_intake


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
