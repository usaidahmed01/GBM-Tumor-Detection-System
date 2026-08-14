from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from gbm_ai.api.dependencies import get_db_session, get_object_store
from gbm_ai.api.models.audit import AuditActorType
from gbm_ai.api.schemas.quantification import TumorQuantificationResponse
from gbm_ai.api.services.analysis_records import StudyNotFoundError, get_study
from gbm_ai.api.services.tumor_quantification import (
    TumorQuantificationServiceError,
    get_latest_tumor_quantification,
    run_tumor_quantification,
)
from gbm_ai.api.storage.local import LocalObjectStore


router = APIRouter(tags=["quantification"])


@router.post(
    "/studies/{study_uuid}/quantification/run",
    response_model=TumorQuantificationResponse,
    summary=(
        "Derive WT/TC/ET physical volumes and axial per-slice areas from the "
        "current validated 3D segmentation without anatomical localization"
    ),
)
def quantify_tumor(
    study_uuid: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db_session),
    storage: LocalObjectStore = Depends(get_object_store),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    try:
        return run_tumor_quantification(
            db,
            storage,
            study,
            request_id=getattr(request.state, "request_id", None),
            actor_type=AuditActorType.DEMO_USER,
        )
    except TumorQuantificationServiceError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )


@router.get(
    "/studies/{study_uuid}/quantification/result",
    response_model=TumorQuantificationResponse,
    summary="Read the current Phase 7 physical tumor quantification result",
)
def read_tumor_quantification(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
    storage: LocalObjectStore = Depends(get_object_store),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    try:
        return get_latest_tumor_quantification(db, storage, study)
    except TumorQuantificationServiceError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )
