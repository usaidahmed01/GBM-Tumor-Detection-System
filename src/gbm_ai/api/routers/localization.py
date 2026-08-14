from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from gbm_ai.api.config import Settings
from gbm_ai.api.dependencies import get_app_settings, get_db_session, get_object_store
from gbm_ai.api.models.audit import AuditActorType
from gbm_ai.api.schemas.localization import AnatomicalLocalizationResponse
from gbm_ai.api.services.analysis_records import StudyNotFoundError, get_study
from gbm_ai.api.services.anatomical_localization import (
    AnatomicalLocalizationServiceError,
    get_latest_anatomical_localization,
    run_anatomical_localization,
)
from gbm_ai.api.storage.local import LocalObjectStore


router = APIRouter(tags=["localization"])


@router.post(
    "/studies/{study_uuid}/localization/run",
    response_model=AnatomicalLocalizationResponse,
    summary="Register current WT mask to MNI152 and derive atlas-based hemisphere/region",
)
def localize_tumor(
    study_uuid: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db_session),
    storage: LocalObjectStore = Depends(get_object_store),
    settings: Settings = Depends(get_app_settings),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")
    try:
        return run_anatomical_localization(
            db,
            storage,
            study,
            atlas_root=settings.localization_atlas_root_resolved,
            request_id=getattr(request.state, "request_id", None),
            actor_type=AuditActorType.DEMO_USER,
        )
    except AnatomicalLocalizationServiceError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )


@router.get(
    "/studies/{study_uuid}/localization/result",
    response_model=AnatomicalLocalizationResponse,
    summary="Read the current atlas-based anatomical localization result",
)
def read_localization(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
    storage: LocalObjectStore = Depends(get_object_store),
    settings: Settings = Depends(get_app_settings),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")
    try:
        return get_latest_anatomical_localization(
            db,
            storage,
            study,
            atlas_root=settings.localization_atlas_root_resolved,
        )
    except AnatomicalLocalizationServiceError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )
