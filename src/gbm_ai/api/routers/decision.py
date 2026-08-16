from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from gbm_ai.api.dependencies import get_db_session
from gbm_ai.api.models.analysis import Study
from gbm_ai.api.schemas.decision import DecisionFusionCurrentResponse, DecisionFusionResponse
from gbm_ai.api.services.decision_fusion import (
    DecisionFusionServiceError,
    fuse_study_decision,
    get_current_fused_decision,
)

router = APIRouter(tags=["decision-fusion"])


def _study_or_404(db: Session, study_uuid: uuid.UUID) -> Study:
    study = db.get(Study, study_uuid)
    if study is None:
        raise HTTPException(status_code=404, detail="study not found")
    return study


@router.post(
    "/studies/{study_uuid}/decision/fuse",
    response_model=DecisionFusionResponse,
)
def decision_fuse(
    study_uuid: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db_session),
):
    study = _study_or_404(db, study_uuid)
    try:
        return fuse_study_decision(
            db,
            study,
            request_id=getattr(request.state, "request_id", None),
        )
    except DecisionFusionServiceError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc)})


@router.get(
    "/studies/{study_uuid}/decision/current",
    response_model=DecisionFusionCurrentResponse,
)
def decision_current(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    study = _study_or_404(db, study_uuid)
    try:
        return get_current_fused_decision(db, study)
    except DecisionFusionServiceError as exc:
        raise HTTPException(status_code=404, detail={"code": exc.code, "message": str(exc)})
