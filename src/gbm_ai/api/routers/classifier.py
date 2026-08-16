from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from gbm_ai.api.config import Settings
from gbm_ai.api.dependencies import get_app_settings, get_db_session, get_object_store
from gbm_ai.api.models.analysis import Study
from gbm_ai.api.schemas.classifier import ClassifierRunResponse, ClassifierRuntimeStatusResponse
from gbm_ai.api.services.classifier_runtime import (
    CLASSIFIER_RUNTIME_VERSION,
    ClassifierRuntimeError,
    classifier_runtime_status,
    run_classifier_for_study,
)
from gbm_ai.api.storage.local import LocalObjectStore

router = APIRouter(tags=["classifier-runtime"])


def _study_or_404(db: Session, study_uuid: uuid.UUID) -> Study:
    study = db.get(Study, study_uuid)
    if study is None:
        raise HTTPException(status_code=404, detail="study not found")
    return study


@router.get(
    "/classifier/runtime",
    response_model=ClassifierRuntimeStatusResponse,
    summary="Frozen standalone 2D classifier runtime readiness",
)
def runtime_status(
    settings: Settings = Depends(get_app_settings),
) -> ClassifierRuntimeStatusResponse:
    return ClassifierRuntimeStatusResponse(**classifier_runtime_status(settings))


@router.post(
    "/studies/{study_uuid}/classifier/run",
    response_model=ClassifierRunResponse,
    summary="Run the frozen standalone 2D classifier for one image study",
)
def run_classifier(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
    storage: LocalObjectStore = Depends(get_object_store),
    settings: Settings = Depends(get_app_settings),
) -> ClassifierRunResponse:
    study = _study_or_404(db, study_uuid)
    try:
        analysis = run_classifier_for_study(db, storage, settings, study)
    except ClassifierRuntimeError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc)})

    return ClassifierRunResponse(
        analysis_run_uuid=analysis.id,
        study_uuid=study.id,
        model_version_uuid=analysis.classifier_model_version_id,
        runtime_version=CLASSIFIER_RUNTIME_VERSION,
        raw_probability_gbm=analysis.raw_probability_gbm,
        calibrated_probability_gbm=analysis.calibrated_probability_gbm,
        decision_state=analysis.decision_state,
        qc_state=analysis.qc_state,
        safety_reason_codes=list(analysis.safety_reason_codes or []),
    )
