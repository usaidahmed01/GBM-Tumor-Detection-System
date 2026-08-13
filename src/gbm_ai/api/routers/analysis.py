from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from gbm_ai.api.dependencies import get_db_session
from gbm_ai.api.schemas.analysis import (
    AnalysisRunCreate,
    AnalysisRunRead,
    ModelVersionCreate,
    ModelVersionRead,
    StudyCreate,
    StudyRead,
)
from gbm_ai.api.services.analysis_records import (
    AssessmentNotFoundForStudyError,
    DuplicateModelVersionError,
    ModelRoleMismatchError,
    ModelVersionNotFoundError,
    StudyNotFoundError,
    create_analysis_run,
    create_model_version,
    create_study,
    get_analysis_run,
    get_study,
    list_model_versions,
)

router = APIRouter(tags=["analysis-records"])


@router.post(
    "/studies",
    response_model=StudyRead,
    status_code=status.HTTP_201_CREATED,
)
def study_create(
    payload: StudyCreate,
    db: Session = Depends(get_db_session),
):
    try:
        return create_study(db, payload)
    except AssessmentNotFoundForStudyError:
        raise HTTPException(status_code=404, detail="assessment not found")


@router.get("/studies/{study_uuid}", response_model=StudyRead)
def study_get(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    try:
        return get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")


@router.post(
    "/model-versions",
    response_model=ModelVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def model_version_create(
    payload: ModelVersionCreate,
    db: Session = Depends(get_db_session),
):
    try:
        return create_model_version(db, payload)
    except DuplicateModelVersionError:
        raise HTTPException(
            status_code=409,
            detail="model name/version already exists",
        )


@router.get(
    "/model-versions",
    response_model=list[ModelVersionRead],
)
def model_versions_list(db: Session = Depends(get_db_session)):
    return list_model_versions(db)


@router.post(
    "/analysis-runs",
    response_model=AnalysisRunRead,
    status_code=status.HTTP_201_CREATED,
)
def analysis_run_create(
    payload: AnalysisRunCreate,
    db: Session = Depends(get_db_session),
):
    try:
        return create_analysis_run(db, payload)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")
    except ModelVersionNotFoundError:
        raise HTTPException(status_code=404, detail="model version not found")
    except ModelRoleMismatchError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get(
    "/analysis-runs/{analysis_run_uuid}",
    response_model=AnalysisRunRead,
)
def analysis_run_get(
    analysis_run_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    try:
        return get_analysis_run(db, analysis_run_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="analysis run not found")
