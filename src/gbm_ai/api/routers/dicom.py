from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from gbm_ai.api.dependencies import get_db_session, get_object_store
from gbm_ai.api.models.audit import AuditActorType
from gbm_ai.api.schemas.analysis import SeriesRead
from gbm_ai.api.schemas.dicom import (
    DicomDeidentificationResponse,
    DicomSeriesListResponse,
)
from gbm_ai.api.services.analysis_records import StudyNotFoundError, get_study
from gbm_ai.api.services.dicom_processing import (
    DicomStudyStateError,
    list_study_series,
    process_dicom_study,
)
from gbm_ai.api.storage.local import LocalObjectStore
from gbm_ai.api.dicom.deidentify import (
    DicomGroupingError,
    DicomModalityError,
    DicomPixelPrivacyRiskError,
    DicomProcessingError,
)

router = APIRouter(tags=["dicom"])


def dicom_processing_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DicomPixelPrivacyRiskError):
        return HTTPException(
            status_code=422,
            detail={
                "code": "DICOM_PIXEL_PRIVACY_RISK",
                "message": str(exc),
                "action": (
                    "Do not run AI inference. Pixel-level de-identification/"
                    "review is required."
                ),
            },
        )
    if isinstance(exc, DicomModalityError):
        return HTTPException(
            status_code=422,
            detail={
                "code": "DICOM_NON_MR_INSTANCE",
                "message": str(exc),
            },
        )
    if isinstance(exc, DicomGroupingError):
        return HTTPException(
            status_code=422,
            detail={
                "code": "DICOM_GROUPING_FAILED",
                "message": str(exc),
            },
        )
    if isinstance(exc, DicomProcessingError):
        return HTTPException(
            status_code=422,
            detail={
                "code": "DICOM_DEIDENTIFICATION_FAILED",
                "message": str(exc),
            },
        )
    if isinstance(exc, DicomStudyStateError):
        return HTTPException(status_code=409, detail=str(exc))

    return HTTPException(
        status_code=500,
        detail="unexpected DICOM processing failure",
    )


@router.post(
    "/studies/{study_uuid}/dicom/deidentify",
    response_model=DicomDeidentificationResponse,
    summary="Create a metadata-deidentified DICOM AI working copy",
)
def deidentify_dicom_study(
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
        result = process_dicom_study(
            db,
            storage,
            study,
            request_id=getattr(request.state, "request_id", None),
            actor_type=AuditActorType.DEMO_USER,
        )
    except Exception as exc:
        raise dicom_processing_http_error(exc)

    return DicomDeidentificationResponse(
        study_uuid=study.id,
        study_status=study.status,
        deidentification_status=study.deidentification_status,
        deidentified_sha256=result["deidentified_sha256"],
        deidentified_size_bytes=result["deidentified_size_bytes"],
        series_count=result["series_count"],
        instance_count=result["instance_count"],
        pixel_privacy_status=result["pixel_privacy_status"],
    )


@router.get(
    "/studies/{study_uuid}/series",
    response_model=DicomSeriesListResponse,
    summary="List de-identified DICOM series metadata",
)
def study_series_list(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    series = list_study_series(db, study)

    return DicomSeriesListResponse(
        study_uuid=study.id,
        series_count=len(series),
        series=[
            SeriesRead.model_validate(item)
            for item in series
        ],
    )
