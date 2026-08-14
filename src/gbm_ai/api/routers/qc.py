from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from gbm_ai.api.dependencies import get_db_session, get_object_store
from gbm_ai.api.models.analysis import StudyQCStatus
from gbm_ai.api.models.audit import AuditActorType
from gbm_ai.api.schemas.analysis import SeriesRead
from gbm_ai.api.schemas.qc import (
    SequenceConfirmationRequest,
    SequenceConfirmationResponse,
    StudyQCResponse,
)
from gbm_ai.api.services.analysis_records import (
    StudyNotFoundError,
    get_study,
)
from gbm_ai.api.services.study_qc import (
    InvalidSequenceConfirmationError,
    SeriesNotFoundError,
    StudyQCStateError,
    confirm_series_sequence,
    run_study_qc,
)
from gbm_ai.api.storage.local import LocalObjectStore

router = APIRouter(tags=["mri-qc"])


def _response_for(study_uuid: uuid.UUID, summary: dict) -> StudyQCResponse:
    status = StudyQCStatus(summary["qc_status"])
    if status == StudyQCStatus.FAIL:
        next_step = "unable_to_assess_manual_review"
    else:
        next_step = "capability_routing"

    return StudyQCResponse(
        study_uuid=study_uuid,
        qc_status=status,
        manual_review_required=bool(summary["manual_review_required"]),
        fail_reasons=list(summary["fail_reasons"]),
        partial_reasons=list(summary["partial_reasons"]),
        warnings=list(summary["warnings"]),
        checks=dict(summary["checks"]),
        inference_started=bool(summary.get("inference_started", False)),
        capability_routing_completed=bool(
            summary.get("capability_routing_completed", False)
        ),
        next_step=next_step,
    )


@router.post(
    "/studies/{study_uuid}/qc",
    response_model=StudyQCResponse,
    summary="Run MRI quality/suitability checks and DICOM sequence detection",
)
def run_qc(
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
        summary = run_study_qc(
            db,
            storage,
            study,
            request_id=getattr(request.state, "request_id", None),
            actor_type=AuditActorType.DEMO_USER,
        )
    except StudyQCStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return _response_for(study.id, summary)


@router.get(
    "/studies/{study_uuid}/qc",
    response_model=StudyQCResponse,
    summary="Read the latest MRI QC result",
)
def get_qc(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    if study.qc_status == StudyQCStatus.PENDING or not study.qc_summary:
        raise HTTPException(
            status_code=409,
            detail="MRI QC has not been completed or is stale",
        )

    return _response_for(study.id, study.qc_summary)


@router.put(
    "/series/{series_uuid}/sequence-confirmation",
    response_model=SequenceConfirmationResponse,
    summary="Record clinician confirmation/correction of a DICOM sequence",
)
def confirm_sequence(
    series_uuid: uuid.UUID,
    payload: SequenceConfirmationRequest,
    request: Request,
    db: Session = Depends(get_db_session),
):
    try:
        series = confirm_series_sequence(
            db,
            series_uuid,
            payload.sequence,
            request_id=getattr(request.state, "request_id", None),
            actor_type=AuditActorType.DEMO_USER,
        )
    except SeriesNotFoundError:
        raise HTTPException(status_code=404, detail="series not found")
    except InvalidSequenceConfirmationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return SequenceConfirmationResponse(
        series=SeriesRead.model_validate(series),
    )
