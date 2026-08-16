from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from gbm_ai.api.dependencies import get_db_session
from gbm_ai.api.models.analysis import Study
from gbm_ai.api.schemas.report import ReportFinalizeRequest, ReportFinalizedResponse, ReportPreviewResponse
from gbm_ai.api.services.clinical_report import (
    ClinicalReportError,
    finalize_report,
    get_current_report,
    preview_report,
    report_response,
)

router = APIRouter(tags=["clinical-report"])


def _study_or_404(db: Session, study_uuid: uuid.UUID) -> Study:
    study = db.get(Study, study_uuid)
    if study is None:
        raise HTTPException(status_code=404, detail="study not found")
    return study


@router.get(
    "/studies/{study_uuid}/report/preview",
    response_model=ReportPreviewResponse,
)
def report_preview(study_uuid: uuid.UUID, db: Session = Depends(get_db_session)):
    study = _study_or_404(db, study_uuid)
    try:
        return preview_report(db, study)
    except ClinicalReportError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc)})


@router.post(
    "/studies/{study_uuid}/report/finalize",
    response_model=ReportFinalizedResponse,
)
def report_finalize(
    study_uuid: uuid.UUID,
    payload: ReportFinalizeRequest,
    request: Request,
    db: Session = Depends(get_db_session),
):
    study = _study_or_404(db, study_uuid)
    try:
        row = finalize_report(
            db,
            study,
            clinician_name=payload.clinician_name,
            clinician_comment=payload.clinician_comment,
            request_id=getattr(request.state, "request_id", None),
        )
        return report_response(row)
    except ClinicalReportError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc)})


@router.get(
    "/studies/{study_uuid}/report/current",
    response_model=ReportFinalizedResponse,
)
def report_current(study_uuid: uuid.UUID, db: Session = Depends(get_db_session)):
    study = _study_or_404(db, study_uuid)
    try:
        return report_response(get_current_report(db, study))
    except ClinicalReportError as exc:
        raise HTTPException(status_code=404, detail={"code": exc.code, "message": str(exc)})
