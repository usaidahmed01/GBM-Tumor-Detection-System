from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from gbm_ai.api.dependencies import get_db_session
from gbm_ai.api.schemas.segmentation import (
    SegmentationPreflightResponse,
)
from gbm_ai.api.services.analysis_records import (
    StudyNotFoundError,
    get_study,
)
from gbm_ai.api.services.segmentation_preflight import (
    SegmentationPreflightError,
    build_segmentation_preflight,
)


router = APIRouter(tags=["segmentation"])


@router.post(
    "/studies/{study_uuid}/segmentation/preflight",
    response_model=SegmentationPreflightResponse,
    summary=(
        "Validate the frozen 3D segmentation input contract "
        "without running model inference"
    ),
)
def segmentation_preflight(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    try:
        study = get_study(db, study_uuid)

    except StudyNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="study not found",
        )

    try:
        return build_segmentation_preflight(
            db,
            study,
        )

    except SegmentationPreflightError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )