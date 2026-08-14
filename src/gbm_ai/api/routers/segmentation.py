from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from gbm_ai.api.dependencies import get_db_session, get_object_store
from gbm_ai.api.models.audit import AuditActorType
from gbm_ai.api.schemas.segmentation import (
    SegmentationModelGeometryResponse,
    SegmentationModelInputResponse,
    SegmentationPreflightResponse,
    SegmentationPreparationResponse,
)
from gbm_ai.api.services.analysis_records import (
    StudyNotFoundError,
    get_study,
)
from gbm_ai.api.services.segmentation_preflight import (
    SegmentationPreflightError,
    build_segmentation_preflight,
)
from gbm_ai.api.services.segmentation_model_geometry import (
    SegmentationModelGeometryPreparationError,
    get_segmentation_model_geometry,
    prepare_segmentation_model_geometry,
)
from gbm_ai.api.services.segmentation_model_input import (
    SegmentationModelInputPreparationError,
    get_segmentation_model_input,
    prepare_segmentation_model_input,
)
from gbm_ai.api.services.segmentation_volume_preparation import (
    SegmentationVolumePreparationError,
    get_segmentation_preparation,
    prepare_segmentation_volumes,
)
from gbm_ai.api.storage.local import LocalObjectStore


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
        raise HTTPException(status_code=404, detail="study not found")

    try:
        return build_segmentation_preflight(db, study)
    except SegmentationPreflightError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post(
    "/studies/{study_uuid}/segmentation/prepare-volumes",
    response_model=SegmentationPreparationResponse,
    summary=(
        "Load four eligible 3D MRI channels, normalize orientation, and "
        "validate inter-modality alignment without model inference"
    ),
)
def prepare_volumes(
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
        return prepare_segmentation_volumes(
            db,
            storage,
            study,
            request_id=getattr(request.state, "request_id", None),
            actor_type=AuditActorType.DEMO_USER,
        )
    except SegmentationVolumePreparationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )


@router.get(
    "/studies/{study_uuid}/segmentation/preparation",
    response_model=SegmentationPreparationResponse,
    summary="Read the latest Phase 6 volume-preparation/alignment state",
)
def read_preparation(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    try:
        return get_segmentation_preparation(study)
    except SegmentationVolumePreparationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )


@router.post(
    "/studies/{study_uuid}/segmentation/prepare-model-geometry",
    response_model=SegmentationModelGeometryResponse,
    summary=(
        "Rigidly register modalities when required and resample all four "
        "channels to the frozen 1 mm model geometry without inference"
    ),
)
def prepare_model_geometry(
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
        return prepare_segmentation_model_geometry(
            db,
            storage,
            study,
            request_id=getattr(request.state, "request_id", None),
            actor_type=AuditActorType.DEMO_USER,
        )
    except SegmentationModelGeometryPreparationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )


@router.get(
    "/studies/{study_uuid}/segmentation/model-geometry",
    response_model=SegmentationModelGeometryResponse,
    summary="Read the latest Phase 6 Step 3 model-geometry preprocessing state",
)
def read_model_geometry(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    try:
        return get_segmentation_model_geometry(study)
    except SegmentationModelGeometryPreparationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )

@router.post(
    "/studies/{study_uuid}/segmentation/prepare-model-input",
    response_model=SegmentationModelInputResponse,
    summary=(
        "Apply the frozen MONAI BraTS non-zero channel-wise intensity "
        "normalization and persist a protected four-channel model input "
        "without running SegResNet"
    ),
)
def prepare_model_input(
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
        return prepare_segmentation_model_input(
            db,
            storage,
            study,
            request_id=getattr(request.state, "request_id", None),
            actor_type=AuditActorType.DEMO_USER,
        )
    except SegmentationModelInputPreparationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )


@router.get(
    "/studies/{study_uuid}/segmentation/model-input",
    response_model=SegmentationModelInputResponse,
    summary="Read the latest Phase 6 Step 4 MONAI model-input preparation state",
)
def read_model_input(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    try:
        return get_segmentation_model_input(study)
    except SegmentationModelInputPreparationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )

