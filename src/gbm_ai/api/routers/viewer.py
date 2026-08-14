from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from gbm_ai.api.config import Settings
from gbm_ai.api.dependencies import get_app_settings, get_db_session, get_object_store
from gbm_ai.api.models.audit import AuditAction, AuditActorType, AuditEntityType
from gbm_ai.api.schemas.viewer import ClinicalViewerManifestResponse
from gbm_ai.api.services.analysis_records import StudyNotFoundError, get_study
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.services.clinical_viewer import (
    ClinicalViewerServiceError,
    build_clinical_viewer_manifest,
    open_verified_viewer_asset,
    resolve_viewer_asset,
)
from gbm_ai.api.storage.local import LocalObjectStore


router = APIRouter(tags=["viewer"])


@router.get(
    "/studies/{study_uuid}/viewer/manifest",
    response_model=ClinicalViewerManifestResponse,
    summary=(
        "Build the Phase 8 clinical-viewer manifest using only approved, "
        "checksum-bound derived MRI and segmentation artifacts"
    ),
)
def clinical_viewer_manifest(
    study_uuid: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db_session),
    settings: Settings = Depends(get_app_settings),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    try:
        payload = build_clinical_viewer_manifest(
            db,
            study,
            api_prefix=settings.api_v1_prefix,
        )
        record_audit_event(
            db,
            action=AuditAction.STUDY_VIEWED,
            entity_type=AuditEntityType.STUDY,
            entity_uuid=study.id,
            actor_type=AuditActorType.DEMO_USER,
            request_id=getattr(request.state, "request_id", None),
            technical_context={
                "operation": "clinical_viewer_manifest",
                "source_format": study.source_format.value,
                "status": "ready",
            },
        )
        return payload
    except ClinicalViewerServiceError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )


@router.get(
    "/studies/{study_uuid}/viewer/assets/{asset_alias}",
    summary=(
        "Stream one approved clinical-viewer asset after server-side "
        "checksum verification without exposing an object-storage path"
    ),
)
def clinical_viewer_asset(
    study_uuid: uuid.UUID,
    asset_alias: str,
    request: Request,
    db: Session = Depends(get_db_session),
    storage: LocalObjectStore = Depends(get_object_store),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    try:
        asset = resolve_viewer_asset(db, study, asset_alias=asset_alias)
        opened = open_verified_viewer_asset(storage, asset)
        record_audit_event(
            db,
            action=AuditAction.OBJECT_DOWNLOADED,
            entity_type=AuditEntityType.STUDY,
            entity_uuid=study.id,
            actor_type=AuditActorType.DEMO_USER,
            request_id=getattr(request.state, "request_id", None),
            technical_context={
                "operation": f"clinical_viewer_asset:{asset.alias}",
                "source_format": study.source_format.value,
                "size_bytes": asset.size_bytes,
                "sha256": asset.checksum_sha256,
            },
        )
    except ClinicalViewerServiceError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )

    def iterator():
        try:
            while True:
                chunk = opened.stream.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            opened.stream.close()

    return StreamingResponse(
        iterator(),
        media_type=asset.media_type,
        headers={
            "Content-Disposition": f'inline; filename="{asset.filename}"',
            "Content-Length": str(asset.size_bytes),
            "ETag": f'"sha256-{asset.checksum_sha256}"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
