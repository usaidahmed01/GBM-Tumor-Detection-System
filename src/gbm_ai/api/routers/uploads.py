from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from gbm_ai.api.config import Settings
from gbm_ai.api.dependencies import (
    get_app_settings,
    get_db_session,
    get_object_store,
)
from gbm_ai.api.models.audit import AuditActorType
from gbm_ai.api.schemas.upload import UploadIntakeResponse
from gbm_ai.api.services.analysis_records import (
    StudyNotFoundError,
    get_study,
)
from gbm_ai.api.services.study_storage import (
    StudyStorageStateError,
    attach_study_source_object,
)
from gbm_ai.api.storage.local import LocalObjectStore, ObjectTooLargeError
from gbm_ai.api.upload.intake import (
    ArchivePolicyError,
    EmptyUploadError,
    UploadIntakeError,
    UploadObjectTooLargeError,
    preflight_upload,
)

router = APIRouter(tags=["uploads"])


@router.post(
    "/studies/{study_uuid}/upload",
    response_model=UploadIntakeResponse,
    summary="Store one MRI upload for a study",
)
def upload_study_source(
    study_uuid: uuid.UUID,
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db_session),
    storage: LocalObjectStore = Depends(get_object_store),
    settings: Settings = Depends(get_app_settings),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    if study.storage_key is not None:
        raise HTTPException(
            status_code=409,
            detail="study already has a source upload",
        )

    try:
        file.file.seek(0)
        preflight = preflight_upload(
            file.file,
            max_object_bytes=settings.storage_max_object_bytes,
            max_archive_entries=settings.upload_max_archive_entries,
            max_archive_uncompressed_bytes=(
                settings.upload_max_archive_uncompressed_bytes
            ),
            max_archive_single_entry_bytes=(
                settings.upload_max_archive_single_entry_bytes
            ),
            max_archive_compression_ratio=(
                settings.upload_max_archive_compression_ratio
            ),
        )
    except (UploadObjectTooLargeError, ObjectTooLargeError) as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except (EmptyUploadError, ArchivePolicyError, UploadIntakeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        file.file.seek(0)
        stored = attach_study_source_object(
            db,
            storage,
            study,
            file.file,
            request_id=getattr(request.state, "request_id", None),
            actor_type=AuditActorType.DEMO_USER,
        )
    except ObjectTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except StudyStorageStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # Deliberately do not expose storage_key or original client filename.
    # Format remains pending until Phase 5 Step 2 parses actual content.
    return UploadIntakeResponse(
        study_uuid=study.id,
        study_status=study.status,
        source_format=study.source_format,
        stored_size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        upload_kind=preflight.upload_kind,
        archive_entry_count=preflight.archive_entry_count,
        archive_total_uncompressed_bytes=(
            preflight.archive_total_uncompressed_bytes
        ),
        archive_max_compression_ratio_observed=(
            preflight.archive_max_compression_ratio_observed
        ),
    )
