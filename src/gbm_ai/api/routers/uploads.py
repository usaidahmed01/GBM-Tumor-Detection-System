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
from gbm_ai.api.dicom.deidentify import (
    DicomGroupingError,
    DicomModalityError,
    DicomPixelPrivacyRiskError,
    DicomProcessingError,
)
from gbm_ai.api.models.analysis import SourceFormat
from gbm_ai.api.models.audit import AuditActorType
from gbm_ai.api.schemas.format_detection import (
    FormatDetectionResponse,
    UploadAndDetectionResponse,
)
from gbm_ai.api.services.analysis_records import (
    StudyNotFoundError,
    get_study,
)
from gbm_ai.api.services.dicom_processing import (
    DicomStudyStateError,
    process_dicom_study,
)
from gbm_ai.api.services.format_detection import (
    StudyHasNoStoredObjectError,
    detect_and_update_study_format,
)
from gbm_ai.api.services.study_storage import (
    StudyStorageStateError,
    attach_study_source_object,
)
from gbm_ai.api.storage.local import LocalObjectStore, ObjectTooLargeError
from gbm_ai.api.upload.format_detection import (
    AmbiguousArchiveError,
    NonMRIInputError,
    UnsupportedInputError,
)
from gbm_ai.api.upload.intake import (
    ArchivePolicyError,
    EmptyUploadError,
    UploadIntakeError,
    UploadObjectTooLargeError,
    preflight_upload,
)

router = APIRouter(tags=["uploads"])


def _format_error_to_http(exc: Exception) -> HTTPException:
    if isinstance(exc, NonMRIInputError):
        return HTTPException(
            status_code=422,
            detail={
                "code": "NON_MRI_DICOM_MODALITY",
                "message": str(exc),
            },
        )
    if isinstance(exc, AmbiguousArchiveError):
        return HTTPException(
            status_code=422,
            detail={
                "code": "AMBIGUOUS_ARCHIVE_FORMAT",
                "message": str(exc),
            },
        )
    if isinstance(exc, UnsupportedInputError):
        return HTTPException(
            status_code=422,
            detail={
                "code": "UNSUPPORTED_OR_CORRUPT_INPUT",
                "message": str(exc),
            },
        )
    return HTTPException(
        status_code=422,
        detail={
            "code": "FORMAT_DETECTION_FAILED",
            "message": str(exc),
        },
    )


def _dicom_error_to_http(exc: Exception) -> HTTPException:
    if isinstance(exc, DicomPixelPrivacyRiskError):
        return HTTPException(
            status_code=422,
            detail={
                "code": "DICOM_PIXEL_PRIVACY_RISK",
                "message": str(exc),
                "action": "AI inference is blocked until pixel privacy is reviewed.",
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
    "/studies/{study_uuid}/upload",
    response_model=UploadAndDetectionResponse,
    summary=(
        "Store one upload, detect its content format, and prepare DICOM "
        "metadata-deidentified working copy when applicable"
    ),
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

    try:
        detected = detect_and_update_study_format(
            db,
            storage,
            study,
        )
    except (NonMRIInputError, AmbiguousArchiveError, UnsupportedInputError) as exc:
        raise _format_error_to_http(exc)

    deidentified_working_copy_created = False
    deidentified_sha256 = None
    dicom_series_count = None
    pixel_privacy_status = None

    if study.source_format == SourceFormat.DICOM:
        try:
            dicom_result = process_dicom_study(
                db,
                storage,
                study,
                request_id=getattr(request.state, "request_id", None),
                actor_type=AuditActorType.DEMO_USER,
            )
        except Exception as exc:
            raise _dicom_error_to_http(exc)

        deidentified_working_copy_created = True
        deidentified_sha256 = dicom_result["deidentified_sha256"]
        dicom_series_count = dicom_result["series_count"]
        pixel_privacy_status = dicom_result["pixel_privacy_status"]
        next_step = "mri_qc_and_sequence_detection"
    else:
        next_step = "mri_qc_and_capability_routing"

    return UploadAndDetectionResponse(
        study_uuid=study.id,
        study_status=study.status,
        source_format=study.source_format,
        modality=study.modality,
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
        parser=detected.parser,
        technical_metadata=detected.technical_metadata,
        deidentification_status=study.deidentification_status,
        deidentified_working_copy_created=(
            deidentified_working_copy_created
        ),
        deidentified_sha256=deidentified_sha256,
        dicom_series_count=dicom_series_count,
        pixel_privacy_status=pixel_privacy_status,
        next_step=next_step,
    )


@router.post(
    "/studies/{study_uuid}/detect-format",
    response_model=FormatDetectionResponse,
    summary="Detect format for an already stored study object",
)
def detect_existing_study_format(
    study_uuid: uuid.UUID,
    db: Session = Depends(get_db_session),
    storage: LocalObjectStore = Depends(get_object_store),
):
    try:
        study = get_study(db, study_uuid)
    except StudyNotFoundError:
        raise HTTPException(status_code=404, detail="study not found")

    try:
        detected = detect_and_update_study_format(
            db,
            storage,
            study,
        )
    except StudyHasNoStoredObjectError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except (NonMRIInputError, AmbiguousArchiveError, UnsupportedInputError) as exc:
        raise _format_error_to_http(exc)

    next_step = (
        "dicom_deidentification"
        if study.source_format == SourceFormat.DICOM
        else "mri_qc_and_capability_routing"
    )

    return FormatDetectionResponse(
        study_uuid=study.id,
        study_status=study.status,
        source_format=study.source_format,
        modality=study.modality,
        parser=detected.parser,
        technical_metadata=detected.technical_metadata,
        deidentification_status=study.deidentification_status,
        next_step=next_step,
    )
