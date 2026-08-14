from __future__ import annotations

from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import (
    SegmentationPreparationStatus,
    SourceFormat,
    Study,
)
from gbm_ai.api.models.audit import (
    AuditAction,
    AuditActorType,
    AuditEntityType,
)
from gbm_ai.api.segmentation.contract import (
    SEGMENTATION_CONTRACT_VERSION,
    SEGMENTATION_INPUT_CHANNEL_ORDER,
    SEGMENTATION_REFERENCE_SPACING_MM,
)
from gbm_ai.api.segmentation.volume_loading import (
    SegmentationVolumeLoadError,
    load_channel_volume,
    summarize_loaded_volume,
    validate_channel_alignment,
)
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.services.segmentation_preflight import (
    SegmentationPreflightError,
    build_segmentation_preflight,
)
from gbm_ai.api.storage.local import LocalObjectStore


SEGMENTATION_PREPARATION_VERSION = "phase6_step2_volume_preparation_v1"


class SegmentationVolumePreparationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _verify_source_integrity(
    storage: LocalObjectStore,
    study: Study,
) -> tuple[str, str]:
    if study.source_format == SourceFormat.DICOM:
        key = study.deidentified_storage_key
        checksum = study.deidentified_checksum_sha256
    elif study.source_format == SourceFormat.NIFTI:
        key = study.storage_key
        checksum = study.checksum_sha256
    else:
        raise SegmentationVolumePreparationError(
            "SEGMENTATION_SOURCE_FORMAT_UNSUPPORTED",
            "Phase 6 volume preparation accepts only DICOM or NIfTI studies",
        )

    if not key or not checksum:
        raise SegmentationVolumePreparationError(
            "SEGMENTATION_SOURCE_REFERENCE_INCOMPLETE",
            "protected volumetric source reference/checksum is incomplete",
        )

    try:
        checksum_ok = storage.verify_checksum(key, checksum)
    except Exception as exc:
        raise SegmentationVolumePreparationError(
            "SEGMENTATION_SOURCE_OBJECT_UNAVAILABLE",
            "protected volumetric source object could not be opened for integrity verification",
        ) from exc

    if not checksum_ok:
        raise SegmentationVolumePreparationError(
            "SEGMENTATION_SOURCE_CHECKSUM_MISMATCH",
            "protected volumetric source checksum does not match the database record",
        )

    return key, checksum


def _failure_summary(study: Study, code: str) -> dict:
    return {
        "version": SEGMENTATION_PREPARATION_VERSION,
        "contract_version": SEGMENTATION_CONTRACT_VERSION,
        "study_uuid": str(study.id),
        "source_format": study.source_format.value,
        "status": "failed",
        "failure_reason_code": code,
        "channel_order": list(SEGMENTATION_INPUT_CHANNEL_ORDER),
        "model_execution_started": False,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
        "registration_performed": False,
        "reference_geometry_resampling_performed": False,
        "clinical_validation_claimed": False,
    }


def _persist_failure(
    db: Session,
    study: Study,
    code: str,
) -> None:
    study.segmentation_preparation_status = SegmentationPreparationStatus.FAILED
    study.segmentation_preparation_summary = _failure_summary(study, code)
    db.commit()
    db.refresh(study)


def prepare_segmentation_volumes(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    *,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> dict:
    """
    Load the four eligible 3D MRI channels, normalize orientation to RAS, and
    validate whether their voxel grids are already aligned.

    This step deliberately does not register/resample modalities and does not
    load or execute MONAI/SegResNet.
    """
    try:
        preflight = build_segmentation_preflight(db, study)
    except SegmentationPreflightError as exc:
        raise SegmentationVolumePreparationError(
            "SEGMENTATION_PREFLIGHT_NOT_READY",
            str(exc),
        ) from exc

    _, source_checksum = _verify_source_integrity(storage, study)

    existing = dict(study.segmentation_preparation_summary or {})
    if (
        existing.get("version") == SEGMENTATION_PREPARATION_VERSION
        and existing.get("source_checksum_sha256") == source_checksum
        and existing.get("status") in {"ready", "registration_required"}
        and existing.get("model_execution_started") is False
    ):
        return existing

    channel_summaries: list[dict] = []

    try:
        for channel in preflight["channels"]:
            loaded = load_channel_volume(
                db,
                storage,
                study,
                channel,
            )
            channel_summaries.append(
                summarize_loaded_volume(loaded)
            )
            # The large voxel array is intentionally released after geometry
            # extraction so Step 2 stays practical on CPU/RAM-constrained hosts.
            del loaded
    except SegmentationVolumeLoadError as exc:
        db.rollback()
        _persist_failure(db, study, exc.code)
        raise SegmentationVolumePreparationError(
            exc.code,
            str(exc),
        ) from exc

    alignment = validate_channel_alignment(channel_summaries)
    aligned = bool(alignment["aligned"])

    status = (
        SegmentationPreparationStatus.READY
        if aligned
        else SegmentationPreparationStatus.REGISTRATION_REQUIRED
    )

    next_step = (
        "phase6_step3_model_geometry_preprocessing"
        if aligned
        else "phase6_step3_registration_and_model_geometry_preprocessing"
    )

    summary = {
        "version": SEGMENTATION_PREPARATION_VERSION,
        "contract_version": SEGMENTATION_CONTRACT_VERSION,
        "study_uuid": str(study.id),
        "source_format": study.source_format.value,
        "source_checksum_sha256": source_checksum,
        "status": status.value,
        "channel_order": list(SEGMENTATION_INPUT_CHANNEL_ORDER),
        "channels": channel_summaries,
        "alignment": alignment,
        "canonical_orientation": "RAS",
        "reference_spacing_target_mm": list(
            SEGMENTATION_REFERENCE_SPACING_MM
        ),
        "registration_performed": False,
        "reference_geometry_resampling_performed": False,
        "model_execution_started": False,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,
        "next_step": next_step,
    }

    study.segmentation_preparation_status = status
    study.segmentation_preparation_summary = summary

    record_audit_event(
        db,
        action=AuditAction.SEGMENTATION_PREPARATION_COMPLETED,
        entity_type=AuditEntityType.STUDY,
        entity_uuid=study.id,
        actor_type=actor_type,
        actor_id=actor_id,
        request_id=request_id,
        technical_context={
            "operation": "segmentation_volume_preparation",
            "status": status.value,
            "source_format": study.source_format.value,
            "result": "complete",
        },
        commit=False,
    )

    db.commit()
    db.refresh(study)
    return summary


def get_segmentation_preparation(study: Study) -> dict:
    summary = dict(study.segmentation_preparation_summary or {})
    if (
        study.segmentation_preparation_status
        == SegmentationPreparationStatus.PENDING
        or not summary
    ):
        raise SegmentationVolumePreparationError(
            "SEGMENTATION_PREPARATION_NOT_RUN",
            "Phase 6 volume preparation has not been completed",
        )
    return summary
