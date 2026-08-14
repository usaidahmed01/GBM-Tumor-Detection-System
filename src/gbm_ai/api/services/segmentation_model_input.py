from __future__ import annotations

from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import SegmentationPreparationStatus, Study
from gbm_ai.api.models.audit import AuditAction, AuditActorType, AuditEntityType
from gbm_ai.api.segmentation.model_geometry import MODEL_GEOMETRY_VERSION
from gbm_ai.api.segmentation.model_input import (
    MODEL_INPUT_VERSION,
    SegmentationModelInputError,
    build_normalized_model_input,
    model_input_summary,
    persist_normalized_model_input,
)
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.services.segmentation_model_geometry import (
    SegmentationModelGeometryPreparationError,
    get_segmentation_model_geometry,
)
from gbm_ai.api.storage.local import LocalObjectStore


class SegmentationModelInputPreparationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _existing_model_input_is_valid(storage: LocalObjectStore, summary: dict) -> bool:
    model_input = dict(summary.get("model_input") or {})
    if model_input.get("version") != MODEL_INPUT_VERSION or model_input.get("status") != "ready":
        return False
    key = model_input.get("storage_key")
    checksum = model_input.get("checksum_sha256")
    if not key or not checksum:
        return False
    try:
        return storage.verify_checksum(key, checksum)
    except Exception:
        return False


def _persist_failure(db: Session, study: Study, code: str) -> None:
    current = dict(study.segmentation_preparation_summary or {})
    current["model_input"] = {
        "version": MODEL_INPUT_VERSION,
        "status": "failed",
        "failure_reason_code": code,
        "intensity_normalization_performed": False,
        "crop_pad_performed": False,
        "model_execution_started": False,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,
    }
    current["model_execution_started"] = False
    current["segmentation_generated"] = False
    current["physical_volume_generated"] = False
    current["anatomical_localization_generated"] = False
    study.segmentation_preparation_summary = current
    study.segmentation_preparation_status = SegmentationPreparationStatus.FAILED
    db.commit()
    db.refresh(study)


def prepare_segmentation_model_input(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    *,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> dict:
    """Create the frozen MONAI-compatible normalized four-channel input without inference."""
    try:
        model_geometry = get_segmentation_model_geometry(study)
    except SegmentationModelGeometryPreparationError as exc:
        raise SegmentationModelInputPreparationError(
            "SEGMENTATION_STEP3_NOT_READY",
            str(exc),
        ) from exc

    if model_geometry.get("version") != MODEL_GEOMETRY_VERSION or model_geometry.get("status") != "ready":
        raise SegmentationModelInputPreparationError(
            "SEGMENTATION_STEP3_NOT_READY",
            "Phase 6 Step 3 model geometry must be ready before MONAI model-input preparation",
        )
    if model_geometry.get("model_execution_started") is not False:
        raise SegmentationModelInputPreparationError(
            "SEGMENTATION_STEP3_INVALID_EXECUTION_STATE",
            "model execution must remain stopped before Step 4 preprocessing",
        )

    current = dict(study.segmentation_preparation_summary or {})
    if _existing_model_input_is_valid(storage, current):
        return dict(current["model_input"])

    stored_key: str | None = None
    try:
        image, affine_ras, channel_stats = build_normalized_model_input(storage, model_geometry)
        stored = persist_normalized_model_input(
            storage,
            study.id,
            image=image,
            affine_ras=affine_ras,
        )
        stored_key = stored.storage_key
        model_input = model_input_summary(
            stored,
            image=image,
            affine_ras=affine_ras,
            channel_stats=channel_stats,
        )
    except SegmentationModelInputError as exc:
        db.rollback()
        if stored_key:
            try:
                if storage.exists(stored_key):
                    storage.delete(stored_key)
            except Exception:
                pass
        _persist_failure(db, study, exc.code)
        raise SegmentationModelInputPreparationError(exc.code, str(exc)) from exc
    except Exception as exc:
        db.rollback()
        if stored_key:
            try:
                if storage.exists(stored_key):
                    storage.delete(stored_key)
            except Exception:
                pass
        code = "SEGMENTATION_MODEL_INPUT_UNEXPECTED_FAILURE"
        _persist_failure(db, study, code)
        raise SegmentationModelInputPreparationError(
            code,
            "unexpected MONAI model-input preparation failure",
        ) from exc

    current = dict(study.segmentation_preparation_summary or {})
    current["model_input"] = model_input
    current["intensity_normalization_performed"] = True
    current["crop_pad_performed"] = False
    current["model_execution_started"] = False
    current["segmentation_generated"] = False
    current["physical_volume_generated"] = False
    current["anatomical_localization_generated"] = False
    current["next_step"] = model_input["next_step"]
    study.segmentation_preparation_summary = current
    study.segmentation_preparation_status = SegmentationPreparationStatus.READY

    record_audit_event(
        db,
        action=AuditAction.SEGMENTATION_PREPARATION_COMPLETED,
        entity_type=AuditEntityType.STUDY,
        entity_uuid=study.id,
        actor_type=actor_type,
        actor_id=actor_id,
        request_id=request_id,
        technical_context={
            "operation": "segmentation_monai_model_input_preparation",
            "status": "ready",
            "result": "complete",
            "model_name": "brats_mri_segmentation",
            "model_version": "0.5.4",
            "preprocessing_version": model_input["preprocessing_version"],
        },
        commit=False,
    )

    db.commit()
    db.refresh(study)
    return model_input


def get_segmentation_model_input(study: Study) -> dict:
    current = dict(study.segmentation_preparation_summary or {})
    model_input = dict(current.get("model_input") or {})
    if not model_input:
        raise SegmentationModelInputPreparationError(
            "SEGMENTATION_MODEL_INPUT_NOT_RUN",
            "Phase 6 Step 4 MONAI model-input preparation has not been completed",
        )
    return model_input
