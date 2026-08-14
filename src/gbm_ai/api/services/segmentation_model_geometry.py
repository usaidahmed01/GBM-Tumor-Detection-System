from __future__ import annotations

from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import SegmentationPreparationStatus, Study
from gbm_ai.api.models.audit import AuditAction, AuditActorType, AuditEntityType
from gbm_ai.api.segmentation.contract import (
    SEGMENTATION_INPUT_CHANNEL_ORDER,
    SEGMENTATION_REFERENCE_SPACING_MM,
)
from gbm_ai.api.segmentation.model_geometry import (
    MODEL_GEOMETRY_VERSION,
    MODEL_REFERENCE_SEQUENCE,
    SegmentationModelGeometryError,
    create_isotropic_reference,
    geometries_match,
    identity_registration_summary,
    loaded_volume_to_sitk,
    persist_resampled_nifti,
    register_rigid_to_reference,
    resample_to_reference,
    sitk_affine_ras,
)
from gbm_ai.api.segmentation.volume_loading import (
    SegmentationVolumeLoadError,
    load_channel_volume,
)
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.services.segmentation_preflight import build_segmentation_preflight
from gbm_ai.api.services.segmentation_volume_preparation import (
    SEGMENTATION_PREPARATION_VERSION,
    SegmentationVolumePreparationError,
    prepare_segmentation_volumes,
)
from gbm_ai.api.storage.local import LocalObjectStore, ObjectNotFoundError


class SegmentationModelGeometryPreparationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _existing_model_geometry_is_valid(storage: LocalObjectStore, summary: dict) -> bool:
    model_geometry = dict(summary.get("model_geometry") or {})
    if model_geometry.get("version") != MODEL_GEOMETRY_VERSION:
        return False
    if model_geometry.get("status") != "ready":
        return False
    artifacts = list(model_geometry.get("channels") or [])
    if len(artifacts) != 4:
        return False

    try:
        for item in artifacts:
            key = item.get("storage_key")
            checksum = item.get("checksum_sha256")
            if not key or not checksum or not storage.verify_checksum(key, checksum):
                return False
    except Exception:
        return False
    return True


def _delete_artifacts_best_effort(storage: LocalObjectStore, stored_keys: list[str]) -> None:
    for key in stored_keys:
        try:
            if storage.exists(key):
                storage.delete(key)
        except Exception:
            pass


def _failure_model_geometry(code: str) -> dict:
    return {
        "version": MODEL_GEOMETRY_VERSION,
        "status": "failed",
        "failure_reason_code": code,
        "reference_sequence": MODEL_REFERENCE_SEQUENCE,
        "channel_order": list(SEGMENTATION_INPUT_CHANNEL_ORDER),
        "target_spacing_mm": list(SEGMENTATION_REFERENCE_SPACING_MM),
        "registration_performed": False,
        "reference_geometry_resampling_performed": False,
        "intensity_normalization_performed": False,
        "crop_pad_performed": False,
        "model_execution_started": False,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,
    }


def _persist_failure(db: Session, study: Study, code: str) -> None:
    current = dict(study.segmentation_preparation_summary or {})
    if current.get("version") != SEGMENTATION_PREPARATION_VERSION:
        current = {
            "version": SEGMENTATION_PREPARATION_VERSION,
            "study_uuid": str(study.id),
            "status": "failed",
            "model_execution_started": False,
        }
    current["model_geometry"] = _failure_model_geometry(code)
    current["model_execution_started"] = False
    current["segmentation_generated"] = False
    current["physical_volume_generated"] = False
    current["anatomical_localization_generated"] = False
    study.segmentation_preparation_summary = current
    db.commit()
    db.refresh(study)


def prepare_segmentation_model_geometry(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    *,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> dict:
    """
    Register channels to T1C only when Step 2 found geometry mismatch, then
    resample all four channels into one frozen 1 mm reference geometry.

    No intensity normalization, crop/pad, MONAI model loading, AnalysisRun, or
    segmentation inference occurs in this step.
    """
    try:
        preflight = build_segmentation_preflight(db, study)
        step2 = prepare_segmentation_volumes(db, storage, study)
    except (ValueError, SegmentationVolumePreparationError) as exc:
        raise SegmentationModelGeometryPreparationError(
            "SEGMENTATION_STEP2_NOT_READY",
            str(exc),
        ) from exc

    if study.segmentation_preparation_status not in {
        SegmentationPreparationStatus.READY,
        SegmentationPreparationStatus.REGISTRATION_REQUIRED,
    }:
        raise SegmentationModelGeometryPreparationError(
            "SEGMENTATION_STEP2_NOT_READY",
            "Phase 6 Step 2 must be ready or registration_required before model-geometry preprocessing",
        )

    if step2.get("version") != SEGMENTATION_PREPARATION_VERSION:
        raise SegmentationModelGeometryPreparationError(
            "SEGMENTATION_STEP2_VERSION_MISMATCH",
            "stored Phase 6 volume-preparation version is not compatible with Step 3",
        )

    if step2.get("model_execution_started") is not False:
        raise SegmentationModelGeometryPreparationError(
            "SEGMENTATION_STEP2_INVALID_EXECUTION_STATE",
            "model execution must remain stopped before Step 3 preprocessing",
        )

    if _existing_model_geometry_is_valid(storage, step2):
        return dict(step2["model_geometry"])

    channel_plan = {
        str(item["sequence"]): item
        for item in preflight["channels"]
    }
    if set(channel_plan) != set(SEGMENTATION_INPUT_CHANNEL_ORDER):
        raise SegmentationModelGeometryPreparationError(
            "SEGMENTATION_CHANNEL_PLAN_INVALID",
            "preflight channel plan no longer contains exactly T1C/T1/T2/FLAIR",
        )

    stored_keys: list[str] = []
    channel_outputs: list[dict] = []
    registration_any = False

    try:
        reference_volume = load_channel_volume(
            db,
            storage,
            study,
            channel_plan[MODEL_REFERENCE_SEQUENCE],
        )
        reference_image = loaded_volume_to_sitk(reference_volume)
        model_reference = create_isotropic_reference(reference_image)
        model_reference_affine_ras = sitk_affine_ras(model_reference)

        for sequence in SEGMENTATION_INPUT_CHANNEL_ORDER:
            if sequence == MODEL_REFERENCE_SEQUENCE:
                moving_volume = reference_volume
                moving_image = reference_image
                registration_summary = identity_registration_summary()
                transform = None
            else:
                moving_volume = load_channel_volume(
                    db,
                    storage,
                    study,
                    channel_plan[sequence],
                )
                moving_image = loaded_volume_to_sitk(moving_volume)

                if geometries_match(reference_volume, moving_volume):
                    registration_summary = identity_registration_summary()
                    transform = None
                else:
                    transform, registration_summary = register_rigid_to_reference(
                        reference_image,
                        moving_image,
                    )
                    registration_any = True

            resampled = resample_to_reference(
                moving_image,
                model_reference,
                transform=transform,
            )
            stored, artifact = persist_resampled_nifti(
                storage,
                study.id,
                sequence=sequence,
                image=resampled,
            )
            stored_keys.append(stored.storage_key)
            artifact["registration"] = registration_summary.as_dict()
            channel_outputs.append(artifact)

            if sequence != MODEL_REFERENCE_SEQUENCE:
                del moving_volume
                del moving_image
            del resampled

        reference_shape = list(model_reference.GetSize())
        reference_spacing = [float(v) for v in model_reference.GetSpacing()]

    except (SegmentationVolumeLoadError, SegmentationModelGeometryError) as exc:
        db.rollback()
        _delete_artifacts_best_effort(storage, stored_keys)
        code = getattr(exc, "code", "SEGMENTATION_MODEL_GEOMETRY_FAILED")
        _persist_failure(db, study, code)
        raise SegmentationModelGeometryPreparationError(code, str(exc)) from exc
    except Exception as exc:
        db.rollback()
        _delete_artifacts_best_effort(storage, stored_keys)
        code = "SEGMENTATION_MODEL_GEOMETRY_UNEXPECTED_FAILURE"
        _persist_failure(db, study, code)
        raise SegmentationModelGeometryPreparationError(
            code,
            "unexpected model-geometry preprocessing failure",
        ) from exc

    model_geometry = {
        "version": MODEL_GEOMETRY_VERSION,
        "status": "ready",
        "reference_sequence": MODEL_REFERENCE_SEQUENCE,
        "channel_order": list(SEGMENTATION_INPUT_CHANNEL_ORDER),
        "target_spacing_mm": list(SEGMENTATION_REFERENCE_SPACING_MM),
        "target_shape": [int(v) for v in reference_shape],
        "target_affine_ras": [
            [round(float(value), 6) for value in row]
            for row in model_reference_affine_ras
        ],
        "channels": channel_outputs,
        "registration_performed": registration_any,
        "registration_method": (
            "rigid_euler3d_mattes_mutual_information"
            if registration_any
            else "not_required"
        ),
        "registration_quality_gate": (
            "automated_geometric_sanity_pass"
            if registration_any
            else "not_required"
        ),
        "reference_geometry_resampling_performed": True,
        "interpolation": "linear",
        "intensity_normalization_performed": False,
        "crop_pad_performed": False,
        "model_execution_started": False,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,
        "next_step": "phase6_step4_monai_bundle_and_inference_preprocessing",
    }

    current = dict(study.segmentation_preparation_summary or step2)
    current["model_geometry"] = model_geometry
    current["registration_performed"] = registration_any
    current["reference_geometry_resampling_performed"] = True
    current["model_execution_started"] = False
    current["segmentation_generated"] = False
    current["physical_volume_generated"] = False
    current["anatomical_localization_generated"] = False
    current["next_step"] = model_geometry["next_step"]
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
            "operation": "segmentation_model_geometry_preparation",
            "status": "ready",
            "result": "complete",
        },
        commit=False,
    )

    db.commit()
    db.refresh(study)
    return model_geometry


def get_segmentation_model_geometry(study: Study) -> dict:
    current = dict(study.segmentation_preparation_summary or {})
    model_geometry = dict(current.get("model_geometry") or {})
    if not model_geometry:
        raise SegmentationModelGeometryPreparationError(
            "SEGMENTATION_MODEL_GEOMETRY_NOT_RUN",
            "Phase 6 Step 3 model-geometry preprocessing has not been completed",
        )
    return model_geometry
