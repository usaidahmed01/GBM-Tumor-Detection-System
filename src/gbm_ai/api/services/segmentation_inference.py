from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import (
    AnalysisRun,
    AnalysisStatus,
    CapabilityRoutingStatus,
    DecisionState,
    ModelRole,
    ModelVersion,
    QCState,
    SegmentationPreparationStatus,
    Study,
    StudyQCStatus,
    StudyStatus,
)
from gbm_ai.api.models.audit import AuditAction, AuditActorType, AuditEntityType
from gbm_ai.api.models.segmentation import (
    Segmentation,
    SegmentationReviewStatus,
    SegmentationStatus,
    SegmentationJob,
)
from gbm_ai.api.segmentation.bundle_runtime import (
    BUNDLE_MODEL_SHA256,
    BUNDLE_NAME,
    BUNDLE_OUTPUT_THRESHOLD,
    BUNDLE_OVERLAP,
    BUNDLE_ROI_SIZE,
    BUNDLE_VERSION,
)
from gbm_ai.api.segmentation.inference import (
    SEGMENTATION_INFERENCE_VERSION,
    SegmentationExecutionResult,
    SegmentationInferenceError,
    execute_and_persist_segmentation,
)
from gbm_ai.api.segmentation.model_input import (
    MODEL_INPUT_PREPROCESSING_VERSION,
    MODEL_INPUT_VERSION,
)
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.storage.local import LocalObjectStore


class SegmentationInferenceServiceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _require_execution_gate(study: Study) -> dict:
    if study.status != StudyStatus.READY_FOR_ANALYSIS:
        raise SegmentationInferenceServiceError(
            "SEGMENTATION_STUDY_NOT_READY",
            "study is not ready for analysis",
        )
    if study.capability_routing_status != CapabilityRoutingStatus.READY:
        raise SegmentationInferenceServiceError(
            "SEGMENTATION_CAPABILITY_ROUTING_NOT_READY",
            "Phase 5 capability routing is no longer ready",
        )
    capabilities = dict((study.capability_summary or {}).get("capabilities") or {})
    segmentation_capability = dict(capabilities.get("three_d_segmentation") or {})
    if segmentation_capability.get("state") != "eligible" or segmentation_capability.get("input_eligible") is not True:
        raise SegmentationInferenceServiceError(
            "SEGMENTATION_CAPABILITY_NOT_ELIGIBLE",
            "Phase 5 no longer marks 3D segmentation as eligible",
        )
    if study.qc_status not in {StudyQCStatus.PASS, StudyQCStatus.PARTIAL}:
        raise SegmentationInferenceServiceError(
            "SEGMENTATION_QC_NOT_READY",
            "MRI QC no longer permits 3D segmentation",
        )
    if study.segmentation_preparation_status != SegmentationPreparationStatus.READY:
        raise SegmentationInferenceServiceError(
            "SEGMENTATION_PREPARATION_NOT_READY",
            "Phase 6 preprocessing is not ready",
        )
    current = dict(study.segmentation_preparation_summary or {})
    model_input = dict(current.get("model_input") or {})
    if model_input.get("version") != MODEL_INPUT_VERSION or model_input.get("status") != "ready":
        raise SegmentationInferenceServiceError(
            "SEGMENTATION_MODEL_INPUT_NOT_READY",
            "Phase 6 Step 4 model input is not ready",
        )
    if current.get("physical_volume_generated") is not False or current.get("anatomical_localization_generated") is not False:
        raise SegmentationInferenceServiceError(
            "SEGMENTATION_PREPARATION_STATE_INVALID",
            "Phase 7 outputs must not exist before Phase 6 segmentation inference",
        )
    return model_input


def require_segmentation_execution_gate(study: Study) -> dict:
    """Public Phase 6 execution gate shared by sync and background paths."""
    return _require_execution_gate(study)


def _get_or_create_model_version(db: Session) -> ModelVersion:
    existing = db.scalar(
        select(ModelVersion).where(
            ModelVersion.model_name == BUNDLE_NAME,
            ModelVersion.version == BUNDLE_VERSION,
        )
    )
    if existing is not None:
        if (
            existing.role != ModelRole.SEGMENTATION
            or str(existing.weights_checksum_sha256 or "").lower() != BUNDLE_MODEL_SHA256
            or existing.architecture != "SegResNet"
            or existing.preprocessing_version != MODEL_INPUT_PREPROCESSING_VERSION
            or existing.threshold_version != "brats_0.5.4_sigmoid_threshold_0.5"
        ):
            raise SegmentationInferenceServiceError(
                "SEGMENTATION_MODEL_REGISTRY_CONFLICT",
                "existing segmentation model registry entry conflicts with the frozen bundle",
            )
        return existing

    model_version = ModelVersion(
        model_name=BUNDLE_NAME,
        version=BUNDLE_VERSION,
        role=ModelRole.SEGMENTATION,
        architecture="SegResNet",
        weights_checksum_sha256=BUNDLE_MODEL_SHA256,
        code_version=SEGMENTATION_INFERENCE_VERSION,
        preprocessing_version=MODEL_INPUT_PREPROCESSING_VERSION,
        threshold_version="brats_0.5.4_sigmoid_threshold_0.5",
        calibration_version=None,
        license_source_notes=(
            "MONAI BraTS MRI Segmentation bundle 0.5.4; Apache-2.0 software bundle. "
            "Research/academic prototype component; not a diagnostic validation claim."
        ),
        is_active=True,
    )
    db.add(model_version)
    db.flush()
    record_audit_event(
        db,
        action=AuditAction.MODEL_VERSION_REGISTERED,
        entity_type=AuditEntityType.MODEL_VERSION,
        entity_uuid=model_version.id,
        actor_type=AuditActorType.SYSTEM,
        technical_context={
            "status": "registered",
            "model_name": BUNDLE_NAME,
            "model_version": BUNDLE_VERSION,
            "model_role": "segmentation",
            "architecture": "SegResNet",
            "preprocessing_version": MODEL_INPUT_PREPROCESSING_VERSION,
            "threshold_version": "brats_0.5.4_sigmoid_threshold_0.5",
        },
        commit=False,
    )
    return model_version


def _latest_segmentation_for_study(
    db: Session,
    study_id: uuid.UUID,
    *,
    model_input_checksum_sha256: str | None = None,
) -> Segmentation | None:
    statement = (
        select(Segmentation)
        .join(AnalysisRun, AnalysisRun.id == Segmentation.analysis_run_id)
        .where(
            AnalysisRun.study_id == study_id,
            AnalysisRun.status == AnalysisStatus.COMPLETE,
            Segmentation.status == SegmentationStatus.GENERATED,
            Segmentation.inference_version == SEGMENTATION_INFERENCE_VERSION,
            Segmentation.weights_checksum_sha256 == BUNDLE_MODEL_SHA256,
        )
    )
    if model_input_checksum_sha256 is not None:
        statement = statement.where(
            Segmentation.model_input_checksum_sha256
            == model_input_checksum_sha256
        )
    return db.scalar(
        statement.order_by(Segmentation.created_at.desc()).limit(1)
    )


def _artifact_refs(segmentation: Segmentation) -> list[tuple[str, str]]:
    return [
        (segmentation.tc_storage_key, segmentation.tc_checksum_sha256),
        (segmentation.wt_storage_key, segmentation.wt_checksum_sha256),
        (segmentation.et_storage_key, segmentation.et_checksum_sha256),
        (segmentation.labelmap_storage_key, segmentation.labelmap_checksum_sha256),
    ]


def _existing_result_is_valid(storage: LocalObjectStore, segmentation: Segmentation) -> bool:
    try:
        return all(storage.verify_checksum(key, checksum) for key, checksum in _artifact_refs(segmentation))
    except Exception:
        return False


def _mask_payload(prefix: str, segmentation: Segmentation) -> dict:
    return {
        "storage_key": getattr(segmentation, f"{prefix}_storage_key"),
        "checksum_sha256": getattr(segmentation, f"{prefix}_checksum_sha256"),
        "size_bytes": getattr(segmentation, f"{prefix}_size_bytes"),
        "foreground_voxels": int((segmentation.voxel_counts or {}).get(prefix.upper(), 0)),
    }


def segmentation_to_response(db: Session, segmentation: Segmentation) -> dict:
    analysis = db.get(AnalysisRun, segmentation.analysis_run_id)
    if analysis is None:
        raise SegmentationInferenceServiceError(
            "SEGMENTATION_ANALYSIS_RUN_MISSING",
            "segmentation result references a missing analysis run",
        )
    label_nonzero = int((segmentation.voxel_counts or {}).get("LABELMAP_NONZERO", 0))
    return {
        "version": SEGMENTATION_INFERENCE_VERSION,
        "status": "complete",
        "analysis_run_uuid": analysis.id,
        "segmentation_uuid": segmentation.id,
        "model_name": segmentation.bundle_name,
        "model_version": segmentation.bundle_version,
        "model_weights_sha256": segmentation.weights_checksum_sha256,
        "model_input_checksum_sha256": segmentation.model_input_checksum_sha256,
        "preprocessing_version": segmentation.preprocessing_version,
        "device": segmentation.device,
        "amp_enabled": bool(segmentation.amp_enabled),
        "roi_size": list(segmentation.roi_size),
        "overlap": float(segmentation.overlap),
        "threshold": float(segmentation.threshold),
        "spatial_shape": list(segmentation.spatial_shape),
        "affine_ras": list(segmentation.affine_ras),
        "output_channel_order": ["TC", "WT", "ET"],
        "tc_mask": _mask_payload("tc", segmentation),
        "wt_mask": _mask_payload("wt", segmentation),
        "et_mask": _mask_payload("et", segmentation),
        "brats_labelmap": {
            "storage_key": segmentation.labelmap_storage_key,
            "checksum_sha256": segmentation.labelmap_checksum_sha256,
            "size_bytes": segmentation.labelmap_size_bytes,
            "foreground_voxels": label_nonzero,
        },
        "voxel_counts": {
            "TC": int((segmentation.voxel_counts or {}).get("TC", 0)),
            "WT": int((segmentation.voxel_counts or {}).get("WT", 0)),
            "ET": int((segmentation.voxel_counts or {}).get("ET", 0)),
        },
        "runtime_seconds": segmentation.runtime_seconds,
        "review_status": segmentation.review_status.value,
        "clinician_modified": bool(segmentation.clinician_modified),
        "decision_state": analysis.decision_state.value,
        "segmentation_is_gbm_diagnosis": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,
        "background_execution_implemented": True,
        "next_step": "phase6_complete",
    }


def _persist_failure_state(db: Session, study: Study, analysis: AnalysisRun, code: str) -> None:
    now = datetime.now(timezone.utc)
    analysis.status = AnalysisStatus.FAILED
    analysis.completed_at = now
    analysis.safety_reason_codes = [code]
    current = dict(study.segmentation_preparation_summary or {})
    current["inference"] = {
        "version": SEGMENTATION_INFERENCE_VERSION,
        "status": "failed",
        "failure_reason_code": code,
        "model_execution_started": True,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,
    }
    current["model_execution_started"] = True
    current["segmentation_generated"] = False
    current["physical_volume_generated"] = False
    current["anatomical_localization_generated"] = False
    study.segmentation_preparation_summary = current


def _delete_execution_artifacts(
    storage: LocalObjectStore,
    result: SegmentationExecutionResult,
) -> None:
    for artifact in (result.tc, result.wt, result.et, result.labelmap):
        try:
            if storage.exists(artifact.storage_key):
                storage.delete(artifact.storage_key)
        except Exception:
            pass


def run_segmentation_inference(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    *,
    bundle_dir: Path,
    device_preference: str,
    max_spatial_voxels: int,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
    background_job: SegmentationJob | None = None,
) -> dict:
    model_input = _require_execution_gate(study)

    current_input_checksum = str(model_input.get("checksum_sha256") or "")
    if len(current_input_checksum) != 64:
        raise SegmentationInferenceServiceError(
            "SEGMENTATION_MODEL_INPUT_CHECKSUM_INVALID",
            "Step 4 model input is missing its immutable SHA-256 binding",
        )

    existing = _latest_segmentation_for_study(
        db,
        study.id,
        model_input_checksum_sha256=current_input_checksum,
    )
    if existing is not None:
        if not _existing_result_is_valid(storage, existing):
            raise SegmentationInferenceServiceError(
                "SEGMENTATION_EXISTING_RESULT_ARTIFACT_INVALID",
                "existing segmentation result failed protected-storage checksum validation",
            )
        return segmentation_to_response(db, existing)

    model_version = _get_or_create_model_version(db)
    analysis = AnalysisRun(
        study_id=study.id,
        classifier_model_version_id=None,
        segmentation_model_version_id=model_version.id,
        status=AnalysisStatus.RUNNING,
        qc_state=QCState.PASS if study.qc_status == StudyQCStatus.PASS else QCState.REVIEW,
        decision_state=DecisionState.PENDING,
        safety_reason_codes=[],
        started_at=datetime.now(timezone.utc),
    )
    db.add(analysis)
    db.flush()
    if background_job is not None:
        if background_job.study_id != study.id:
            raise SegmentationInferenceServiceError(
                "SEGMENTATION_BACKGROUND_JOB_STUDY_MISMATCH",
                "background segmentation job is bound to a different study",
            )
        if background_job.model_input_checksum_sha256 != current_input_checksum:
            raise SegmentationInferenceServiceError(
                "SEGMENTATION_BACKGROUND_JOB_INPUT_MISMATCH",
                "background segmentation job is bound to a different model input",
            )
        background_job.analysis_run_id = analysis.id
    record_audit_event(
        db,
        action=AuditAction.ANALYSIS_STARTED,
        entity_type=AuditEntityType.ANALYSIS_RUN,
        entity_uuid=analysis.id,
        actor_type=actor_type,
        actor_id=actor_id,
        request_id=request_id,
        technical_context={
            "status": "running",
            "operation": "segmentation_inference",
            "model_name": BUNDLE_NAME,
            "model_version": BUNDLE_VERSION,
            "model_role": "segmentation",
            "preprocessing_version": MODEL_INPUT_PREPROCESSING_VERSION,
        },
        commit=False,
    )
    db.commit()
    db.refresh(analysis)

    try:
        result: SegmentationExecutionResult = execute_and_persist_segmentation(
            storage,
            study.id,
            model_input=model_input,
            bundle_dir=bundle_dir,
            device_preference=device_preference,
            max_spatial_voxels=max_spatial_voxels,
        )
    except SegmentationInferenceError as exc:
        db.rollback()
        study = db.get(Study, study.id)
        analysis = db.get(AnalysisRun, analysis.id)
        if study is not None and analysis is not None:
            _persist_failure_state(db, study, analysis, exc.code)
            record_audit_event(
                db,
                action=AuditAction.ANALYSIS_FAILED,
                entity_type=AuditEntityType.ANALYSIS_RUN,
                entity_uuid=analysis.id,
                actor_type=actor_type,
                actor_id=actor_id,
                request_id=request_id,
                technical_context={
                    "status": "failed",
                    "operation": "segmentation_inference",
                    "model_name": BUNDLE_NAME,
                    "model_version": BUNDLE_VERSION,
                    "error_type": exc.code,
                },
                commit=False,
            )
            db.commit()
        raise SegmentationInferenceServiceError(exc.code, str(exc)) from exc
    except Exception as exc:
        db.rollback()
        study = db.get(Study, study.id)
        analysis = db.get(AnalysisRun, analysis.id)
        code = "SEGMENTATION_INFERENCE_UNEXPECTED_FAILURE"
        if study is not None and analysis is not None:
            _persist_failure_state(db, study, analysis, code)
            record_audit_event(
                db,
                action=AuditAction.ANALYSIS_FAILED,
                entity_type=AuditEntityType.ANALYSIS_RUN,
                entity_uuid=analysis.id,
                actor_type=actor_type,
                actor_id=actor_id,
                request_id=request_id,
                technical_context={
                    "status": "failed",
                    "operation": "segmentation_inference",
                    "model_name": BUNDLE_NAME,
                    "model_version": BUNDLE_VERSION,
                    "error_type": code,
                },
                commit=False,
            )
            db.commit()
        raise SegmentationInferenceServiceError(
            code,
            "unexpected guarded segmentation inference failure",
        ) from exc

    study = db.get(Study, study.id)
    analysis = db.get(AnalysisRun, analysis.id)
    if study is None or analysis is None:
        _delete_execution_artifacts(storage, result)
        raise SegmentationInferenceServiceError(
            "SEGMENTATION_PERSISTENCE_CONTEXT_LOST",
            "study or analysis run disappeared before segmentation persistence",
        )

    segmentation = Segmentation(
        # Allocate the UUID before writing it into study JSON. SQLAlchemy column
        # defaults are otherwise applied only at flush/INSERT time.
        id=uuid.uuid4(),
        analysis_run_id=analysis.id,
        status=SegmentationStatus.GENERATED,
        model_input_checksum_sha256=str(model_input["checksum_sha256"]),
        inference_version=SEGMENTATION_INFERENCE_VERSION,
        preprocessing_version=MODEL_INPUT_PREPROCESSING_VERSION,
        bundle_name=BUNDLE_NAME,
        bundle_version=BUNDLE_VERSION,
        weights_checksum_sha256=BUNDLE_MODEL_SHA256,
        device=result.device,
        amp_enabled=result.amp_enabled,
        roi_size=list(BUNDLE_ROI_SIZE),
        overlap=BUNDLE_OVERLAP,
        threshold=BUNDLE_OUTPUT_THRESHOLD,
        spatial_shape=list(result.spatial_shape),
        affine_ras=[
            [round(float(value), 6) for value in row]
            for row in result.affine_ras
        ],
        tc_storage_key=result.tc.storage_key,
        tc_checksum_sha256=result.tc.checksum_sha256,
        tc_size_bytes=result.tc.size_bytes,
        wt_storage_key=result.wt.storage_key,
        wt_checksum_sha256=result.wt.checksum_sha256,
        wt_size_bytes=result.wt.size_bytes,
        et_storage_key=result.et.storage_key,
        et_checksum_sha256=result.et.checksum_sha256,
        et_size_bytes=result.et.size_bytes,
        labelmap_storage_key=result.labelmap.storage_key,
        labelmap_checksum_sha256=result.labelmap.checksum_sha256,
        labelmap_size_bytes=result.labelmap.size_bytes,
        voxel_counts={
            **result.voxel_counts,
            "LABELMAP_NONZERO": result.labelmap.foreground_voxels,
        },
        runtime_seconds=round(float(result.runtime_seconds), 3),
        review_status=SegmentationReviewStatus.UNREVIEWED,
        clinician_modified=False,
        physical_volume_generated=False,
        anatomical_localization_generated=False,
        clinical_validation_claimed=False,
    )
    db.add(segmentation)
    analysis.status = AnalysisStatus.COMPLETE
    analysis.completed_at = datetime.now(timezone.utc)
    analysis.decision_state = DecisionState.PENDING

    current = dict(study.segmentation_preparation_summary or {})
    current["inference"] = {
        "version": SEGMENTATION_INFERENCE_VERSION,
        "status": "complete",
        "analysis_run_uuid": str(analysis.id),
        "segmentation_uuid": str(segmentation.id),
        "model_name": BUNDLE_NAME,
        "model_version": BUNDLE_VERSION,
        "model_weights_sha256": BUNDLE_MODEL_SHA256,
        "model_input_checksum_sha256": str(model_input["checksum_sha256"]),
        "output_channel_order": ["TC", "WT", "ET"],
        "review_status": "unreviewed",
        "model_execution_started": True,
        "segmentation_generated": True,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,
    }
    current["model_execution_started"] = True
    current["segmentation_generated"] = True
    current["physical_volume_generated"] = False
    current["anatomical_localization_generated"] = False
    current["next_step"] = "phase6_complete"
    study.segmentation_preparation_summary = current
    study.segmentation_preparation_status = SegmentationPreparationStatus.READY

    db.flush()
    record_audit_event(
        db,
        action=AuditAction.ANALYSIS_COMPLETED,
        entity_type=AuditEntityType.ANALYSIS_RUN,
        entity_uuid=analysis.id,
        actor_type=actor_type,
        actor_id=actor_id,
        request_id=request_id,
        technical_context={
            "status": "complete",
            "operation": "segmentation_inference",
            "model_name": BUNDLE_NAME,
            "model_version": BUNDLE_VERSION,
            "model_role": "segmentation",
            "preprocessing_version": MODEL_INPUT_PREPROCESSING_VERSION,
            "result": "wt_tc_et_generated",
        },
        commit=False,
    )
    try:
        db.commit()
    except Exception:
        db.rollback()
        _delete_execution_artifacts(storage, result)
        persisted_study = db.get(Study, study.id)
        persisted_analysis = db.get(AnalysisRun, analysis.id)
        code = "SEGMENTATION_RESULT_DATABASE_PERSIST_FAILED"
        if persisted_study is not None and persisted_analysis is not None:
            _persist_failure_state(db, persisted_study, persisted_analysis, code)
            try:
                record_audit_event(
                    db,
                    action=AuditAction.ANALYSIS_FAILED,
                    entity_type=AuditEntityType.ANALYSIS_RUN,
                    entity_uuid=persisted_analysis.id,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    request_id=request_id,
                    technical_context={
                        "status": "failed",
                        "operation": "segmentation_inference",
                        "model_name": BUNDLE_NAME,
                        "model_version": BUNDLE_VERSION,
                        "error_type": code,
                    },
                    commit=False,
                )
                db.commit()
            except Exception:
                db.rollback()
        raise SegmentationInferenceServiceError(
            code,
            "generated masks were discarded because segmentation metadata could not be persisted",
        )
    db.refresh(segmentation)
    return segmentation_to_response(db, segmentation)


def get_latest_segmentation_result(db: Session, study: Study) -> dict:
    current = dict(study.segmentation_preparation_summary or {})
    model_input = dict(current.get("model_input") or {})
    current_checksum = str(model_input.get("checksum_sha256") or "")
    if (
        study.segmentation_preparation_status != SegmentationPreparationStatus.READY
        or model_input.get("status") != "ready"
        or len(current_checksum) != 64
    ):
        raise SegmentationInferenceServiceError(
            "SEGMENTATION_RESULT_STALE_OR_PREPARATION_INVALIDATED",
            "current Phase 6 preparation was invalidated; an older segmentation result must not be served as current",
        )

    segmentation = _latest_segmentation_for_study(
        db,
        study.id,
        model_input_checksum_sha256=current_checksum,
    )
    if segmentation is None:
        raise SegmentationInferenceServiceError(
            "SEGMENTATION_RESULT_NOT_AVAILABLE",
            "no completed 3D segmentation result exists for the current prepared model input",
        )
    return segmentation_to_response(db, segmentation)
