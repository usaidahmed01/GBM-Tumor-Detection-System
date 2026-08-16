from __future__ import annotations

import json
import uuid
from io import BytesIO

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import (
    AnalysisRun,
    AnalysisStatus,
    CapabilityRoutingStatus,
    SegmentationPreparationStatus,
    SourceFormat,
    Study,
)
from gbm_ai.api.models.audit import AuditAction, AuditActorType, AuditEntityType
from gbm_ai.api.models.quantification import TumorQuantification
from gbm_ai.api.models.segmentation import Segmentation, SegmentationReviewStatus, SegmentationStatus
from gbm_ai.api.quantification import (
    PHYSICAL_QUANTIFICATION_VERSION,
    PhysicalQuantificationError,
    load_and_validate_mask,
    measure_region,
    source_fingerprint,
    validate_physical_geometry,
)
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.services.current_segmentation import resolve_current_completed_segmentation
from gbm_ai.api.storage.local import LocalObjectStore


class TumorQuantificationServiceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _current_segmentation(db: Session, study: Study) -> tuple[AnalysisRun, Segmentation]:
    if study.source_format not in {SourceFormat.DICOM, SourceFormat.NIFTI}:
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_SOURCE_FORMAT_UNSUPPORTED",
            "physical tumor volume is available only for validated volumetric DICOM/NIfTI studies",
        )
    if study.capability_routing_status != CapabilityRoutingStatus.READY:
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_CAPABILITY_NOT_READY",
            "current capability routing is not ready for volumetric analysis",
        )
    if study.segmentation_preparation_status != SegmentationPreparationStatus.READY:
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_SEGMENTATION_PREPARATION_INVALID",
            "current segmentation preparation is not ready",
        )

    summary = dict(study.segmentation_preparation_summary or {})
    model_input = dict(summary.get("model_input") or {})
    current_input_checksum = str(model_input.get("checksum_sha256") or "")
    if model_input.get("status") != "ready" or len(current_input_checksum) != 64:
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_CURRENT_SEGMENTATION_NOT_READY",
            "a current completed 3D segmentation is required before physical quantification",
        )

    resolved = resolve_current_completed_segmentation(db, study, repair_summary=True)
    if resolved is None:
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_CURRENT_SEGMENTATION_MISSING",
            "current segmentation metadata could not be resolved from the database",
        )
    analysis, segmentation = resolved
    if segmentation.model_input_checksum_sha256 != current_input_checksum:
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_CURRENT_SEGMENTATION_MISSING",
            "current segmentation does not match the prepared model input",
        )
    if segmentation.review_status == SegmentationReviewStatus.REJECTED:
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_SEGMENTATION_REJECTED",
            "physical tumor volume must not be reported from a rejected segmentation",
        )
    return analysis, segmentation


def _mask_checksums(segmentation: Segmentation) -> dict[str, str]:
    return {
        "WT": segmentation.wt_checksum_sha256,
        "TC": segmentation.tc_checksum_sha256,
        "ET": segmentation.et_checksum_sha256,
        "LABELMAP": segmentation.labelmap_checksum_sha256,
    }


def _measurement_payload(region: str, quantification: TumorQuantification) -> dict:
    prefix = region.lower()
    return {
        "region": region,
        "voxel_count": int(getattr(quantification, f"{prefix}_voxel_count")),
        "volume_mm3": float(getattr(quantification, f"{prefix}_volume_mm3")),
        "volume_cm3": float(getattr(quantification, f"{prefix}_volume_cm3")),
        "max_axial_area_mm2": float(getattr(quantification, f"{prefix}_max_axial_area_mm2")),
        "max_axial_slice_index": getattr(quantification, f"{prefix}_max_axial_slice_index"),
        "axial_nonzero_slice_count": int(getattr(quantification, f"{prefix}_axial_nonzero_slice_count")),
    }


def quantification_to_response(
    study: Study,
    analysis: AnalysisRun,
    segmentation: Segmentation,
    quantification: TumorQuantification,
) -> dict:
    return {
        "version": PHYSICAL_QUANTIFICATION_VERSION,
        "status": "complete",
        "study_uuid": study.id,
        "analysis_run_uuid": analysis.id,
        "segmentation_uuid": segmentation.id,
        "quantification_uuid": quantification.id,
        "source_format": study.source_format.value,
        "source_review_status": segmentation.review_status.value,
        "clinician_modified": bool(segmentation.clinician_modified),
        "spatial_shape": list(quantification.spatial_shape),
        "affine_ras": list(quantification.affine_ras),
        "voxel_spacing_mm": list(quantification.voxel_spacing_mm),
        "voxel_volume_mm3": float(quantification.voxel_volume_mm3),
        "axial_pixel_area_mm2": float(quantification.axial_pixel_area_mm2),
        "primary_quantitative_region": "WT",
        "regions": [
            _measurement_payload("WT", quantification),
            _measurement_payload("TC", quantification),
            _measurement_payload("ET", quantification),
        ],
        "per_slice_area_artifact": {
            "storage_key": quantification.per_slice_area_storage_key,
            "checksum_sha256": quantification.per_slice_area_checksum_sha256,
            "size_bytes": quantification.per_slice_area_size_bytes,
            "format": "json",
            "plane": "axial_ras",
        },
        "measurement_basis": "3d_segmentation_plus_valid_spatial_metadata",
        "segmentation_is_gbm_diagnosis": False,
        "physical_volume_generated": True,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,
        "next_step": "phase7_step2_anatomical_localization",
    }


def _latest_matching_quantification(
    db: Session,
    *,
    segmentation_id: uuid.UUID,
    fingerprint: str,
) -> TumorQuantification | None:
    return db.scalar(
        select(TumorQuantification)
        .where(
            TumorQuantification.segmentation_id == segmentation_id,
            TumorQuantification.source_fingerprint_sha256 == fingerprint,
        )
        .order_by(TumorQuantification.created_at.desc())
        .limit(1)
    )


def run_tumor_quantification(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    *,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> dict:
    analysis, segmentation = _current_segmentation(db, study)

    expected_shape = tuple(int(v) for v in (segmentation.spatial_shape or []))
    if len(expected_shape) != 3 or any(v <= 0 for v in expected_shape):
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_SPATIAL_SHAPE_INVALID",
            "segmentation does not contain a valid 3D spatial shape",
        )

    try:
        expected_affine = np.asarray(segmentation.affine_ras, dtype=np.float64)
        geometry = validate_physical_geometry(expected_affine)
    except PhysicalQuantificationError as exc:
        raise TumorQuantificationServiceError(exc.code, str(exc)) from exc

    checksums = _mask_checksums(segmentation)
    fingerprint = source_fingerprint(
        segmentation_uuid=str(segmentation.id),
        spatial_shape=list(expected_shape),
        affine_ras=list(segmentation.affine_ras),
        mask_checksums=checksums,
    )
    existing = _latest_matching_quantification(
        db,
        segmentation_id=segmentation.id,
        fingerprint=fingerprint,
    )
    if existing is not None:
        try:
            existing_valid = storage.verify_checksum(
                existing.per_slice_area_storage_key,
                existing.per_slice_area_checksum_sha256,
            )
        except Exception:
            existing_valid = False
        if not existing_valid:
            raise TumorQuantificationServiceError(
                "PHYSICAL_QUANTIFICATION_EXISTING_ARTIFACT_INVALID",
                "existing per-slice quantification artifact failed checksum validation",
            )
        return quantification_to_response(study, analysis, segmentation, existing)

    # Verify the combined label map too, even though WT/TC/ET measurements use
    # the three binary masks directly.
    try:
        labelmap_valid = storage.verify_checksum(
            segmentation.labelmap_storage_key,
            segmentation.labelmap_checksum_sha256,
        )
    except Exception:
        labelmap_valid = False
    if not labelmap_valid:
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_LABELMAP_CHECKSUM_MISMATCH",
            "segmentation label map failed protected-storage checksum validation",
        )

    masks: dict[str, np.ndarray] = {}
    artifact_info = {
        "WT": (segmentation.wt_storage_key, segmentation.wt_checksum_sha256),
        "TC": (segmentation.tc_storage_key, segmentation.tc_checksum_sha256),
        "ET": (segmentation.et_storage_key, segmentation.et_checksum_sha256),
    }
    try:
        for region, (storage_key, checksum) in artifact_info.items():
            masks[region] = load_and_validate_mask(
                storage,
                storage_key=storage_key,
                checksum_sha256=checksum,
                expected_shape=expected_shape,
                expected_affine_ras=expected_affine,
            )
    except PhysicalQuantificationError as exc:
        raise TumorQuantificationServiceError(exc.code, str(exc)) from exc

    measurements = {}
    slice_payload: dict[str, object] = {
        "version": PHYSICAL_QUANTIFICATION_VERSION,
        "plane": "axial_ras",
        "spatial_shape": list(expected_shape),
        "voxel_spacing_mm": [round(float(v), 6) for v in geometry.voxel_spacing_mm],
        "voxel_volume_mm3": round(float(geometry.voxel_volume_mm3), 9),
        "axial_pixel_area_mm2": round(float(geometry.axial_pixel_area_mm2), 9),
        "regions": {},
    }
    for region in ("WT", "TC", "ET"):
        measurement, per_slice = measure_region(region, masks[region], geometry)
        expected_count = int((segmentation.voxel_counts or {}).get(region, -1))
        if expected_count != measurement.voxel_count:
            raise TumorQuantificationServiceError(
                "PHYSICAL_QUANTIFICATION_VOXEL_COUNT_MISMATCH",
                f"{region} mask foreground count no longer matches segmentation provenance",
            )
        measurements[region] = measurement
        slice_payload["regions"][region] = per_slice

    serialized = json.dumps(
        slice_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    storage_key = storage.generate_study_derived_key(
        study.id,
        "quantification_areas",
        suffix=".json",
    )
    stored = storage.put_stream(storage_key, BytesIO(serialized))

    def m(region: str):
        return measurements[region]

    quantification = TumorQuantification(
        segmentation_id=segmentation.id,
        quantification_version=PHYSICAL_QUANTIFICATION_VERSION,
        source_fingerprint_sha256=fingerprint,
        source_review_status=segmentation.review_status.value,
        source_clinician_modified=bool(segmentation.clinician_modified),
        source_mask_checksums=dict(checksums),
        spatial_shape=list(expected_shape),
        affine_ras=list(segmentation.affine_ras),
        voxel_spacing_mm=[round(float(v), 6) for v in geometry.voxel_spacing_mm],
        voxel_volume_mm3=round(float(geometry.voxel_volume_mm3), 9),
        axial_pixel_area_mm2=round(float(geometry.axial_pixel_area_mm2), 9),
        wt_voxel_count=m("WT").voxel_count,
        wt_volume_mm3=round(m("WT").volume_mm3, 6),
        wt_volume_cm3=round(m("WT").volume_cm3, 6),
        wt_max_axial_area_mm2=round(m("WT").max_axial_area_mm2, 6),
        wt_max_axial_slice_index=m("WT").max_axial_slice_index,
        wt_axial_nonzero_slice_count=m("WT").axial_nonzero_slice_count,
        tc_voxel_count=m("TC").voxel_count,
        tc_volume_mm3=round(m("TC").volume_mm3, 6),
        tc_volume_cm3=round(m("TC").volume_cm3, 6),
        tc_max_axial_area_mm2=round(m("TC").max_axial_area_mm2, 6),
        tc_max_axial_slice_index=m("TC").max_axial_slice_index,
        tc_axial_nonzero_slice_count=m("TC").axial_nonzero_slice_count,
        et_voxel_count=m("ET").voxel_count,
        et_volume_mm3=round(m("ET").volume_mm3, 6),
        et_volume_cm3=round(m("ET").volume_cm3, 6),
        et_max_axial_area_mm2=round(m("ET").max_axial_area_mm2, 6),
        et_max_axial_slice_index=m("ET").max_axial_slice_index,
        et_axial_nonzero_slice_count=m("ET").axial_nonzero_slice_count,
        per_slice_area_storage_key=stored.storage_key,
        per_slice_area_checksum_sha256=stored.sha256,
        per_slice_area_size_bytes=stored.size_bytes,
        physical_volume_generated=True,
        anatomical_localization_generated=False,
        clinical_validation_claimed=False,
    )
    db.add(quantification)
    segmentation.physical_volume_generated = True

    current = dict(study.segmentation_preparation_summary or {})
    inference = dict(current.get("inference") or {})
    if str(inference.get("segmentation_uuid") or "") == str(segmentation.id):
        inference["physical_volume_generated"] = True
        inference["anatomical_localization_generated"] = False
        current["inference"] = inference
    current["physical_volume_generated"] = True
    current["anatomical_localization_generated"] = False
    current["quantification"] = {
        "version": PHYSICAL_QUANTIFICATION_VERSION,
        "status": "complete",
        "segmentation_uuid": str(segmentation.id),
        "source_fingerprint_sha256": fingerprint,
        "physical_volume_generated": True,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,
    }
    current["next_step"] = "phase7_step2_anatomical_localization"
    study.segmentation_preparation_summary = current

    db.flush()
    record_audit_event(
        db,
        action=AuditAction.QUANTIFICATION_COMPLETED,
        entity_type=AuditEntityType.QUANTIFICATION,
        entity_uuid=quantification.id,
        actor_type=actor_type,
        actor_id=actor_id,
        request_id=request_id,
        technical_context={
            "status": "complete",
            "operation": "physical_tumor_quantification",
            "result": "wt_tc_et_volume_and_axial_area_generated",
        },
        commit=False,
    )
    try:
        db.commit()
    except Exception:
        db.rollback()
        try:
            if storage.exists(stored.storage_key):
                storage.delete(stored.storage_key)
        except Exception:
            pass
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_DATABASE_PERSIST_FAILED",
            "physical measurements were discarded because quantification metadata could not be persisted",
        )

    db.refresh(quantification)
    db.refresh(segmentation)
    return quantification_to_response(study, analysis, segmentation, quantification)


def get_latest_tumor_quantification(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
) -> dict:
    analysis, segmentation = _current_segmentation(db, study)
    fingerprint = source_fingerprint(
        segmentation_uuid=str(segmentation.id),
        spatial_shape=list(segmentation.spatial_shape),
        affine_ras=list(segmentation.affine_ras),
        mask_checksums=_mask_checksums(segmentation),
    )
    quantification = _latest_matching_quantification(
        db,
        segmentation_id=segmentation.id,
        fingerprint=fingerprint,
    )
    if quantification is None:
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_RESULT_NOT_AVAILABLE",
            "no current physical tumor quantification exists for this segmentation",
        )

    source_artifacts = [
        (segmentation.wt_storage_key, segmentation.wt_checksum_sha256),
        (segmentation.tc_storage_key, segmentation.tc_checksum_sha256),
        (segmentation.et_storage_key, segmentation.et_checksum_sha256),
        (segmentation.labelmap_storage_key, segmentation.labelmap_checksum_sha256),
    ]
    try:
        source_artifacts_valid = all(
            storage.verify_checksum(key, checksum)
            for key, checksum in source_artifacts
        )
    except Exception:
        source_artifacts_valid = False
    if not source_artifacts_valid:
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_SOURCE_ARTIFACT_INVALID",
            "current segmentation artifacts failed protected-storage checksum validation",
        )

    try:
        artifact_valid = storage.verify_checksum(
            quantification.per_slice_area_storage_key,
            quantification.per_slice_area_checksum_sha256,
        )
    except Exception:
        artifact_valid = False
    if not artifact_valid:
        raise TumorQuantificationServiceError(
            "PHYSICAL_QUANTIFICATION_RESULT_ARTIFACT_INVALID",
            "current quantification artifact failed protected-storage checksum validation",
        )
    return quantification_to_response(study, analysis, segmentation, quantification)
