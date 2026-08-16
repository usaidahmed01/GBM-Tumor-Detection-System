from __future__ import annotations

import hashlib
import shutil
import tempfile
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import Study
from gbm_ai.api.models.audit import AuditAction, AuditActorType, AuditEntityType
from gbm_ai.api.models.segmentation import (
    Segmentation,
    SegmentationReviewAction,
    SegmentationReviewRevision,
    SegmentationReviewStatus,
)
from gbm_ai.api.services.anatomical_localization import (
    AnatomicalLocalizationServiceError,
    run_anatomical_localization,
)
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.services.clinical_viewer import _require_current_segmentation
from gbm_ai.api.services.tumor_quantification import (
    TumorQuantificationServiceError,
    run_tumor_quantification,
)
from gbm_ai.api.storage.local import LocalObjectStore


SEGMENTATION_REVIEW_VERSION = "phase8_step3_clinician_mask_review_v1"
ALLOWED_LABEL_VALUES = frozenset({0, 1, 2, 4})


class SegmentationReviewServiceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _artifact_snapshot(segmentation: Segmentation) -> dict[str, dict[str, object]]:
    return {
        "WT": {
            "storage_key": segmentation.wt_storage_key,
            "checksum_sha256": segmentation.wt_checksum_sha256,
            "size_bytes": int(segmentation.wt_size_bytes),
        },
        "TC": {
            "storage_key": segmentation.tc_storage_key,
            "checksum_sha256": segmentation.tc_checksum_sha256,
            "size_bytes": int(segmentation.tc_size_bytes),
        },
        "ET": {
            "storage_key": segmentation.et_storage_key,
            "checksum_sha256": segmentation.et_checksum_sha256,
            "size_bytes": int(segmentation.et_size_bytes),
        },
        "LABELMAP": {
            "storage_key": segmentation.labelmap_storage_key,
            "checksum_sha256": segmentation.labelmap_checksum_sha256,
            "size_bytes": int(segmentation.labelmap_size_bytes),
        },
    }


def _public_checksums(snapshot: dict[str, dict[str, object]]) -> dict[str, str]:
    return {
        name: str(item["checksum_sha256"])
        for name, item in snapshot.items()
    }


def _next_revision_number(db: Session, segmentation_id: uuid.UUID) -> int:
    latest = db.scalar(
        select(func.max(SegmentationReviewRevision.revision_number)).where(
            SegmentationReviewRevision.segmentation_id == segmentation_id
        )
    )
    return int(latest or 0) + 1


def _normalized_note(note: str | None) -> str | None:
    value = str(note or "").strip()
    if not value:
        return None
    if len(value) > 1000:
        raise SegmentationReviewServiceError(
            "SEGMENTATION_REVIEW_NOTE_TOO_LONG",
            "review note must be 1000 characters or fewer",
        )
    return value


def _invalidate_downstream(study: Study, segmentation: Segmentation, *, reason: str) -> None:
    segmentation.physical_volume_generated = False
    segmentation.anatomical_localization_generated = False

    current = dict(study.segmentation_preparation_summary or {})
    inference = dict(current.get("inference") or {})
    if str(inference.get("segmentation_uuid") or "") == str(segmentation.id):
        inference["physical_volume_generated"] = False
        inference["anatomical_localization_generated"] = False
        current["inference"] = inference
    current["physical_volume_generated"] = False
    current["anatomical_localization_generated"] = False
    current["quantification"] = {
        "status": "stale",
        "reason": reason,
        "segmentation_uuid": str(segmentation.id),
    }
    current["localization"] = {
        "status": "stale",
        "reason": reason,
        "segmentation_uuid": str(segmentation.id),
    }
    current["next_step"] = "phase8_step3_clinician_mask_review_and_correction"
    study.segmentation_preparation_summary = current


def _mark_rejected(study: Study, segmentation: Segmentation) -> None:
    _invalidate_downstream(study, segmentation, reason="segmentation_rejected_by_clinician")
    current = dict(study.segmentation_preparation_summary or {})
    current["quantification"] = {
        "status": "blocked",
        "reason": "segmentation_rejected_by_clinician",
        "segmentation_uuid": str(segmentation.id),
    }
    current["localization"] = {
        "status": "blocked",
        "reason": "segmentation_rejected_by_clinician",
        "segmentation_uuid": str(segmentation.id),
    }
    study.segmentation_preparation_summary = current


def _revision_payload(revision: SegmentationReviewRevision) -> dict:
    return {
        "revision_uuid": revision.id,
        "revision_number": int(revision.revision_number),
        "action": revision.action.value,
        "source_review_status": revision.source_review_status,
        "result_review_status": revision.result_review_status,
        "source_mask_checksums": _public_checksums(dict(revision.source_artifacts or {})),
        "result_mask_checksums": _public_checksums(dict(revision.result_artifacts or {})),
        "modified_voxel_count": int(revision.modified_voxel_count),
        "note": revision.note,
        "downstream_quantification_policy": revision.downstream_quantification_policy,
        "downstream_localization_policy": revision.downstream_localization_policy,
        "created_at": revision.created_at,
    }


def _result_payload(
    segmentation: Segmentation,
    revision: SegmentationReviewRevision,
    *,
    quantification_status: str,
    localization_status: str,
    quantification_error: str | None = None,
    localization_error: str | None = None,
) -> dict:
    return {
        "version": SEGMENTATION_REVIEW_VERSION,
        "status": "complete",
        "segmentation_uuid": segmentation.id,
        "review_status": segmentation.review_status.value,
        "clinician_modified": bool(segmentation.clinician_modified),
        "revision": _revision_payload(revision),
        "current_mask_checksums": _public_checksums(_artifact_snapshot(segmentation)),
        "downstream": {
            "quantification": quantification_status,
            "localization": localization_status,
            "quantification_error": quantification_error,
            "localization_error": localization_error,
        },
        "segmentation_is_gbm_diagnosis": False,
        "clinical_validation_claimed": False,
    }


def review_segmentation(
    db: Session,
    study: Study,
    *,
    action: SegmentationReviewAction,
    note: str | None = None,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> dict:
    _, segmentation = _require_current_segmentation(db, study)
    note = _normalized_note(note)
    source_status = segmentation.review_status.value
    source_artifacts = _artifact_snapshot(segmentation)

    if action == SegmentationReviewAction.ACCEPT:
        resulting = SegmentationReviewStatus.ACCEPTED
        quant_policy = "retain_if_mask_checksums_match"
        loc_policy = "retain_if_quantification_remains_current"
    elif action == SegmentationReviewAction.REJECT:
        resulting = SegmentationReviewStatus.REJECTED
        quant_policy = "blocked_until_new_mask_correction"
        loc_policy = "blocked_until_new_mask_correction"
    else:
        raise SegmentationReviewServiceError(
            "SEGMENTATION_REVIEW_ACTION_INVALID",
            "review endpoint accepts only accept or reject actions",
        )

    segmentation.review_status = resulting
    if resulting == SegmentationReviewStatus.REJECTED:
        _mark_rejected(study, segmentation)

    revision = SegmentationReviewRevision(
        segmentation_id=segmentation.id,
        revision_number=_next_revision_number(db, segmentation.id),
        review_version=SEGMENTATION_REVIEW_VERSION,
        action=action,
        source_review_status=source_status,
        result_review_status=resulting.value,
        source_artifacts=source_artifacts,
        result_artifacts=source_artifacts,
        modified_voxel_count=0,
        note=note,
        downstream_quantification_policy=quant_policy,
        downstream_localization_policy=loc_policy,
    )
    db.add(revision)
    db.flush()
    record_audit_event(
        db,
        action=AuditAction.SEGMENTATION_EDITED,
        entity_type=AuditEntityType.SEGMENTATION,
        entity_uuid=segmentation.id,
        actor_type=actor_type,
        actor_id=actor_id,
        request_id=request_id,
        technical_context={
            "status": resulting.value,
            "operation": f"segmentation_review_{action.value}",
            "result": "clinician_review_state_recorded",
            "manual_review_required": resulting != SegmentationReviewStatus.ACCEPTED,
        },
        commit=False,
    )
    db.commit()
    db.refresh(revision)
    db.refresh(segmentation)

    return _result_payload(
        segmentation,
        revision,
        quantification_status=("blocked" if resulting == SegmentationReviewStatus.REJECTED else "retained"),
        localization_status=("blocked" if resulting == SegmentationReviewStatus.REJECTED else "retained"),
    )


def _load_current_labelmap(storage: LocalObjectStore, segmentation: Segmentation):
    if not storage.verify_checksum(
        segmentation.labelmap_storage_key,
        segmentation.labelmap_checksum_sha256,
    ):
        raise SegmentationReviewServiceError(
            "SEGMENTATION_CORRECTION_SOURCE_CHECKSUM_INVALID",
            "current segmentation labelmap failed checksum validation",
        )

    import nibabel as nib

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp:
            temp_path = Path(temp.name)
            with storage.open_read(segmentation.labelmap_storage_key) as source:
                shutil.copyfileobj(source, temp)
        image = nib.load(str(temp_path))
        data = np.asarray(image.dataobj, dtype=np.uint8)
        return image, data
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _store_nifti_uint8(
    storage: LocalObjectStore,
    study: Study,
    *,
    category: str,
    data: np.ndarray,
    affine: np.ndarray,
    header,
):
    import nibabel as nib

    temp_path: Path | None = None
    try:
        result_header = header.copy()
        result_header.set_data_dtype(np.uint8)
        image = nib.Nifti1Image(data.astype(np.uint8, copy=False), affine, header=result_header)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp:
            temp_path = Path(temp.name)
        nib.save(image, str(temp_path))
        key = storage.generate_study_derived_key(
            study.id,
            category,
            suffix=".nii.gz",
        )
        with temp_path.open("rb") as source:
            return storage.put_stream(key, source)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)




def decode_clinician_labelmap(raw_labelmap_bytes: bytes, shape: tuple[int, int, int]) -> np.ndarray:
    expected_bytes = int(np.prod(shape, dtype=np.int64))
    if len(raw_labelmap_bytes) != expected_bytes:
        raise SegmentationReviewServiceError(
            "SEGMENTATION_CORRECTION_BYTE_COUNT_INVALID",
            f"correction payload must contain exactly {expected_bytes} uint8 voxels",
        )
    edited = np.frombuffer(raw_labelmap_bytes, dtype=np.uint8).reshape(shape, order="F")
    unique = set(int(v) for v in np.unique(edited))
    invalid = sorted(unique - ALLOWED_LABEL_VALUES)
    if invalid:
        raise SegmentationReviewServiceError(
            "SEGMENTATION_CORRECTION_LABEL_INVALID",
            "edited labelmap contains unsupported label value(s): " + ", ".join(map(str, invalid)),
        )
    return edited


def brats_binary_masks_from_labelmap(labelmap: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    et = labelmap == 4
    tc = np.logical_or(labelmap == 1, et)
    wt = labelmap != 0
    return tc, wt, et


def apply_labelmap_correction(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    *,
    raw_labelmap_bytes: bytes,
    source_checksum_sha256: str,
    atlas_root: Path,
    note: str | None = None,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> dict:
    _, segmentation = _require_current_segmentation(db, study)
    note = _normalized_note(note)

    source_checksum = str(source_checksum_sha256 or "").strip().lower()
    if len(source_checksum) != 64 or source_checksum != str(segmentation.labelmap_checksum_sha256).lower():
        raise SegmentationReviewServiceError(
            "SEGMENTATION_CORRECTION_CONFLICT",
            "the segmentation changed after this viewer session loaded; refresh before saving corrections",
        )

    shape = tuple(int(v) for v in (segmentation.spatial_shape or []))
    if len(shape) != 3 or any(v <= 0 for v in shape):
        raise SegmentationReviewServiceError(
            "SEGMENTATION_CORRECTION_SHAPE_INVALID",
            "current segmentation shape is invalid",
        )
    edited = decode_clinician_labelmap(raw_labelmap_bytes, shape)

    image, source_data = _load_current_labelmap(storage, segmentation)
    if tuple(int(v) for v in source_data.shape) != shape:
        raise SegmentationReviewServiceError(
            "SEGMENTATION_CORRECTION_SOURCE_SHAPE_MISMATCH",
            "persisted labelmap shape does not match segmentation metadata",
        )

    modified_voxels = int(np.count_nonzero(edited != source_data))
    if modified_voxels == 0:
        raise SegmentationReviewServiceError(
            "SEGMENTATION_CORRECTION_NO_CHANGES",
            "no voxel changes were detected; nothing was saved",
        )

    tc, wt, et = brats_binary_masks_from_labelmap(edited)
    old_artifacts = _artifact_snapshot(segmentation)
    stored = []
    try:
        for category, data in (
            ("clinician_wt", wt),
            ("clinician_tc", tc),
            ("clinician_et", et),
            ("clinician_labelmap", edited),
        ):
            stored.append(
                _store_nifti_uint8(
                    storage,
                    study,
                    category=category,
                    data=data,
                    affine=np.asarray(image.affine, dtype=np.float64),
                    header=image.header,
                )
            )

        by_category = dict(zip(("WT", "TC", "ET", "LABELMAP"), stored, strict=True))
        segmentation.wt_storage_key = by_category["WT"].storage_key
        segmentation.wt_checksum_sha256 = by_category["WT"].sha256
        segmentation.wt_size_bytes = by_category["WT"].size_bytes
        segmentation.tc_storage_key = by_category["TC"].storage_key
        segmentation.tc_checksum_sha256 = by_category["TC"].sha256
        segmentation.tc_size_bytes = by_category["TC"].size_bytes
        segmentation.et_storage_key = by_category["ET"].storage_key
        segmentation.et_checksum_sha256 = by_category["ET"].sha256
        segmentation.et_size_bytes = by_category["ET"].size_bytes
        segmentation.labelmap_storage_key = by_category["LABELMAP"].storage_key
        segmentation.labelmap_checksum_sha256 = by_category["LABELMAP"].sha256
        segmentation.labelmap_size_bytes = by_category["LABELMAP"].size_bytes
        segmentation.voxel_counts = {
            "TC": int(np.count_nonzero(tc)),
            "WT": int(np.count_nonzero(wt)),
            "ET": int(np.count_nonzero(et)),
            "LABELMAP_NONZERO": int(np.count_nonzero(edited)),
        }
        source_status = segmentation.review_status.value
        segmentation.review_status = SegmentationReviewStatus.EDITED
        segmentation.clinician_modified = True
        _invalidate_downstream(study, segmentation, reason="clinician_mask_correction")

        new_artifacts = _artifact_snapshot(segmentation)
        revision = SegmentationReviewRevision(
            segmentation_id=segmentation.id,
            revision_number=_next_revision_number(db, segmentation.id),
            review_version=SEGMENTATION_REVIEW_VERSION,
            action=SegmentationReviewAction.EDIT,
            source_review_status=source_status,
            result_review_status=SegmentationReviewStatus.EDITED.value,
            source_artifacts=old_artifacts,
            result_artifacts=new_artifacts,
            modified_voxel_count=modified_voxels,
            note=note,
            downstream_quantification_policy="automatic_recompute_requested",
            downstream_localization_policy="automatic_recompute_requested_after_quantification",
        )
        db.add(revision)
        db.flush()
        record_audit_event(
            db,
            action=AuditAction.SEGMENTATION_EDITED,
            entity_type=AuditEntityType.SEGMENTATION,
            entity_uuid=segmentation.id,
            actor_type=actor_type,
            actor_id=actor_id,
            request_id=request_id,
            technical_context={
                "status": "edited",
                "operation": "clinician_labelmap_correction",
                "result": "mask_revision_persisted_downstream_recalculation_requested",
                "manual_review_required": True,
            },
            commit=False,
        )
        db.commit()
        db.refresh(revision)
        db.refresh(segmentation)
    except Exception:
        db.rollback()
        for item in stored:
            try:
                if storage.exists(item.storage_key):
                    storage.delete(item.storage_key)
            except Exception:
                pass
        raise

    quant_status = "failed"
    loc_status = "not_run"
    quant_error = None
    loc_error = None
    try:
        run_tumor_quantification(
            db,
            storage,
            study,
            request_id=request_id,
            actor_type=actor_type,
            actor_id=actor_id,
        )
        quant_status = "recalculated"
    except TumorQuantificationServiceError as exc:
        quant_error = f"{exc.code}: {exc}"

    if quant_status == "recalculated":
        try:
            run_anatomical_localization(
                db,
                storage,
                study,
                atlas_root=atlas_root,
                request_id=request_id,
                actor_type=actor_type,
                actor_id=actor_id,
            )
            loc_status = "recalculated"
        except AnatomicalLocalizationServiceError as exc:
            loc_status = "failed"
            loc_error = f"{exc.code}: {exc}"

    return _result_payload(
        segmentation,
        revision,
        quantification_status=quant_status,
        localization_status=loc_status,
        quantification_error=quant_error,
        localization_error=loc_error,
    )


def review_history(db: Session, study: Study) -> dict:
    _, segmentation = _require_current_segmentation(db, study)
    revisions = db.scalars(
        select(SegmentationReviewRevision)
        .where(SegmentationReviewRevision.segmentation_id == segmentation.id)
        .order_by(SegmentationReviewRevision.revision_number.desc())
    ).all()
    return {
        "version": SEGMENTATION_REVIEW_VERSION,
        "segmentation_uuid": segmentation.id,
        "current_review_status": segmentation.review_status.value,
        "clinician_modified": bool(segmentation.clinician_modified),
        "revisions": [_revision_payload(item) for item in revisions],
        "immutable_history": True,
        "clinical_validation_claimed": False,
    }
