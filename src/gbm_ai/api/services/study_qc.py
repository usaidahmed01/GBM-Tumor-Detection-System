from __future__ import annotations

import uuid
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import (
    DeidentificationStatus,
    Series,
    SourceFormat,
    Study,
    StudyQCStatus,
    StudyStatus,
)
from gbm_ai.api.models.audit import (
    AuditAction,
    AuditActorType,
    AuditEntityType,
)
from gbm_ai.api.qc.sequence_detection import (
    ALLOWED_SEQUENCE_LABELS,
    detect_series_sequence,
)
from gbm_ai.api.qc.validators import (
    qc_nifti_object,
    qc_raster_image,
    sample_dicom_pixel_quality,
)
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.services.segmentation_state import invalidate_segmentation_preparation
from gbm_ai.api.storage.local import LocalObjectStore


REQUIRED_SEGMENTATION_SEQUENCES = ("T1", "T1C", "T2", "FLAIR")


class StudyQCStateError(ValueError):
    pass


class SeriesNotFoundError(ValueError):
    pass


class InvalidSequenceConfirmationError(ValueError):
    pass


def _effective_sequence(series: Series) -> str | None:
    if series.confirmed_sequence in REQUIRED_SEGMENTATION_SEQUENCES:
        return series.confirmed_sequence
    if series.detected_sequence in REQUIRED_SEGMENTATION_SEQUENCES:
        return series.detected_sequence
    return None


def _run_dicom_qc(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
) -> dict:
    fail_reasons: list[str] = []
    partial_reasons: list[str] = []
    warnings: list[str] = []

    if (
        study.deidentification_status
        != DeidentificationStatus.METADATA_DEIDENTIFIED
    ):
        fail_reasons.append("DICOM_DEIDENTIFICATION_INCOMPLETE")

    series = list(
        db.scalars(
            select(Series)
            .where(Series.study_id == study.id)
            .order_by(Series.series_number.asc().nulls_last())
        )
    )
    if not series:
        fail_reasons.append("DICOM_NO_SERIES")

    brain_hints = set()
    geometry_issues = []
    sequence_states = Counter()

    for item in series:
        metadata = dict(item.sequence_metadata or {})
        detection = detect_series_sequence(metadata)

        item.detected_sequence = detection.state
        item.sequence_confidence = detection.confidence

        metadata["sequence_detection"] = {
            "state": detection.state,
            "suggested_sequence": detection.suggested_sequence,
            "confidence": detection.confidence,
            "evidence": detection.evidence,
            "scores": detection.scores,
            "clinician_confirmation_required": (
                detection.state in {"NEEDS_CONFIRMATION", "UNKNOWN"}
            ),
            "method": "phase5_step4_engineering_heuristic_v1",
            "clinically_validated": False,
        }
        item.sequence_metadata = metadata
        sequence_states[detection.state] += 1

        for hint in metadata.get("body_part_hints", []) or []:
            brain_hints.add(str(hint))

        geometry = item.spacing_orientation_metadata or {}
        if not geometry.get("pixel_spacing_consistent", True):
            geometry_issues.append(
                f"SERIES_{item.id}_INCONSISTENT_PIXEL_SPACING"
            )
        if not geometry.get("orientation_consistent", True):
            geometry_issues.append(
                f"SERIES_{item.id}_INCONSISTENT_ORIENTATION"
            )

        if not geometry.get("pixel_spacing"):
            partial_reasons.append("DICOM_MISSING_PIXEL_SPACING")
        if not geometry.get("image_orientation_patient"):
            partial_reasons.append("DICOM_MISSING_IMAGE_ORIENTATION")
        if item.slice_count < 8:
            partial_reasons.append("DICOM_LOW_SLICE_COUNT_SERIES")

        matrix_sizes = metadata.get("matrix_sizes") or []
        for size in matrix_sizes:
            if (
                isinstance(size, list)
                and len(size) == 2
                and min(int(size[0]), int(size[1])) < 64
            ):
                partial_reasons.append("DICOM_LOW_INPLANE_RESOLUTION")

        if detection.state in {"NEEDS_CONFIRMATION", "UNKNOWN"}:
            partial_reasons.append("DICOM_SEQUENCE_MAPPING_REQUIRES_CONFIRMATION")

    if geometry_issues:
        partial_reasons.extend(geometry_issues)

    if "NON_BRAIN" in brain_hints and "BRAIN_OR_HEAD" not in brain_hints:
        fail_reasons.append("DICOM_EXPLICIT_NON_BRAIN_BODY_PART")
        brain_scope_status = "OUT_OF_SCOPE"
    elif "BRAIN_OR_HEAD" in brain_hints:
        brain_scope_status = "SUPPORTED_BY_DICOM_HINT"
        if "NON_BRAIN" in brain_hints:
            partial_reasons.append("DICOM_MIXED_BODY_PART_HINTS")
    else:
        brain_scope_status = "UNVERIFIED"
        partial_reasons.append("DICOM_BRAIN_SCOPE_UNVERIFIED")

    effective = [
        label
        for item in series
        if (label := _effective_sequence(item)) is not None
    ]
    effective_counts = Counter(effective)
    missing_sequences = [
        label
        for label in REQUIRED_SEGMENTATION_SEQUENCES
        if effective_counts[label] == 0
    ]

    for label in missing_sequences:
        partial_reasons.append(
            f"SEGMENTATION_SEQUENCE_MISSING_{label}"
        )

    pixel_qc = {
        "decoded_sample_count": 0,
        "decode_error_count": 0,
        "blank_sample_count": 0,
        "low_resolution_sample_count": 0,
        "samples": [],
    }

    if study.deidentified_storage_key:
        with storage.open_read(study.deidentified_storage_key) as source:
            pixel_qc = sample_dicom_pixel_quality(source)

        decoded = int(pixel_qc["decoded_sample_count"])
        blank = int(pixel_qc["blank_sample_count"])
        errors = int(pixel_qc["decode_error_count"])

        if decoded > 0 and blank == decoded:
            fail_reasons.append("DICOM_SAMPLED_PIXELS_BLANK_OR_CONSTANT")
        elif blank > 0:
            partial_reasons.append("DICOM_SOME_SAMPLED_PIXELS_BLANK")

        if int(pixel_qc["low_resolution_sample_count"]) > 0:
            partial_reasons.append("DICOM_LOW_RESOLUTION_SAMPLE")

        if decoded == 0 and errors > 0:
            partial_reasons.append("DICOM_PIXEL_DECODE_UNVERIFIED")
    else:
        fail_reasons.append("DICOM_DEIDENTIFIED_WORKING_COPY_MISSING")

    # Step 3 performs metadata reduction but does not formally validate pixel
    # privacy or claim full PS3.15 conformance.
    warnings.append("DICOM_PIXEL_PRIVACY_NOT_FORMALLY_VALIDATED")
    warnings.append("SEQUENCE_DETECTION_HEURISTIC_NOT_CLINICALLY_VALIDATED")

    return {
        "fail_reasons": sorted(set(fail_reasons)),
        "partial_reasons": sorted(set(partial_reasons)),
        "warnings": sorted(set(warnings)),
        "checks": {
            "series_count": len(series),
            "brain_scope_status": brain_scope_status,
            "body_part_hints": sorted(brain_hints),
            "sequence_state_counts": dict(sequence_states),
            "effective_sequence_counts": dict(effective_counts),
            "required_segmentation_sequences": list(
                REQUIRED_SEGMENTATION_SEQUENCES
            ),
            "missing_segmentation_sequences": missing_sequences,
            "geometry_issue_count": len(geometry_issues),
            "pixel_sample_qc": pixel_qc,
            "deidentification_status": study.deidentification_status.value,
        },
    }


def _status_from_reasons(
    fail_reasons: list[str],
    partial_reasons: list[str],
) -> StudyQCStatus:
    if fail_reasons:
        return StudyQCStatus.FAIL
    if partial_reasons:
        return StudyQCStatus.PARTIAL
    return StudyQCStatus.PASS


def run_study_qc(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    *,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> dict:
    if not study.storage_key:
        raise StudyQCStateError("study has no uploaded source object")

    if study.source_format == SourceFormat.PENDING:
        raise StudyQCStateError(
            "study format must be detected before MRI QC"
        )

    invalidate_segmentation_preparation(study)

    if study.source_format == SourceFormat.DICOM:
        result = _run_dicom_qc(db, storage, study)
    elif study.source_format == SourceFormat.IMAGE:
        with storage.open_read(study.storage_key) as source:
            basic = qc_raster_image(source)
        result = {
            "fail_reasons": basic.fail_reasons,
            "partial_reasons": basic.partial_reasons,
            "warnings": basic.warnings,
            "checks": basic.checks,
        }
    elif study.source_format == SourceFormat.NIFTI:
        with storage.open_read(study.storage_key) as source:
            basic = qc_nifti_object(source)
        result = {
            "fail_reasons": basic.fail_reasons,
            "partial_reasons": basic.partial_reasons,
            "warnings": basic.warnings,
            "checks": basic.checks,
        }
    else:
        raise StudyQCStateError(
            f"unsupported source format for QC: {study.source_format}"
        )

    fail_reasons = sorted(set(result["fail_reasons"]))
    partial_reasons = sorted(set(result["partial_reasons"]))
    warnings = sorted(set(result["warnings"]))

    qc_status = _status_from_reasons(
        fail_reasons,
        partial_reasons,
    )
    manual_review_required = qc_status == StudyQCStatus.PARTIAL

    summary = {
        "version": "phase5_step4_qc_v1",
        "qc_status": qc_status.value,
        "fail_reasons": fail_reasons,
        "partial_reasons": partial_reasons,
        "warnings": warnings,
        "manual_review_required": manual_review_required,
        "source_format": study.source_format.value,
        "checks": result["checks"],
        "inference_started": False,
        "clinically_validated_qc_thresholds": False,
        "capability_routing_completed": False,
    }

    study.qc_status = qc_status
    study.qc_summary = summary

    if qc_status == StudyQCStatus.FAIL:
        study.status = StudyStatus.FAILED
    elif study.status != StudyStatus.FAILED:
        # Capability routing in Step 5 decides whether/when this can become
        # READY_FOR_ANALYSIS.
        study.status = StudyStatus.UPLOADED

    record_audit_event(
        db,
        action=AuditAction.STUDY_QC_COMPLETED,
        entity_type=AuditEntityType.STUDY,
        entity_uuid=study.id,
        actor_type=actor_type,
        actor_id=actor_id,
        request_id=request_id,
        technical_context={
            "operation": "mri_qc",
            "qc_status": qc_status.value,
            "manual_review_required": manual_review_required,
            "reason_count": len(fail_reasons) + len(partial_reasons),
            "result": "complete",
        },
        commit=False,
    )

    db.commit()
    db.refresh(study)
    return summary


def confirm_series_sequence(
    db: Session,
    series_uuid: uuid.UUID,
    sequence_label: str,
    *,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> Series:
    label = sequence_label.strip().upper()
    if label not in ALLOWED_SEQUENCE_LABELS:
        raise InvalidSequenceConfirmationError(
            "sequence label must be one of: "
            + ", ".join(sorted(ALLOWED_SEQUENCE_LABELS))
        )

    series = db.get(Series, series_uuid)
    if series is None:
        raise SeriesNotFoundError(str(series_uuid))

    series.confirmed_sequence = label

    study = db.get(Study, series.study_id)
    if study is not None:
        invalidate_segmentation_preparation(study)
        prior_summary = dict(study.qc_summary or {})
        prior_summary["stale"] = True
        prior_summary["stale_reason"] = "series_sequence_confirmation_changed"
        prior_summary["capability_routing_completed"] = False
        study.qc_summary = prior_summary
        study.qc_status = StudyQCStatus.PENDING
        if study.status != StudyStatus.FAILED:
            study.status = StudyStatus.UPLOADED

    metadata = dict(series.sequence_metadata or {})
    confirmation = {
        "status": "confirmed",
        "label": label,
        "actor_type": actor_type.value,
    }
    metadata["sequence_confirmation"] = confirmation
    series.sequence_metadata = metadata

    record_audit_event(
        db,
        action=AuditAction.SERIES_SEQ_CONFIRMED,
        entity_type=AuditEntityType.SERIES,
        entity_uuid=series.id,
        actor_type=actor_type,
        actor_id=actor_id,
        request_id=request_id,
        technical_context={
            "operation": "sequence_confirmation",
            "sequence_label": label,
            "sequence_status": "confirmed",
            "result": "success",
        },
        commit=False,
    )

    db.commit()
    db.refresh(series)
    return series
