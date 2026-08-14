from __future__ import annotations

import enum
import uuid
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import (
    BrainScopeStatus,
    CapabilityRoutingStatus,
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
from gbm_ai.api.models.clinical import Assessment, Patient, ScopeStatus
from gbm_ai.api.services.audit import record_audit_event


REQUIRED_3D_SEQUENCES = ("T1", "T1C", "T2", "FLAIR")


class CapabilityState(str, enum.Enum):
    ELIGIBLE = "eligible"
    REVIEW_REQUIRED = "review_required"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


class CapabilityRoutingError(ValueError):
    pass


class StudyScopeConfirmationError(ValueError):
    pass


class NiftiSequenceMappingError(ValueError):
    pass


def _capability(
    state: CapabilityState,
    *reasons: str,
    user_message: str | None = None,
    prerequisites: list[str] | None = None,
) -> dict:
    return {
        "state": state.value,
        "reasons": list(reasons),
        "user_message": user_message,
        "prerequisites": prerequisites or [],
        "input_eligible": state == CapabilityState.ELIGIBLE,
        # This phase only routes input. It never starts model execution.
        "execution_started": False,
    }


def _load_clinical_scope(
    db: Session,
    study: Study,
) -> tuple[Assessment, Patient]:
    assessment = db.get(Assessment, study.assessment_id)
    if assessment is None:
        raise CapabilityRoutingError("study assessment not found")

    patient = db.get(Patient, assessment.patient_id)
    if patient is None:
        raise CapabilityRoutingError("assessment patient not found")

    return assessment, patient


def _sync_brain_scope_from_qc(study: Study) -> None:
    if study.brain_scope_status in {
        BrainScopeStatus.CLINICIAN_CONFIRMED,
        BrainScopeStatus.OUT_OF_SCOPE,
    }:
        return

    checks = dict((study.qc_summary or {}).get("checks") or {})
    qc_scope = str(checks.get("brain_scope_status") or "").upper()

    if qc_scope == "SUPPORTED_BY_DICOM_HINT":
        study.brain_scope_status = BrainScopeStatus.SUPPORTED_BY_METADATA
    elif qc_scope == "OUT_OF_SCOPE":
        study.brain_scope_status = BrainScopeStatus.OUT_OF_SCOPE
    else:
        study.brain_scope_status = BrainScopeStatus.PENDING


def _global_scope(
    assessment: Assessment,
    patient: Patient,
    study: Study,
) -> tuple[list[str], list[str]]:
    hard_block: list[str] = []
    review: list[str] = []

    if (
        assessment.prior_treatment
        or assessment.scope_status == ScopeStatus.OUT_OF_SCOPE_PRIOR_TREATMENT
    ):
        hard_block.append("OUT_OF_SCOPE_PRIOR_TREATMENT")

    if patient.age_years is not None and patient.age_years < 18:
        hard_block.append("OUT_OF_SCOPE_PEDIATRIC")
    elif patient.age_years is None:
        review.append("AGE_SCOPE_UNVERIFIED")

    if study.qc_status == StudyQCStatus.FAIL:
        hard_block.append("MRI_QC_FAILED")

    if study.brain_scope_status == BrainScopeStatus.OUT_OF_SCOPE:
        hard_block.append("NON_BRAIN_OR_OUT_OF_SCOPE_STUDY")
    elif study.brain_scope_status == BrainScopeStatus.PENDING:
        review.append("BRAIN_SCOPE_CONFIRMATION_REQUIRED")

    return sorted(set(hard_block)), sorted(set(review))


def _effective_dicom_sequences(series: list[Series]) -> tuple[Counter, int]:
    labels: list[str] = []
    ambiguous = 0

    for item in series:
        effective = item.confirmed_sequence or item.detected_sequence
        if effective in REQUIRED_3D_SEQUENCES:
            labels.append(effective)
        elif effective in {"NEEDS_CONFIRMATION", "UNKNOWN", None}:
            ambiguous += 1

    return Counter(labels), ambiguous


def _unresolved_image_quality_reasons(
    study: Study,
) -> list[str]:
    reasons = list((study.qc_summary or {}).get("partial_reasons") or [])
    resolved = set()

    if study.brain_scope_status in {
        BrainScopeStatus.CLINICIAN_CONFIRMED,
        BrainScopeStatus.SUPPORTED_BY_METADATA,
    }:
        resolved.add("BRAIN_SCOPE_UNVERIFIED_FOR_RASTER")

    return sorted(reason for reason in reasons if reason not in resolved)


def _route_image(
    study: Study,
    hard_block: list[str],
    global_review: list[str],
) -> dict:
    if hard_block:
        classification = _capability(
            CapabilityState.BLOCKED,
            *hard_block,
        )
    else:
        quality_review = _unresolved_image_quality_reasons(study)
        review_reasons = sorted(set(global_review + quality_review))

        if review_reasons:
            classification = _capability(
                CapabilityState.REVIEW_REQUIRED,
                *review_reasons,
                user_message=(
                    "2D classification is technically compatible with this "
                    "raster input, but manual scope/quality review is required "
                    "before inference."
                ),
            )
        else:
            classification = _capability(
                CapabilityState.ELIGIBLE,
                "CURRENT_2D_CLASSIFIER_INPUT_DOMAIN",
                user_message=(
                    "Input is eligible for the validated 2D preprocessing "
                    "path. Model execution is not started by capability routing."
                ),
            )

    if classification["state"] == CapabilityState.ELIGIBLE.value:
        gradcam = _capability(
            CapabilityState.DEFERRED,
            "REQUIRES_2D_CLASSIFIER_EXECUTION",
            prerequisites=["successful_2d_classification"],
        )
    elif classification["state"] == CapabilityState.REVIEW_REQUIRED.value:
        gradcam = _capability(
            CapabilityState.REVIEW_REQUIRED,
            *classification["reasons"],
            prerequisites=["approved_2d_classification_input"],
        )
    else:
        gradcam = _capability(
            CapabilityState.BLOCKED,
            "2D_CLASSIFICATION_BLOCKED",
        )

    return {
        "two_d_classification": classification,
        "gradcam_2d": gradcam,
        "three_d_segmentation": _capability(
            CapabilityState.BLOCKED,
            "STANDALONE_RASTER_HAS_NO_MULTIMODAL_3D_CONTEXT",
        ),
        "physical_volume": _capability(
            CapabilityState.BLOCKED,
            "NO_RELIABLE_3D_SPATIAL_METADATA",
            user_message="Physical tumor volume unavailable for this upload",
        ),
        "anatomical_localization": _capability(
            CapabilityState.BLOCKED,
            "ANATOMICAL_LOCALIZATION_REQUIRES_3D_SEGMENTATION_AND_REGISTRATION",
            user_message=(
                "Anatomical localization is unavailable for a standalone "
                "2D image; Grad-CAM is not a substitute for anatomical location."
            ),
        ),
    }


def _dicom_spatial_review_reasons(study: Study) -> list[str]:
    reasons = list((study.qc_summary or {}).get("partial_reasons") or [])

    handled_elsewhere = {
        "DICOM_SEQUENCE_MAPPING_REQUIRES_CONFIRMATION",
        "DICOM_BRAIN_SCOPE_UNVERIFIED",
    }

    unresolved: set[str] = set()
    for reason in reasons:
        if reason in handled_elsewhere:
            continue
        if reason.startswith("SEGMENTATION_SEQUENCE_MISSING_"):
            # Missing-channel state is derived again from the actual current
            # Series mappings below, so stale textual reasons cannot override
            # confirmed/updated sequence records.
            continue
        unresolved.add(reason)

    return sorted(unresolved)


def _route_dicom(
    db: Session,
    study: Study,
    hard_block: list[str],
    global_review: list[str],
) -> dict:
    classification = _capability(
        CapabilityState.BLOCKED,
        "VOLUMETRIC_TO_2D_CLASSIFIER_BRIDGE_NOT_VALIDATED",
        user_message=(
            "The current GBM classifier was trained on standalone 2D images. "
            "DICOM slice extraction/aggregation must be separately validated "
            "before classification is enabled for volumetric studies."
        ),
    )

    if hard_block:
        segmentation = _capability(
            CapabilityState.BLOCKED,
            *hard_block,
        )
    elif (
        study.deidentification_status
        != DeidentificationStatus.METADATA_DEIDENTIFIED
    ):
        segmentation = _capability(
            CapabilityState.BLOCKED,
            "DICOM_DEIDENTIFICATION_NOT_READY",
        )
    else:
        series = list(
            db.scalars(
                select(Series).where(Series.study_id == study.id)
            )
        )
        counts, ambiguous_count = _effective_dicom_sequences(series)
        missing = [
            label
            for label in REQUIRED_3D_SEQUENCES
            if counts[label] == 0
        ]
        duplicate = [
            label
            for label in REQUIRED_3D_SEQUENCES
            if counts[label] > 1
        ]
        spatial_review = _dicom_spatial_review_reasons(study)

        reasons = list(global_review)

        if duplicate:
            reasons.extend(
                f"DICOM_MULTIPLE_SERIES_MAPPED_{label}"
                for label in duplicate
            )

        if missing and ambiguous_count > 0:
            reasons.extend(
                f"DICOM_SEQUENCE_CONFIRMATION_MAY_RESOLVE_{label}"
                for label in missing
            )

        if spatial_review:
            reasons.extend(spatial_review)

        if reasons:
            segmentation = _capability(
                CapabilityState.REVIEW_REQUIRED,
                *sorted(set(reasons)),
                prerequisites=[
                    "resolved_sequence_mapping",
                    "acceptable_spatial_quality",
                ],
            )
        elif missing:
            segmentation = _capability(
                CapabilityState.BLOCKED,
                *[
                    f"SEGMENTATION_SEQUENCE_MISSING_{label}"
                    for label in missing
                ],
            )
        else:
            segmentation = _capability(
                CapabilityState.ELIGIBLE,
                "T1_T1C_T2_FLAIR_AVAILABLE",
                "DICOM_METADATA_DEIDENTIFIED",
                "QC_ACCEPTABLE_FOR_3D_PREPROCESSING",
                prerequisites=[
                    "phase6_alignment_and_preprocessing_validation"
                ],
                user_message=(
                    "Input is eligible to enter the 3D segmentation "
                    "preprocessing pipeline. Segmentation has not run yet."
                ),
            )

    if segmentation["state"] == CapabilityState.ELIGIBLE.value:
        volume = _capability(
            CapabilityState.DEFERRED,
            "REQUIRES_VALID_SEGMENTATION_MASK",
            prerequisites=[
                "successful_3d_segmentation",
                "validated_voxel_spacing",
            ],
        )
        localization = _capability(
            CapabilityState.DEFERRED,
            "REQUIRES_SEGMENTATION_AND_STANDARD_SPACE_REGISTRATION",
            prerequisites=[
                "successful_3d_segmentation",
                "validated_orientation_and_laterality",
                "validated_registration",
                "frozen_licensed_atlas",
            ],
        )
    elif segmentation["state"] == CapabilityState.REVIEW_REQUIRED.value:
        volume = _capability(
            CapabilityState.REVIEW_REQUIRED,
            "SEGMENTATION_INPUT_REQUIRES_REVIEW",
        )
        localization = _capability(
            CapabilityState.REVIEW_REQUIRED,
            "SEGMENTATION_INPUT_REQUIRES_REVIEW",
        )
    else:
        volume = _capability(
            CapabilityState.BLOCKED,
            "SEGMENTATION_INPUT_NOT_ELIGIBLE",
            user_message="Physical tumor volume unavailable for this upload",
        )
        localization = _capability(
            CapabilityState.BLOCKED,
            "SEGMENTATION_INPUT_NOT_ELIGIBLE",
        )

    return {
        "two_d_classification": classification,
        "gradcam_2d": _capability(
            CapabilityState.BLOCKED,
            "2D_CLASSIFICATION_NOT_ENABLED_FOR_DICOM",
        ),
        "three_d_segmentation": segmentation,
        "physical_volume": volume,
        "anatomical_localization": localization,
    }


def _validate_nifti_mapping_against_qc(
    study: Study,
) -> list[str]:
    mapping = dict(study.nifti_sequence_mapping or {})
    if set(mapping) != set(REQUIRED_3D_SEQUENCES):
        return ["NIFTI_SEQUENCE_MAPPING_INCOMPLETE"]

    values = list(mapping.values())
    if len(set(values)) != 4:
        return ["NIFTI_SEQUENCE_MAPPING_REUSES_VOLUME"]

    volumes = list(
        ((study.qc_summary or {}).get("checks") or {}).get("volumes") or []
    )
    by_index = {
        int(item["volume_index"]): item
        for item in volumes
        if "volume_index" in item
    }

    problems: list[str] = []
    for label in REQUIRED_3D_SEQUENCES:
        index = mapping.get(label)
        if not isinstance(index, int) or index not in by_index:
            problems.append(f"NIFTI_MAPPING_INVALID_INDEX_{label}")
            continue

        volume = by_index[index]
        if not volume.get("is_3d"):
            problems.append(f"NIFTI_MAPPING_{label}_NOT_3D")
        if not volume.get("spacing_valid"):
            problems.append(f"NIFTI_MAPPING_{label}_INVALID_SPACING")
        if not volume.get("affine_valid"):
            problems.append(f"NIFTI_MAPPING_{label}_INVALID_AFFINE")

    return sorted(set(problems))


def _route_nifti(
    study: Study,
    hard_block: list[str],
    global_review: list[str],
) -> dict:
    classification = _capability(
        CapabilityState.BLOCKED,
        "VOLUMETRIC_TO_2D_CLASSIFIER_BRIDGE_NOT_VALIDATED",
    )

    if hard_block:
        segmentation = _capability(
            CapabilityState.BLOCKED,
            *hard_block,
        )
    else:
        mapping_problems = _validate_nifti_mapping_against_qc(study)
        reasons = list(global_review)

        if not study.nifti_sequence_mapping:
            reasons.append("NIFTI_SEQUENCE_MAPPING_CONFIRMATION_REQUIRED")
        elif mapping_problems:
            reasons.extend(mapping_problems)

        qc_partial = set(
            (study.qc_summary or {}).get("partial_reasons") or []
        )
        resolved_by_scope = (
            study.brain_scope_status
            in {
                BrainScopeStatus.CLINICIAN_CONFIRMED,
                BrainScopeStatus.SUPPORTED_BY_METADATA,
            }
        )
        if resolved_by_scope:
            qc_partial.discard("BRAIN_SCOPE_UNVERIFIED_FOR_NIFTI")
        if study.nifti_sequence_mapping and not mapping_problems:
            qc_partial.discard("NIFTI_SEQUENCE_MAPPING_REQUIRES_CONFIRMATION")

        # 4D volume selection is not implemented by this routing step.
        reasons.extend(sorted(qc_partial))

        if reasons:
            segmentation = _capability(
                CapabilityState.REVIEW_REQUIRED,
                *sorted(set(reasons)),
                prerequisites=[
                    "four_confirmed_3d_nifti_channels",
                    "valid_spatial_metadata",
                ],
            )
        else:
            segmentation = _capability(
                CapabilityState.ELIGIBLE,
                "T1_T1C_T2_FLAIR_MAPPING_CONFIRMED",
                "NIFTI_SPATIAL_HEADERS_VALID",
                prerequisites=[
                    "phase6_alignment_and_preprocessing_validation"
                ],
            )

    if segmentation["state"] == CapabilityState.ELIGIBLE.value:
        volume = _capability(
            CapabilityState.DEFERRED,
            "REQUIRES_VALID_SEGMENTATION_MASK",
            prerequisites=[
                "successful_3d_segmentation",
                "validated_voxel_spacing",
            ],
        )
        localization = _capability(
            CapabilityState.DEFERRED,
            "REQUIRES_SEGMENTATION_AND_STANDARD_SPACE_REGISTRATION",
            prerequisites=[
                "successful_3d_segmentation",
                "validated_registration",
                "frozen_licensed_atlas",
            ],
        )
    elif segmentation["state"] == CapabilityState.REVIEW_REQUIRED.value:
        volume = _capability(
            CapabilityState.REVIEW_REQUIRED,
            "SEGMENTATION_INPUT_REQUIRES_REVIEW",
        )
        localization = _capability(
            CapabilityState.REVIEW_REQUIRED,
            "SEGMENTATION_INPUT_REQUIRES_REVIEW",
        )
    else:
        volume = _capability(
            CapabilityState.BLOCKED,
            "SEGMENTATION_INPUT_NOT_ELIGIBLE",
            user_message="Physical tumor volume unavailable for this upload",
        )
        localization = _capability(
            CapabilityState.BLOCKED,
            "SEGMENTATION_INPUT_NOT_ELIGIBLE",
        )

    return {
        "two_d_classification": classification,
        "gradcam_2d": _capability(
            CapabilityState.BLOCKED,
            "2D_CLASSIFICATION_NOT_ENABLED_FOR_NIFTI",
        ),
        "three_d_segmentation": segmentation,
        "physical_volume": volume,
        "anatomical_localization": localization,
    }


def _routing_status(capabilities: dict) -> CapabilityRoutingStatus:
    states = {
        item["state"]
        for item in capabilities.values()
    }

    if CapabilityState.ELIGIBLE.value in states:
        return CapabilityRoutingStatus.READY
    if CapabilityState.REVIEW_REQUIRED.value in states:
        return CapabilityRoutingStatus.REVIEW_REQUIRED
    return CapabilityRoutingStatus.NO_SUPPORTED_ANALYSIS


def route_study_capabilities(
    db: Session,
    study: Study,
    *,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> dict:
    if study.qc_status == StudyQCStatus.PENDING or not study.qc_summary:
        raise CapabilityRoutingError(
            "MRI QC must be completed and non-stale before capability routing"
        )

    _sync_brain_scope_from_qc(study)
    assessment, patient = _load_clinical_scope(db, study)
    hard_block, global_review = _global_scope(
        assessment,
        patient,
        study,
    )

    if study.source_format == SourceFormat.IMAGE:
        capabilities = _route_image(
            study,
            hard_block,
            global_review,
        )
    elif study.source_format == SourceFormat.DICOM:
        capabilities = _route_dicom(
            db,
            study,
            hard_block,
            global_review,
        )
    elif study.source_format == SourceFormat.NIFTI:
        capabilities = _route_nifti(
            study,
            hard_block,
            global_review,
        )
    else:
        raise CapabilityRoutingError(
            f"unsupported source format for capability routing: "
            f"{study.source_format.value}"
        )

    routing_status = _routing_status(capabilities)
    eligible_count = sum(
        item["state"] == CapabilityState.ELIGIBLE.value
        for item in capabilities.values()
    )
    review_count = sum(
        item["state"] == CapabilityState.REVIEW_REQUIRED.value
        for item in capabilities.values()
    )

    summary = {
        "version": "phase5_step5_capability_routing_v1",
        "routing_status": routing_status.value,
        "source_format": study.source_format.value,
        "brain_scope_status": study.brain_scope_status.value,
        "assessment_scope_status": assessment.scope_status.value,
        "age_scope_status": (
            "adult"
            if patient.age_years is not None and patient.age_years >= 18
            else "unverified"
        ),
        "global_block_reasons": hard_block,
        "global_review_reasons": global_review,
        "manual_review_required": (
            routing_status == CapabilityRoutingStatus.REVIEW_REQUIRED
        ),
        "capabilities": capabilities,
        "eligible_capability_count": eligible_count,
        "review_capability_count": review_count,
        "model_execution_started": False,
        "classifier_deployment_strategy_frozen": False,
        "volumetric_to_2d_classifier_bridge_validated": False,
        "clinical_validation_claimed": False,
    }

    study.capability_routing_status = routing_status
    study.capability_summary = summary

    if routing_status == CapabilityRoutingStatus.READY:
        study.status = StudyStatus.READY_FOR_ANALYSIS
    elif routing_status == CapabilityRoutingStatus.REVIEW_REQUIRED:
        if study.status != StudyStatus.FAILED:
            study.status = StudyStatus.UPLOADED
    else:
        if hard_block:
            study.status = StudyStatus.FAILED
        elif study.status != StudyStatus.FAILED:
            # Valid input, but no currently validated branch can analyze it.
            study.status = StudyStatus.UPLOADED

    record_audit_event(
        db,
        action=AuditAction.STUDY_CAPABILITY_ROUTED,
        entity_type=AuditEntityType.STUDY,
        entity_uuid=study.id,
        actor_type=actor_type,
        actor_id=actor_id,
        request_id=request_id,
        technical_context={
            "operation": "capability_routing",
            "routing_status": routing_status.value,
            "brain_scope_status": study.brain_scope_status.value,
            "eligible_capability_count": eligible_count,
            "review_capability_count": review_count,
            "result": "complete",
        },
        commit=False,
    )

    db.commit()
    db.refresh(study)
    return summary


def confirm_brain_scope(
    db: Session,
    study: Study,
    *,
    is_brain_mri: bool,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> Study:
    if study.qc_status == StudyQCStatus.FAIL:
        raise StudyScopeConfirmationError(
            "brain-scope confirmation cannot override failed MRI QC"
        )

    study.brain_scope_status = (
        BrainScopeStatus.CLINICIAN_CONFIRMED
        if is_brain_mri
        else BrainScopeStatus.OUT_OF_SCOPE
    )
    study.capability_routing_status = CapabilityRoutingStatus.PENDING
    study.capability_summary = {
        "stale": True,
        "stale_reason": "brain_scope_confirmation_changed",
    }

    if is_brain_mri:
        if study.status == StudyStatus.FAILED:
            study.status = StudyStatus.UPLOADED
    else:
        study.status = StudyStatus.FAILED

    record_audit_event(
        db,
        action=AuditAction.STUDY_SCOPE_CONFIRMED,
        entity_type=AuditEntityType.STUDY,
        entity_uuid=study.id,
        actor_type=actor_type,
        actor_id=actor_id,
        request_id=request_id,
        technical_context={
            "operation": "brain_scope_confirmation",
            "brain_scope_status": study.brain_scope_status.value,
            "result": "confirmed" if is_brain_mri else "out_of_scope",
        },
        commit=False,
    )

    db.commit()
    db.refresh(study)
    return study


def confirm_nifti_sequence_mapping(
    db: Session,
    study: Study,
    mapping: dict[str, int],
    *,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> Study:
    if study.source_format != SourceFormat.NIFTI:
        raise NiftiSequenceMappingError(
            "NIfTI sequence mapping is only valid for NIfTI studies"
        )
    if study.qc_status == StudyQCStatus.PENDING or not study.qc_summary:
        raise NiftiSequenceMappingError(
            "run MRI QC before confirming NIfTI sequence mapping"
        )

    normalized = {
        str(key).strip().upper(): value
        for key, value in mapping.items()
    }
    if set(normalized) != set(REQUIRED_3D_SEQUENCES):
        raise NiftiSequenceMappingError(
            "mapping must contain exactly T1, T1C, T2 and FLAIR"
        )
    if not all(isinstance(value, int) for value in normalized.values()):
        raise NiftiSequenceMappingError(
            "all NIfTI mapping values must be integer volume indexes"
        )
    if len(set(normalized.values())) != 4:
        raise NiftiSequenceMappingError(
            "each required sequence must reference a distinct volume"
        )

    volumes = list(
        ((study.qc_summary or {}).get("checks") or {}).get("volumes") or []
    )
    valid_indexes = {
        int(item["volume_index"])
        for item in volumes
        if "volume_index" in item
    }
    unknown = set(normalized.values()) - valid_indexes
    if unknown:
        raise NiftiSequenceMappingError(
            f"mapping references unknown volume index(es): {sorted(unknown)}"
        )

    for label, index in normalized.items():
        volume = next(
            item
            for item in volumes
            if int(item["volume_index"]) == index
        )
        if not volume.get("is_3d"):
            raise NiftiSequenceMappingError(
                f"{label} must reference a 3D NIfTI volume"
            )
        if not volume.get("spacing_valid") or not volume.get("affine_valid"):
            raise NiftiSequenceMappingError(
                f"{label} volume has invalid spatial metadata"
            )

    study.nifti_sequence_mapping = normalized
    study.capability_routing_status = CapabilityRoutingStatus.PENDING
    study.capability_summary = {
        "stale": True,
        "stale_reason": "nifti_sequence_mapping_changed",
    }

    record_audit_event(
        db,
        action=AuditAction.NIFTI_SEQUENCE_MAPPED,
        entity_type=AuditEntityType.STUDY,
        entity_uuid=study.id,
        actor_type=actor_type,
        actor_id=actor_id,
        request_id=request_id,
        technical_context={
            "operation": "nifti_sequence_mapping",
            "sequence_status": "confirmed",
            "result": "success",
        },
        commit=False,
    )

    db.commit()
    db.refresh(study)
    return study
