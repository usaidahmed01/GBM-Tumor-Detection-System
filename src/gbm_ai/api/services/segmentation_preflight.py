from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import (
    CapabilityRoutingStatus,
    DeidentificationStatus,
    Series,
    SourceFormat,
    Study,
    StudyQCStatus,
    StudyStatus,
)
from gbm_ai.api.segmentation.contract import (
    SEGMENTATION_INPUT_CHANNEL_ORDER,
    segmentation_contract_dict,
)


class SegmentationPreflightError(ValueError):
    pass


def _require_phase5_segmentation_gate(study: Study) -> dict:
    if study.source_format not in {
        SourceFormat.DICOM,
        SourceFormat.NIFTI,
    }:
        raise SegmentationPreflightError(
            "3D segmentation preflight accepts only DICOM or NIfTI studies"
        )

    if study.status != StudyStatus.READY_FOR_ANALYSIS:
        raise SegmentationPreflightError(
            "study is not ready for analysis; "
            "complete QC/capability routing first"
        )

    if study.qc_status not in {
        StudyQCStatus.PASS,
        StudyQCStatus.PARTIAL,
    }:
        raise SegmentationPreflightError(
            "MRI QC must pass or have its review conditions resolved before "
            "3D segmentation preflight"
        )

    if (
        study.capability_routing_status
        != CapabilityRoutingStatus.READY
    ):
        raise SegmentationPreflightError(
            "Phase 5 capability routing is not ready for this study"
        )

    summary = dict(study.capability_summary or {})

    if not summary or summary.get("stale"):
        raise SegmentationPreflightError(
            "Phase 5 capability routing is missing or stale"
        )

    if summary.get("model_execution_started") is not False:
        raise SegmentationPreflightError(
            "invalid capability state: "
            "Phase 5 must not start model execution"
        )

    capabilities = dict(summary.get("capabilities") or {})
    segmentation = dict(
        capabilities.get("three_d_segmentation") or {}
    )

    if (
        segmentation.get("state") != "eligible"
        or segmentation.get("input_eligible") is not True
        or segmentation.get("execution_started") is not False
    ):
        raise SegmentationPreflightError(
            "Phase 5 did not mark 3D segmentation input as eligible"
        )

    return summary


def _dicom_channel_plan(
    db: Session,
    study: Study,
) -> list[dict]:
    if (
        study.deidentification_status
        != DeidentificationStatus.METADATA_DEIDENTIFIED
    ):
        raise SegmentationPreflightError(
            "DICOM AI working copy is not metadata-deidentified"
        )

    if (
        not study.deidentified_storage_key
        or not study.deidentified_checksum_sha256
    ):
        raise SegmentationPreflightError(
            "DICOM de-identified working object reference is incomplete"
        )

    series = list(
        db.scalars(
            select(Series)
            .where(Series.study_id == study.id)
            .order_by(
                Series.series_number.asc().nulls_last(),
                Series.id.asc(),
            )
        )
    )

    by_sequence: dict[str, list[Series]] = defaultdict(list)

    for item in series:
        effective = (
            item.confirmed_sequence
            or item.detected_sequence
        )

        if effective in SEGMENTATION_INPUT_CHANNEL_ORDER:
            by_sequence[effective].append(item)

    missing = [
        label
        for label in SEGMENTATION_INPUT_CHANNEL_ORDER
        if len(by_sequence[label]) == 0
    ]

    duplicate = [
        label
        for label in SEGMENTATION_INPUT_CHANNEL_ORDER
        if len(by_sequence[label]) > 1
    ]

    if missing:
        raise SegmentationPreflightError(
            "required DICOM segmentation channels are missing: "
            + ", ".join(missing)
        )

    if duplicate:
        raise SegmentationPreflightError(
            "multiple DICOM series are mapped to required "
            "segmentation channels: "
            + ", ".join(duplicate)
        )

    plan: list[dict] = []

    for channel_index, label in enumerate(
        SEGMENTATION_INPUT_CHANNEL_ORDER
    ):
        item = by_sequence[label][0]

        plan.append(
            {
                "channel_index": channel_index,
                "sequence": label,
                "source_kind": "dicom_series",
                "series_uuid": item.id,
                "volume_index": None,
                "mapping_source": (
                    "clinician_confirmed"
                    if item.confirmed_sequence == label
                    else "phase5_detected"
                ),
            }
        )

    return plan


def _nifti_channel_plan(study: Study) -> list[dict]:
    if not study.storage_key or not study.checksum_sha256:
        raise SegmentationPreflightError(
            "NIfTI source object reference is incomplete"
        )

    mapping = dict(study.nifti_sequence_mapping or {})

    if set(mapping) != set(
        SEGMENTATION_INPUT_CHANNEL_ORDER
    ):
        raise SegmentationPreflightError(
            "NIfTI T1C/T1/T2/FLAIR mapping is incomplete"
        )

    mapped_indices = list(mapping.values())

    if (
        not all(
            isinstance(index, int) and index >= 0
            for index in mapped_indices
        )
        or len(set(mapped_indices))
        != len(SEGMENTATION_INPUT_CHANNEL_ORDER)
    ):
        raise SegmentationPreflightError(
            "NIfTI sequence mapping must use "
            "four distinct valid volume indices"
        )

    volumes = list(
        (
            (study.qc_summary or {})
            .get("checks", {})
            .get("volumes", [])
        )
    )

    by_index = {
        int(item["volume_index"]): item
        for item in volumes
        if isinstance(item, dict)
        and "volume_index" in item
    }

    plan: list[dict] = []

    for channel_index, label in enumerate(
        SEGMENTATION_INPUT_CHANNEL_ORDER
    ):
        volume_index = mapping[label]
        volume = by_index.get(volume_index)

        if volume is None:
            raise SegmentationPreflightError(
                f"NIfTI mapping for {label} "
                "references a missing QC volume"
            )

        if not volume.get("is_3d"):
            raise SegmentationPreflightError(
                f"NIfTI mapping for {label} "
                "is not a validated 3D volume"
            )

        if (
            not volume.get("spacing_valid")
            or not volume.get("affine_valid")
        ):
            raise SegmentationPreflightError(
                f"NIfTI mapping for {label} "
                "has invalid spatial metadata"
            )

        plan.append(
            {
                "channel_index": channel_index,
                "sequence": label,
                "source_kind": "nifti_volume",
                "series_uuid": None,
                "volume_index": volume_index,
                "mapping_source": "clinician_confirmed",
            }
        )

    return plan


def build_segmentation_preflight(
    db: Session,
    study: Study,
) -> dict:
    routing_summary = _require_phase5_segmentation_gate(
        study
    )

    if study.source_format == SourceFormat.DICOM:
        channels = _dicom_channel_plan(db, study)

    elif study.source_format == SourceFormat.NIFTI:
        channels = _nifti_channel_plan(study)

    else:
        raise SegmentationPreflightError(
            "unsupported source format for "
            "3D segmentation preflight"
        )

    return {
        "version": "phase6_step1_segmentation_preflight_v1",
        "study_uuid": study.id,
        "source_format": study.source_format.value,
        "status": "ready_for_preprocessing",
        "qc_status": study.qc_status.value,
        "routing_version": routing_summary.get("version"),
        "model_contract": segmentation_contract_dict(),
        "channels": channels,

        # Step 1 is explicitly pre-inference.
        "model_execution_started": False,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,

        "next_step": (
            "phase6_step2_volume_loading_and_alignment_validation"
        ),
    }