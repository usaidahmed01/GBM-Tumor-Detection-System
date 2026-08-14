from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from gbm_ai.api.upload.sequence_detection import (
    MRISequence,
    StudySequenceMapping,
)


class InputFormat(str, Enum):
    IMAGE_2D = "image_2d"
    DICOM = "dicom"
    NIFTI = "nifti"
    UNKNOWN = "unknown"


class AnalysisCapability(str, Enum):
    CLASSIFICATION = "classification"
    GRAD_CAM = "gradcam"
    SEGMENTATION_3D = "segmentation_3d"
    PHYSICAL_VOLUME = "physical_volume"
    ANATOMICAL_LOCALIZATION = "anatomical_localization"


class CapabilityState(str, Enum):
    ELIGIBLE = "eligible"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class CapabilityDecision:
    capability: AnalysisCapability
    state: CapabilityState
    reasons: tuple[str, ...]
    user_message: str | None = None


@dataclass(frozen=True)
class CapabilityRoutingInput:
    """
    Phase 5 Step 5 consumes already-validated upstream facts.

    `detected_format` MUST come from Step 2 content-based detection. This
    router never inspects or trusts a filename extension.

    `dicom_sequence_mapping` comes from Step 4.
    `nifti_sequences` contains Step-4 detected/confirmed NIfTI channel names.

    `eligible` means the uploaded data are suitable to enter that later
    analysis path. It does NOT mean the Phase 6/7 model has already run.
    """

    detected_format: str | InputFormat

    integrity_ok: bool = True
    scope_ok: bool = True
    brain_study_ok: bool = True
    mr_modality_ok: bool = True
    qc_status: str = "pass"

    spatial_valid: bool | None = None

    dicom_sequence_mapping: StudySequenceMapping | None = None

    nifti_sequences: tuple[str, ...] = ()
    nifti_sequence_confirmation_required: bool = False

    sequence_confirmation_completed: bool = False

    # The V1 classifier is trained on 2D images. Keep volumetric
    # classification disabled until that bridge is separately validated.
    volume_to_classifier_bridge_validated: bool = False


@dataclass(frozen=True)
class CapabilityRoutingResult:
    input_format: InputFormat
    decisions: dict[str, CapabilityDecision]
    eligible_capabilities: tuple[str, ...]
    confirmation_required_capabilities: tuple[str, ...]
    blocked_capabilities: tuple[str, ...]
    warnings: tuple[str, ...]
    routing_version: str = "phase5_step5_v1"


_REQUIRED_3D_SEQUENCES: tuple[MRISequence, ...] = (
    MRISequence.T1,
    MRISequence.T1C,
    MRISequence.T2,
    MRISequence.FLAIR,
)


def normalize_input_format(value: str | InputFormat) -> InputFormat:
    if isinstance(value, InputFormat):
        return value

    normalized = str(value).strip().lower()

    # These are labels produced by Step 2, not extension-based detection.
    aliases = {
        "image_2d": InputFormat.IMAGE_2D,
        "2d_image": InputFormat.IMAGE_2D,
        "jpg": InputFormat.IMAGE_2D,
        "jpeg": InputFormat.IMAGE_2D,
        "png": InputFormat.IMAGE_2D,
        "dicom": InputFormat.DICOM,
        "dcm": InputFormat.DICOM,
        "dicom_study": InputFormat.DICOM,
        "nifti": InputFormat.NIFTI,
        "nii": InputFormat.NIFTI,
        "nii.gz": InputFormat.NIFTI,
    }
    return aliases.get(normalized, InputFormat.UNKNOWN)


def _decision(
    capability: AnalysisCapability,
    state: CapabilityState,
    *reasons: str,
    user_message: str | None = None,
) -> CapabilityDecision:
    return CapabilityDecision(
        capability=capability,
        state=state,
        reasons=tuple(reason for reason in reasons if reason),
        user_message=user_message,
    )


def _all_blocked(reason: str) -> dict[str, CapabilityDecision]:
    return {
        capability.value: _decision(
            capability,
            CapabilityState.BLOCKED,
            reason,
        )
        for capability in AnalysisCapability
    }


def _normalize_sequence_names(values: Iterable[str]) -> set[str]:
    aliases = {
        "t1": MRISequence.T1.value,
        "t1w": MRISequence.T1.value,
        "t1-weighted": MRISequence.T1.value,
        "t1c": MRISequence.T1C.value,
        "t1ce": MRISequence.T1C.value,
        "t1gd": MRISequence.T1C.value,
        "t1-gd": MRISequence.T1C.value,
        "t1 post": MRISequence.T1C.value,
        "t1 post-contrast": MRISequence.T1C.value,
        "t2": MRISequence.T2.value,
        "t2w": MRISequence.T2.value,
        "t2-weighted": MRISequence.T2.value,
        "flair": MRISequence.FLAIR.value,
        "t2 flair": MRISequence.FLAIR.value,
    }
    result: set[str] = set()
    for value in values:
        normalized = str(value).strip().lower().replace("_", " ")
        mapped = aliases.get(normalized, normalized.replace(" ", ""))
        if mapped in {sequence.value for sequence in _REQUIRED_3D_SEQUENCES}:
            result.add(mapped)
    return result


def _dicom_sequence_state(
    routing_input: CapabilityRoutingInput,
) -> tuple[CapabilityState, tuple[str, ...], tuple[str, ...]]:
    mapping = routing_input.dicom_sequence_mapping

    if mapping is None:
        return (
            CapabilityState.BLOCKED,
            ("DICOM sequence mapping is unavailable.",),
            (),
        )

    missing = tuple(mapping.missing_sequences)
    if missing:
        return (
            CapabilityState.BLOCKED,
            (
                "Required 3D MRI sequence(s) are missing: "
                + ", ".join(missing)
                + ".",
            ),
            tuple(mapping.warnings),
        )

    unresolved_assignment = any(
        mapping.assignments.get(sequence.value) is None
        or mapping.assignments[sequence.value].selected_series_uid is None
        for sequence in _REQUIRED_3D_SEQUENCES
    )
    if unresolved_assignment:
        return (
            CapabilityState.BLOCKED,
            ("One or more required DICOM sequence assignments are unresolved.",),
            tuple(mapping.warnings),
        )

    confirmation_needed = (
        mapping.clinician_confirmation_required
        or any(
            mapping.assignments[sequence.value].ambiguous
            for sequence in _REQUIRED_3D_SEQUENCES
        )
    )

    if confirmation_needed and not routing_input.sequence_confirmation_completed:
        return (
            CapabilityState.REQUIRES_CONFIRMATION,
            (
                "Automatic DICOM sequence mapping is ambiguous and requires "
                "clinician confirmation.",
            ),
            tuple(mapping.warnings),
        )

    return (
        CapabilityState.ELIGIBLE,
        ("T1, T1c, T2 and FLAIR are mapped.",),
        tuple(mapping.warnings),
    )


def _nifti_sequence_state(
    routing_input: CapabilityRoutingInput,
) -> tuple[CapabilityState, tuple[str, ...], tuple[str, ...]]:
    present = _normalize_sequence_names(routing_input.nifti_sequences)
    required = {sequence.value for sequence in _REQUIRED_3D_SEQUENCES}
    missing = tuple(sorted(required - present))

    if missing:
        return (
            CapabilityState.BLOCKED,
            (
                "Required 3D MRI sequence(s) are missing: "
                + ", ".join(missing)
                + ".",
            ),
            (),
        )

    if (
        routing_input.nifti_sequence_confirmation_required
        and not routing_input.sequence_confirmation_completed
    ):
        return (
            CapabilityState.REQUIRES_CONFIRMATION,
            (
                "NIfTI sequence mapping requires clinician confirmation.",
            ),
            (),
        )

    return (
        CapabilityState.ELIGIBLE,
        ("T1, T1c, T2 and FLAIR are mapped.",),
        (),
    )


def _route_2d(
    routing_input: CapabilityRoutingInput,
) -> tuple[dict[str, CapabilityDecision], tuple[str, ...]]:
    decisions = {
        AnalysisCapability.CLASSIFICATION.value: _decision(
            AnalysisCapability.CLASSIFICATION,
            CapabilityState.ELIGIBLE,
            "Supported standalone 2D MRI image.",
        ),
        AnalysisCapability.GRAD_CAM.value: _decision(
            AnalysisCapability.GRAD_CAM,
            CapabilityState.ELIGIBLE,
            "2D classifier explanation path is eligible.",
        ),
        AnalysisCapability.SEGMENTATION_3D.value: _decision(
            AnalysisCapability.SEGMENTATION_3D,
            CapabilityState.BLOCKED,
            "Standalone 2D images do not provide the required four aligned 3D MRI volumes.",
        ),
        AnalysisCapability.PHYSICAL_VOLUME.value: _decision(
            AnalysisCapability.PHYSICAL_VOLUME,
            CapabilityState.BLOCKED,
            "Reliable physical voxel spacing and 3D tumor mask are unavailable.",
            user_message="Physical tumor volume unavailable for this upload",
        ),
        AnalysisCapability.ANATOMICAL_LOCALIZATION.value: _decision(
            AnalysisCapability.ANATOMICAL_LOCALIZATION,
            CapabilityState.BLOCKED,
            "Complete 3D anatomical localization requires volumetric spatial metadata and registration.",
        ),
    }

    warnings = (
        "2D input supports GBM classification, but must not be treated as "
        "volumetric MRI.",
    )
    return decisions, warnings


def _route_volumetric(
    routing_input: CapabilityRoutingInput,
    *,
    input_format: InputFormat,
) -> tuple[dict[str, CapabilityDecision], tuple[str, ...]]:
    if input_format == InputFormat.DICOM:
        sequence_state, sequence_reasons, sequence_warnings = _dicom_sequence_state(
            routing_input
        )
    else:
        sequence_state, sequence_reasons, sequence_warnings = _nifti_sequence_state(
            routing_input
        )

    if routing_input.volume_to_classifier_bridge_validated:
        classification = _decision(
            AnalysisCapability.CLASSIFICATION,
            CapabilityState.ELIGIBLE,
            "A separately validated volumetric-to-2D classifier bridge is enabled.",
        )
    else:
        classification = _decision(
            AnalysisCapability.CLASSIFICATION,
            CapabilityState.BLOCKED,
            "The current GBM classifier is 2D; volumetric classification requires separate validation.",
        )

    grad_cam = _decision(
        AnalysisCapability.GRAD_CAM,
        CapabilityState.BLOCKED,
        "The current Grad-CAM path belongs to the validated 2D classifier workflow.",
    )

    segmentation = _decision(
        AnalysisCapability.SEGMENTATION_3D,
        sequence_state,
        *sequence_reasons,
    )

    if sequence_state == CapabilityState.REQUIRES_CONFIRMATION:
        physical_volume = _decision(
            AnalysisCapability.PHYSICAL_VOLUME,
            CapabilityState.REQUIRES_CONFIRMATION,
            "Physical volume depends on a confirmed 3D sequence mapping.",
            user_message="Physical tumor volume unavailable for this upload",
        )
        localization = _decision(
            AnalysisCapability.ANATOMICAL_LOCALIZATION,
            CapabilityState.REQUIRES_CONFIRMATION,
            "Anatomical localization depends on a confirmed 3D sequence mapping.",
        )
    elif sequence_state == CapabilityState.BLOCKED:
        physical_volume = _decision(
            AnalysisCapability.PHYSICAL_VOLUME,
            CapabilityState.BLOCKED,
            "Physical volume requires an eligible 3D segmentation path.",
            user_message="Physical tumor volume unavailable for this upload",
        )
        localization = _decision(
            AnalysisCapability.ANATOMICAL_LOCALIZATION,
            CapabilityState.BLOCKED,
            "Anatomical localization requires an eligible 3D segmentation path.",
        )
    elif routing_input.spatial_valid is not True:
        physical_volume = _decision(
            AnalysisCapability.PHYSICAL_VOLUME,
            CapabilityState.BLOCKED,
            "Spatial metadata/affine validation did not pass.",
            user_message="Physical tumor volume unavailable for this upload",
        )
        localization = _decision(
            AnalysisCapability.ANATOMICAL_LOCALIZATION,
            CapabilityState.BLOCKED,
            "Spatial metadata/affine validation did not pass; later registration/localization must not proceed.",
        )
    else:
        physical_volume = _decision(
            AnalysisCapability.PHYSICAL_VOLUME,
            CapabilityState.ELIGIBLE,
            "Required sequences and valid spatial metadata are available.",
        )
        localization = _decision(
            AnalysisCapability.ANATOMICAL_LOCALIZATION,
            CapabilityState.ELIGIBLE,
            "Required sequences and valid spatial metadata are available for the later registration/localization pipeline.",
        )

    decisions = {
        AnalysisCapability.CLASSIFICATION.value: classification,
        AnalysisCapability.GRAD_CAM.value: grad_cam,
        AnalysisCapability.SEGMENTATION_3D.value: segmentation,
        AnalysisCapability.PHYSICAL_VOLUME.value: physical_volume,
        AnalysisCapability.ANATOMICAL_LOCALIZATION.value: localization,
    }

    warnings = list(sequence_warnings)
    if routing_input.spatial_valid is not True:
        warnings.append(
            "Spatial validation is not valid; physical volume and anatomical "
            "localization are blocked."
        )

    if not routing_input.volume_to_classifier_bridge_validated:
        warnings.append(
            "Volumetric GBM classification remains disabled until the 2D-to-3D "
            "processing bridge is separately validated."
        )

    return decisions, tuple(dict.fromkeys(warnings))


def route_analysis_capabilities(
    routing_input: CapabilityRoutingInput,
) -> CapabilityRoutingResult:
    """
    Phase 5 Step 5 capability router.

    This function selects data-eligible analysis paths only. It does not run
    the classifier, segmentation model, registration, quantification, or final
    Phase-9 safety fusion.
    """
    input_format = normalize_input_format(routing_input.detected_format)

    hard_failure: str | None = None

    if input_format == InputFormat.UNKNOWN:
        hard_failure = "Unsupported or unresolved input format."
    elif not routing_input.integrity_ok:
        hard_failure = "Input integrity validation failed."
    elif not routing_input.scope_ok:
        hard_failure = "Case is outside the V1 intended-use scope."
    elif not routing_input.brain_study_ok:
        hard_failure = "Input is not validated as a brain study."
    elif routing_input.qc_status.strip().lower() == "fail":
        hard_failure = "MRI quality control failed."
    elif (
        input_format in {InputFormat.DICOM, InputFormat.NIFTI}
        and not routing_input.mr_modality_ok
    ):
        hard_failure = "Volumetric input is not validated as MR."

    if hard_failure is not None:
        decisions = _all_blocked(hard_failure)
        warnings = (hard_failure,)
    elif input_format == InputFormat.IMAGE_2D:
        decisions, warnings = _route_2d(routing_input)
    else:
        decisions, warnings = _route_volumetric(
            routing_input,
            input_format=input_format,
        )

    eligible = tuple(
        capability
        for capability, decision in decisions.items()
        if decision.state == CapabilityState.ELIGIBLE
    )
    confirmation = tuple(
        capability
        for capability, decision in decisions.items()
        if decision.state == CapabilityState.REQUIRES_CONFIRMATION
    )
    blocked = tuple(
        capability
        for capability, decision in decisions.items()
        if decision.state == CapabilityState.BLOCKED
    )

    return CapabilityRoutingResult(
        input_format=input_format,
        decisions=decisions,
        eligible_capabilities=eligible,
        confirmation_required_capabilities=confirmation,
        blocked_capabilities=blocked,
        warnings=warnings,
    )
