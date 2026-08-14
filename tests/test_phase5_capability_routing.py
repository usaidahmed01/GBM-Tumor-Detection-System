from __future__ import annotations

from gbm_ai.api.upload.capability_routing import (
    CapabilityRoutingInput,
    CapabilityState,
    route_analysis_capabilities,
)
from gbm_ai.api.upload.sequence_detection import (
    MRISequence,
    SequenceAssignment,
    StudySequenceMapping,
)


def complete_mapping(
    *,
    confirmation_required: bool = False,
    missing: tuple[str, ...] = (),
) -> StudySequenceMapping:
    assignments = {}

    for sequence in (
        MRISequence.T1,
        MRISequence.T1C,
        MRISequence.T2,
        MRISequence.FLAIR,
    ):
        is_missing = sequence.value in missing
        assignments[sequence.value] = SequenceAssignment(
            sequence=sequence,
            selected_series_uid=None if is_missing else f"series-{sequence.value}",
            confidence=0.95 if not confirmation_required else 0.60,
            ambiguous=confirmation_required and not is_missing,
            candidate_series_uids=(
                ()
                if is_missing
                else (f"series-{sequence.value}",)
            ),
            reasons=(
                ("clinician confirmation required",)
                if confirmation_required and not is_missing
                else ()
            ),
        )

    return StudySequenceMapping(
        assignments=assignments,
        missing_sequences=missing,
        clinician_confirmation_required=confirmation_required,
        warnings=(),
    )


def state(result, capability: str) -> CapabilityState:
    return result.decisions[capability].state


def test_2d_image_routes_only_2d_classifier_and_gradcam():
    result = route_analysis_capabilities(
        CapabilityRoutingInput(
            detected_format="png",
            qc_status="pass",
        )
    )

    assert state(result, "classification") == CapabilityState.ELIGIBLE
    assert state(result, "gradcam") == CapabilityState.ELIGIBLE
    assert state(result, "segmentation_3d") == CapabilityState.BLOCKED
    assert state(result, "physical_volume") == CapabilityState.BLOCKED
    assert state(result, "anatomical_localization") == CapabilityState.BLOCKED
    assert (
        result.decisions["physical_volume"].user_message
        == "Physical tumor volume unavailable for this upload"
    )


def test_qc_failure_blocks_every_capability():
    result = route_analysis_capabilities(
        CapabilityRoutingInput(
            detected_format="dicom",
            qc_status="fail",
            dicom_sequence_mapping=complete_mapping(),
            spatial_valid=True,
        )
    )

    assert result.eligible_capabilities == ()
    assert len(result.blocked_capabilities) == 5


def test_complete_dicom_study_routes_3d_paths_but_not_2d_classifier():
    result = route_analysis_capabilities(
        CapabilityRoutingInput(
            detected_format="dicom",
            dicom_sequence_mapping=complete_mapping(),
            spatial_valid=True,
        )
    )

    assert state(result, "classification") == CapabilityState.BLOCKED
    assert state(result, "segmentation_3d") == CapabilityState.ELIGIBLE
    assert state(result, "physical_volume") == CapabilityState.ELIGIBLE
    assert state(result, "anatomical_localization") == CapabilityState.ELIGIBLE


def test_missing_flair_blocks_3d_dependent_paths():
    result = route_analysis_capabilities(
        CapabilityRoutingInput(
            detected_format="dicom",
            dicom_sequence_mapping=complete_mapping(missing=("flair",)),
            spatial_valid=True,
        )
    )

    assert state(result, "segmentation_3d") == CapabilityState.BLOCKED
    assert state(result, "physical_volume") == CapabilityState.BLOCKED
    assert state(result, "anatomical_localization") == CapabilityState.BLOCKED


def test_ambiguous_dicom_mapping_requires_confirmation():
    result = route_analysis_capabilities(
        CapabilityRoutingInput(
            detected_format="dicom",
            dicom_sequence_mapping=complete_mapping(
                confirmation_required=True
            ),
            spatial_valid=True,
            sequence_confirmation_completed=False,
        )
    )

    assert (
        state(result, "segmentation_3d")
        == CapabilityState.REQUIRES_CONFIRMATION
    )
    assert (
        state(result, "physical_volume")
        == CapabilityState.REQUIRES_CONFIRMATION
    )
    assert (
        state(result, "anatomical_localization")
        == CapabilityState.REQUIRES_CONFIRMATION
    )


def test_confirmed_dicom_mapping_can_continue():
    result = route_analysis_capabilities(
        CapabilityRoutingInput(
            detected_format="dicom",
            dicom_sequence_mapping=complete_mapping(
                confirmation_required=True
            ),
            sequence_confirmation_completed=True,
            spatial_valid=True,
        )
    )

    assert state(result, "segmentation_3d") == CapabilityState.ELIGIBLE
    assert state(result, "physical_volume") == CapabilityState.ELIGIBLE


def test_spatial_invalid_keeps_segmentation_but_blocks_volume_and_location():
    result = route_analysis_capabilities(
        CapabilityRoutingInput(
            detected_format="dicom",
            dicom_sequence_mapping=complete_mapping(),
            spatial_valid=False,
        )
    )

    assert state(result, "segmentation_3d") == CapabilityState.ELIGIBLE
    assert state(result, "physical_volume") == CapabilityState.BLOCKED
    assert state(result, "anatomical_localization") == CapabilityState.BLOCKED


def test_complete_nifti_routes_volumetric_paths():
    result = route_analysis_capabilities(
        CapabilityRoutingInput(
            detected_format="nifti",
            nifti_sequences=("T1", "T1c", "T2", "FLAIR"),
            spatial_valid=True,
        )
    )

    assert state(result, "segmentation_3d") == CapabilityState.ELIGIBLE
    assert state(result, "physical_volume") == CapabilityState.ELIGIBLE
    assert state(result, "anatomical_localization") == CapabilityState.ELIGIBLE
    assert state(result, "classification") == CapabilityState.BLOCKED


def test_incomplete_nifti_blocks_3d_dependent_paths():
    result = route_analysis_capabilities(
        CapabilityRoutingInput(
            detected_format="nii.gz",
            nifti_sequences=("T1", "T2", "FLAIR"),
            spatial_valid=True,
        )
    )

    assert state(result, "segmentation_3d") == CapabilityState.BLOCKED
    assert state(result, "physical_volume") == CapabilityState.BLOCKED
    assert state(result, "anatomical_localization") == CapabilityState.BLOCKED


def test_validated_volumetric_classifier_bridge_can_be_enabled_later():
    result = route_analysis_capabilities(
        CapabilityRoutingInput(
            detected_format="dicom",
            dicom_sequence_mapping=complete_mapping(),
            spatial_valid=True,
            volume_to_classifier_bridge_validated=True,
        )
    )

    assert state(result, "classification") == CapabilityState.ELIGIBLE


def test_out_of_scope_case_blocks_everything():
    result = route_analysis_capabilities(
        CapabilityRoutingInput(
            detected_format="png",
            scope_ok=False,
        )
    )

    assert result.eligible_capabilities == ()
    assert len(result.blocked_capabilities) == 5
