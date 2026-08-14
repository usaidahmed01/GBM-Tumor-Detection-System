from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


SegmentationSequence = Literal[
    "T1C",
    "T1",
    "T2",
    "FLAIR",
]

SegmentationOutput = Literal[
    "TC",
    "WT",
    "ET",
]


class SegmentationModelContractResponse(BaseModel):
    contract_version: str
    bundle_name: Literal["brats_mri_segmentation"]
    architecture: Literal["SegResNet"]
    required_input_channel_order: list[SegmentationSequence]
    output_channel_order: list[SegmentationOutput]
    reference_spacing_mm: list[float]
    requires_orientation_normalization: Literal[True] = True
    requires_alignment_validation: Literal[True] = True
    requires_reference_geometry_resampling: Literal[True] = True
    runtime_model_loading_implemented: Literal[True] = True
    inference_implemented: Literal[True] = True
    clinical_validation_claimed: Literal[False] = False


class SegmentationChannelPlan(BaseModel):
    channel_index: int
    sequence: SegmentationSequence
    source_kind: Literal[
        "dicom_series",
        "nifti_volume",
    ]
    series_uuid: uuid.UUID | None = None
    volume_index: int | None = None
    mapping_source: Literal[
        "clinician_confirmed",
        "phase5_detected",
    ]


class SegmentationPreflightResponse(BaseModel):
    version: Literal[
        "phase6_step1_segmentation_preflight_v1"
    ]
    study_uuid: uuid.UUID
    source_format: Literal[
        "dicom",
        "nifti",
    ]
    status: Literal[
        "ready_for_preprocessing"
    ]
    qc_status: Literal[
        "pass",
        "partial",
    ]
    routing_version: str | None
    model_contract: SegmentationModelContractResponse
    channels: list[SegmentationChannelPlan]
    model_execution_started: Literal[False] = False
    segmentation_generated: Literal[False] = False
    physical_volume_generated: Literal[False] = False
    anatomical_localization_generated: Literal[False] = False
    clinical_validation_claimed: Literal[False] = False
    next_step: Literal[
        "phase6_step2_volume_loading_and_alignment_validation"
    ] = "phase6_step2_volume_loading_and_alignment_validation"


class SegmentationPreparedChannelResponse(BaseModel):
    sequence: SegmentationSequence
    source_kind: Literal["dicom_series", "nifti_volume"]
    source_reference: str
    shape: list[int]
    spacing_mm: list[float]
    orientation_codes: list[str]
    affine_ras: list[list[float]]
    voxel_count: int
    dtype: Literal["float32"] = "float32"
    orientation_normalized: Literal[True] = True


class SegmentationAlignmentResponse(BaseModel):
    aligned: bool
    reference_sequence: Literal["T1C"] = "T1C"
    reasons: list[str]
    affine_tolerance_mm: float | None = None
    spacing_tolerance_mm: float | None = None


class SegmentationPreparationResponse(BaseModel):
    version: Literal["phase6_step2_volume_preparation_v1"]
    contract_version: str
    study_uuid: uuid.UUID
    source_format: Literal["dicom", "nifti"]
    status: Literal["ready", "registration_required", "failed"]
    failure_reason_code: str | None = None
    channel_order: list[SegmentationSequence]
    channels: list[SegmentationPreparedChannelResponse] = Field(default_factory=list)
    alignment: SegmentationAlignmentResponse | None = None
    canonical_orientation: Literal["RAS"] | None = None
    reference_spacing_target_mm: list[float] | None = None
    registration_performed: bool = False
    reference_geometry_resampling_performed: bool = False
    model_execution_started: bool = False
    segmentation_generated: bool = False
    physical_volume_generated: Literal[False] = False
    anatomical_localization_generated: Literal[False] = False
    clinical_validation_claimed: Literal[False] = False
    next_step: str | None = None


class SegmentationRegistrationResponse(BaseModel):
    performed: bool
    metric: str | None = None
    final_metric_value: float | None = None
    optimizer_stop_condition: str | None = None
    transform: str | None = None
    max_sampled_displacement_mm: float | None = None
    deterministic_seed: int | None = None


class SegmentationModelGeometryChannelResponse(BaseModel):
    sequence: SegmentationSequence
    storage_key: str
    checksum_sha256: str
    size_bytes: int
    shape: list[int]
    spacing_mm: list[float]
    affine_ras: list[list[float]]
    dtype: Literal["float32"] = "float32"
    interpolation: Literal["linear"] = "linear"
    registration: SegmentationRegistrationResponse


class SegmentationModelGeometryResponse(BaseModel):
    version: Literal["phase6_step3_model_geometry_v1"]
    status: Literal["ready", "failed"]
    failure_reason_code: str | None = None
    reference_sequence: Literal["T1C"] = "T1C"
    channel_order: list[SegmentationSequence]
    target_spacing_mm: list[float]
    target_shape: list[int] | None = None
    target_affine_ras: list[list[float]] | None = None
    channels: list[SegmentationModelGeometryChannelResponse] = Field(default_factory=list)
    registration_performed: bool = False
    registration_method: str | None = None
    registration_quality_gate: str | None = None
    reference_geometry_resampling_performed: bool = False
    interpolation: Literal["linear"] | None = None
    intensity_normalization_performed: Literal[False] = False
    crop_pad_performed: Literal[False] = False
    model_execution_started: Literal[False] = False
    segmentation_generated: Literal[False] = False
    physical_volume_generated: Literal[False] = False
    anatomical_localization_generated: Literal[False] = False
    clinical_validation_claimed: Literal[False] = False
    next_step: str | None = None


class SegmentationModelInputNormalizationChannelResponse(BaseModel):
    sequence: SegmentationSequence
    nonzero_voxels: int
    mean_before: float
    std_before: float
    mean_after: float
    std_after: float


class SegmentationModelInputNormalizationResponse(BaseModel):
    transform: Literal["MONAI NormalizeIntensity"]
    nonzero: Literal[True] = True
    channel_wise: Literal[True] = True
    channel_stats: list[SegmentationModelInputNormalizationChannelResponse]


class SegmentationInferenceContractResponse(BaseModel):
    roi_size: list[int]
    sw_batch_size: int
    overlap: float
    activation: Literal["sigmoid"]
    threshold: float


class SegmentationModelInputResponse(BaseModel):
    version: Literal["phase6_step4_monai_model_input_v1"]
    status: Literal["ready", "failed"]
    failure_reason_code: str | None = None
    preprocessing_version: str | None = None
    bundle_name: Literal["brats_mri_segmentation"] | None = None
    bundle_version: Literal["0.5.4"] | None = None
    bundle_model_sha256: str | None = None
    channel_order: list[SegmentationSequence] = Field(default_factory=list)
    shape: list[int] | None = None
    spatial_shape: list[int] | None = None
    dtype: Literal["float32"] | None = None
    affine_ras: list[list[float]] | None = None
    storage_key: str | None = None
    checksum_sha256: str | None = None
    size_bytes: int | None = None
    normalization: SegmentationModelInputNormalizationResponse | None = None
    inference_contract: SegmentationInferenceContractResponse | None = None
    domain_warnings: list[str] = Field(default_factory=list)
    intensity_normalization_performed: bool = False
    crop_pad_performed: bool = False
    sliding_window_padding_deferred_to_inferer: bool = False
    bundle_runtime_loading_verified_per_environment: bool = False
    model_execution_started: Literal[False] = False
    segmentation_generated: Literal[False] = False
    physical_volume_generated: Literal[False] = False
    anatomical_localization_generated: Literal[False] = False
    clinical_validation_claimed: Literal[False] = False
    next_step: str | None = None


class SegmentationMaskArtifactResponse(BaseModel):
    storage_key: str
    checksum_sha256: str
    size_bytes: int
    foreground_voxels: int


class SegmentationInferenceResponse(BaseModel):
    version: Literal["phase6_step5_guarded_segmentation_inference_v1"]
    status: Literal["complete"]
    analysis_run_uuid: uuid.UUID
    segmentation_uuid: uuid.UUID
    model_name: Literal["brats_mri_segmentation"]
    model_version: Literal["0.5.4"]
    model_weights_sha256: str
    model_input_checksum_sha256: str
    preprocessing_version: str
    device: str
    amp_enabled: bool
    roi_size: list[int]
    overlap: float
    threshold: float
    spatial_shape: list[int]
    affine_ras: list[list[float]]
    output_channel_order: list[SegmentationOutput]
    tc_mask: SegmentationMaskArtifactResponse
    wt_mask: SegmentationMaskArtifactResponse
    et_mask: SegmentationMaskArtifactResponse
    brats_labelmap: SegmentationMaskArtifactResponse
    voxel_counts: dict[str, int]
    runtime_seconds: float | None = None
    review_status: Literal["unreviewed", "accepted", "edited", "rejected"]
    clinician_modified: bool
    decision_state: Literal["pending"]
    segmentation_is_gbm_diagnosis: Literal[False] = False
    physical_volume_generated: Literal[False] = False
    anatomical_localization_generated: Literal[False] = False
    clinical_validation_claimed: Literal[False] = False
    background_execution_implemented: Literal[True] = True
    next_step: Literal["phase6_complete"] = "phase6_complete"


class SegmentationJobResponse(BaseModel):
    version: Literal["phase6_step6_background_segmentation_job_v1"]
    job_uuid: uuid.UUID
    study_uuid: uuid.UUID
    status: Literal["queued", "running", "complete", "failed"]
    model_input_checksum_sha256: str
    attempts: int
    max_attempts: int
    available_at: datetime
    claimed_at: datetime | None = None
    heartbeat_at: datetime | None = None
    lease_expires_at: datetime | None = None
    completed_at: datetime | None = None
    analysis_run_uuid: uuid.UUID | None = None
    segmentation_uuid: uuid.UUID | None = None
    last_error_code: str | None = None
    worker_assigned: bool
    result_available: bool
    background_execution_implemented: Literal[True] = True
    physical_volume_generated: Literal[False] = False
    anatomical_localization_generated: Literal[False] = False
    clinical_validation_claimed: Literal[False] = False
    next_step: Literal[
        "phase6_step6_background_execution_and_recovery",
        "phase6_complete",
    ]
