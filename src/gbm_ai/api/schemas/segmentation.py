from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel


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

    required_input_channel_order: list[
        SegmentationSequence
    ]

    output_channel_order: list[
        SegmentationOutput
    ]

    reference_spacing_mm: list[float]

    requires_orientation_normalization: Literal[True] = True
    requires_alignment_validation: Literal[True] = True
    requires_reference_geometry_resampling: Literal[True] = True

    runtime_model_loading_implemented: Literal[False] = False
    inference_implemented: Literal[False] = False
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