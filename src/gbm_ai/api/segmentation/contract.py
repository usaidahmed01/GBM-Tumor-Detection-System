from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final


SEGMENTATION_CONTRACT_VERSION: Final = (
    "phase6_step1_segmentation_input_contract_v1"
)

SEGMENTATION_BUNDLE_NAME: Final = "brats_mri_segmentation"
SEGMENTATION_ARCHITECTURE: Final = "SegResNet"

SEGMENTATION_INPUT_CHANNEL_ORDER: Final = (
    "T1C",
    "T1",
    "T2",
    "FLAIR",
)

SEGMENTATION_OUTPUT_CHANNEL_ORDER: Final = (
    "TC",
    "WT",
    "ET",
)

SEGMENTATION_REFERENCE_SPACING_MM: Final = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class SegmentationModelContract:
    contract_version: str
    bundle_name: str
    architecture: str
    required_input_channel_order: tuple[str, str, str, str]
    output_channel_order: tuple[str, str, str]
    reference_spacing_mm: tuple[float, float, float]
    requires_orientation_normalization: bool
    requires_alignment_validation: bool
    requires_reference_geometry_resampling: bool
    runtime_model_loading_implemented: bool
    inference_implemented: bool
    clinical_validation_claimed: bool


SEGMENTATION_MODEL_CONTRACT: Final = SegmentationModelContract(
    contract_version=SEGMENTATION_CONTRACT_VERSION,
    bundle_name=SEGMENTATION_BUNDLE_NAME,
    architecture=SEGMENTATION_ARCHITECTURE,
    required_input_channel_order=SEGMENTATION_INPUT_CHANNEL_ORDER,
    output_channel_order=SEGMENTATION_OUTPUT_CHANNEL_ORDER,
    reference_spacing_mm=SEGMENTATION_REFERENCE_SPACING_MM,
    requires_orientation_normalization=True,
    requires_alignment_validation=True,
    requires_reference_geometry_resampling=True,
    runtime_model_loading_implemented=False,
    inference_implemented=False,
    clinical_validation_claimed=False,
)


def segmentation_contract_dict() -> dict:
    return asdict(SEGMENTATION_MODEL_CONTRACT)