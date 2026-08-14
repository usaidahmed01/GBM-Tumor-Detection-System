from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


ANATOMICAL_LOCALIZATION_VERSION = "phase7_step2_anatomical_localization_v1"
STANDARD_SPACE = "MNI152NLin6Asym"
STANDARD_TEMPLATE_NAME = "TemplateFlow_MNI152NLin6Asym_res-02_desc-brain_T1w"
ATLAS_NAME = "Harvard-Oxford cortical + subcortical structural atlases"
ATLAS_VERSION = "FSL5_maxprob_thr25_2mm_lateralized"
ATLAS_LICENSE = "CC BY-SA 4.0"
ATLAS_CORTICAL_ID = "cort-maxprob-thr25-2mm"
ATLAS_SUBCORTICAL_ID = "sub-maxprob-thr25-2mm"
ATLAS_SYMMETRIC_SPLIT = True
REGISTRATION_METHOD = "rigid_plus_affine_mattes_mutual_information"
REGISTRATION_METRIC = "MattesMutualInformation"
REGISTRATION_RANDOM_SEED = 42
REGISTRATION_SUPPORT_DICE_MIN = 0.40
SECONDARY_REGION_MIN_FRACTION_OF_WT = 0.05
MIDLINE_TOLERANCE_MM = 2.0


class AnatomicalLocalizationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RegionOverlap:
    atlas: str
    label_index: int
    label: str
    overlap_voxels: int
    fraction_of_wt: float

    def as_dict(self) -> dict:
        return {
            "atlas": self.atlas,
            "label_index": self.label_index,
            "label": self.label,
            "overlap_voxels": self.overlap_voxels,
            "fraction_of_wt": round(self.fraction_of_wt, 6),
        }


def localization_source_fingerprint(
    *,
    segmentation_uuid: str,
    quantification_uuid: str,
    wt_checksum_sha256: str,
    t1_checksum_sha256: str,
    atlas_manifest_checksum_sha256: str,
) -> str:
    payload = {
        "version": ANATOMICAL_LOCALIZATION_VERSION,
        "segmentation_uuid": str(segmentation_uuid),
        "quantification_uuid": str(quantification_uuid),
        "wt_checksum_sha256": wt_checksum_sha256.lower(),
        "t1_checksum_sha256": t1_checksum_sha256.lower(),
        "atlas_manifest_checksum_sha256": atlas_manifest_checksum_sha256.lower(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_integer_atlas(atlas: np.ndarray, expected_shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(atlas)
    if arr.ndim != 3 or tuple(int(v) for v in arr.shape) != expected_shape:
        raise AnatomicalLocalizationError(
            "LOCALIZATION_ATLAS_GEOMETRY_MISMATCH",
            "atlas geometry does not match the frozen standard-space template",
        )
    if not bool(np.isfinite(arr).all()):
        raise AnatomicalLocalizationError(
            "LOCALIZATION_ATLAS_NONFINITE",
            "atlas contains non-finite values",
        )
    rounded = np.rint(arr).astype(np.int32)
    if not np.allclose(arr, rounded, atol=1e-6, rtol=0.0):
        raise AnatomicalLocalizationError(
            "LOCALIZATION_ATLAS_NOT_INTEGER_LABELS",
            "deterministic atlas must contain integer labels",
        )
    if int(rounded.min()) < 0:
        raise AnatomicalLocalizationError(
            "LOCALIZATION_ATLAS_NEGATIVE_LABEL",
            "atlas contains an invalid negative label",
        )
    return rounded


def compute_region_overlaps(
    wt_mask: np.ndarray,
    atlas: np.ndarray,
    labels: list[str],
    *,
    atlas_name: str,
) -> list[RegionOverlap]:
    wt = np.asarray(wt_mask, dtype=bool)
    if wt.ndim != 3:
        raise AnatomicalLocalizationError(
            "LOCALIZATION_WT_MASK_INVALID",
            "WT mask must be a 3D binary volume in standard space",
        )
    total_wt = int(np.count_nonzero(wt))
    if total_wt <= 0:
        raise AnatomicalLocalizationError(
            "LOCALIZATION_WT_MASK_EMPTY",
            "anatomical localization cannot be generated from an empty WT mask",
        )
    atlas_labels = validate_integer_atlas(atlas, tuple(int(v) for v in wt.shape))

    overlaps: list[RegionOverlap] = []
    unique, counts = np.unique(atlas_labels[wt], return_counts=True)
    for index, count in zip(unique.tolist(), counts.tolist()):
        index = int(index)
        count = int(count)
        if index == 0 or count <= 0:
            continue
        if index >= len(labels):
            raise AnatomicalLocalizationError(
                "LOCALIZATION_ATLAS_LABEL_INDEX_INVALID",
                "atlas label index exceeds the frozen label table",
            )
        label = str(labels[index]).strip()
        if not label or label.lower() == "background":
            continue
        overlaps.append(
            RegionOverlap(
                atlas=atlas_name,
                label_index=index,
                label=label,
                overlap_voxels=count,
                fraction_of_wt=float(count / total_wt),
            )
        )

    overlaps.sort(key=lambda item: (-item.overlap_voxels, item.atlas, item.label))
    return overlaps


def merge_region_overlaps(*groups: Iterable[RegionOverlap]) -> list[RegionOverlap]:
    merged: list[RegionOverlap] = []
    for group in groups:
        merged.extend(list(group))
    merged.sort(key=lambda item: (-item.overlap_voxels, item.atlas, item.label))
    return merged


def centroid_world_mm(mask: np.ndarray, affine_ras: np.ndarray) -> tuple[float, float, float]:
    foreground = np.argwhere(np.asarray(mask, dtype=bool))
    if foreground.size == 0:
        raise AnatomicalLocalizationError(
            "LOCALIZATION_WT_MASK_EMPTY",
            "cannot calculate a standard-space centroid from an empty WT mask",
        )
    affine = np.asarray(affine_ras, dtype=np.float64)
    if affine.shape != (4, 4) or not bool(np.isfinite(affine).all()):
        raise AnatomicalLocalizationError(
            "LOCALIZATION_STANDARD_AFFINE_INVALID",
            "standard-space affine must be a finite 4x4 matrix",
        )
    voxel_center = np.mean(foreground.astype(np.float64), axis=0)
    world = affine @ np.array([*voxel_center.tolist(), 1.0], dtype=np.float64)
    return tuple(round(float(v), 3) for v in world[:3])


def hemisphere_from_standard_mask(mask: np.ndarray, affine_ras: np.ndarray) -> str:
    foreground = np.argwhere(np.asarray(mask, dtype=bool))
    if foreground.size == 0:
        raise AnatomicalLocalizationError(
            "LOCALIZATION_WT_MASK_EMPTY",
            "cannot determine laterality from an empty WT mask",
        )
    affine = np.asarray(affine_ras, dtype=np.float64)
    homogeneous = np.c_[foreground.astype(np.float64), np.ones(len(foreground))]
    x = (homogeneous @ affine.T)[:, 0]
    left = int(np.count_nonzero(x < -MIDLINE_TOLERANCE_MM))
    right = int(np.count_nonzero(x > MIDLINE_TOLERANCE_MM))
    total_lateral = left + right
    if total_lateral == 0:
        return "midline"
    left_fraction = left / total_lateral
    right_fraction = right / total_lateral
    if left_fraction >= 0.60:
        return "left"
    if right_fraction >= 0.60:
        return "right"
    return "bilateral"


def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    if aa.shape != bb.shape:
        raise AnatomicalLocalizationError(
            "LOCALIZATION_QC_MASK_SHAPE_MISMATCH",
            "registration QC masks must share the same standard-space shape",
        )
    denom = int(np.count_nonzero(aa)) + int(np.count_nonzero(bb))
    if denom == 0:
        return 1.0
    intersection = int(np.count_nonzero(aa & bb))
    value = float(2.0 * intersection / denom)
    if not math.isfinite(value):
        raise AnatomicalLocalizationError(
            "LOCALIZATION_REGISTRATION_QC_INVALID",
            "registration QC produced a non-finite overlap score",
        )
    return value
