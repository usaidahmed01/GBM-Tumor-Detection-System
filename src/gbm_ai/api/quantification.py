from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gbm_ai.api.storage.local import LocalObjectStore


PHYSICAL_QUANTIFICATION_VERSION = "phase7_step1_physical_quantification_v1"
GEOMETRY_ATOL = 1e-4


class PhysicalQuantificationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PhysicalGeometry:
    voxel_spacing_mm: tuple[float, float, float]
    voxel_volume_mm3: float
    axial_pixel_area_mm2: float


@dataclass(frozen=True)
class RegionMeasurement:
    region: str
    voxel_count: int
    volume_mm3: float
    volume_cm3: float
    max_axial_area_mm2: float
    max_axial_slice_index: int | None
    axial_nonzero_slice_count: int


def validate_physical_geometry(affine_ras: Any) -> PhysicalGeometry:
    affine = np.asarray(affine_ras, dtype=np.float64)
    if affine.shape != (4, 4) or not bool(np.isfinite(affine).all()):
        raise PhysicalQuantificationError(
            "QUANTIFICATION_AFFINE_INVALID",
            "segmentation affine must be a finite 4x4 physical-space transform",
        )
    if not np.allclose(affine[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=GEOMETRY_ATOL):
        raise PhysicalQuantificationError(
            "QUANTIFICATION_AFFINE_HOMOGENEOUS_ROW_INVALID",
            "segmentation affine has an invalid homogeneous row",
        )

    basis = affine[:3, :3]
    spacing = tuple(float(np.linalg.norm(basis[:, axis])) for axis in range(3))
    if any((not math.isfinite(value) or value <= 0.0) for value in spacing):
        raise PhysicalQuantificationError(
            "QUANTIFICATION_SPACING_INVALID",
            "segmentation affine does not encode valid positive voxel spacing",
        )

    voxel_volume = float(abs(np.linalg.det(basis)))
    axial_pixel_area = float(np.linalg.norm(np.cross(basis[:, 0], basis[:, 1])))
    if (
        not math.isfinite(voxel_volume)
        or voxel_volume <= 1e-9
        or not math.isfinite(axial_pixel_area)
        or axial_pixel_area <= 1e-9
    ):
        raise PhysicalQuantificationError(
            "QUANTIFICATION_PHYSICAL_GEOMETRY_DEGENERATE",
            "segmentation geometry is degenerate and cannot support physical measurements",
        )

    return PhysicalGeometry(
        voxel_spacing_mm=spacing,
        voxel_volume_mm3=voxel_volume,
        axial_pixel_area_mm2=axial_pixel_area,
    )


def validate_binary_mask(mask: Any, *, expected_shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 3 or tuple(int(v) for v in array.shape) != expected_shape:
        raise PhysicalQuantificationError(
            "QUANTIFICATION_MASK_SHAPE_MISMATCH",
            "segmentation mask shape does not match the persisted segmentation geometry",
        )
    if not bool(np.isfinite(array).all()):
        raise PhysicalQuantificationError(
            "QUANTIFICATION_MASK_NONFINITE",
            "segmentation mask contains non-finite values",
        )
    if not bool(np.isin(array, (0, 1)).all()):
        raise PhysicalQuantificationError(
            "QUANTIFICATION_MASK_NOT_BINARY",
            "WT/TC/ET artifacts must be binary before physical quantification",
        )
    return np.asarray(array, dtype=np.uint8)


def measure_region(
    region: str,
    mask: np.ndarray,
    geometry: PhysicalGeometry,
) -> tuple[RegionMeasurement, list[dict[str, float | int]]]:
    foreground = np.asarray(mask, dtype=bool)
    voxel_count = int(np.count_nonzero(foreground))
    volume_mm3 = float(voxel_count * geometry.voxel_volume_mm3)
    volume_cm3 = float(volume_mm3 / 1000.0)

    counts_by_axial_slice = np.count_nonzero(foreground, axis=(0, 1)).astype(np.int64)
    nonzero_indices = np.flatnonzero(counts_by_axial_slice)
    per_slice: list[dict[str, float | int]] = []
    for index in nonzero_indices.tolist():
        count = int(counts_by_axial_slice[index])
        per_slice.append(
            {
                "slice_index": int(index),
                "foreground_voxels": count,
                "area_mm2": round(float(count * geometry.axial_pixel_area_mm2), 6),
            }
        )

    if nonzero_indices.size:
        max_index = int(np.argmax(counts_by_axial_slice))
        max_area = float(counts_by_axial_slice[max_index] * geometry.axial_pixel_area_mm2)
    else:
        max_index = None
        max_area = 0.0

    measurement = RegionMeasurement(
        region=region,
        voxel_count=voxel_count,
        volume_mm3=volume_mm3,
        volume_cm3=volume_cm3,
        max_axial_area_mm2=max_area,
        max_axial_slice_index=max_index,
        axial_nonzero_slice_count=int(nonzero_indices.size),
    )
    return measurement, per_slice


def source_fingerprint(
    *,
    segmentation_uuid: str,
    spatial_shape: list[int],
    affine_ras: list[list[float]],
    mask_checksums: dict[str, str],
) -> str:
    payload = {
        "version": PHYSICAL_QUANTIFICATION_VERSION,
        "segmentation_uuid": str(segmentation_uuid),
        "spatial_shape": [int(v) for v in spatial_shape],
        "affine_ras": affine_ras,
        "mask_checksums": {key: str(mask_checksums[key]).lower() for key in sorted(mask_checksums)},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_nifti_from_storage(
    storage: LocalObjectStore,
    storage_key: str,
    expected_checksum_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        checksum_valid = storage.verify_checksum(storage_key, expected_checksum_sha256)
    except Exception as exc:
        raise PhysicalQuantificationError(
            "QUANTIFICATION_SOURCE_ARTIFACT_UNAVAILABLE",
            "segmentation artifact is unavailable in protected storage",
        ) from exc
    if not checksum_valid:
        raise PhysicalQuantificationError(
            "QUANTIFICATION_SOURCE_CHECKSUM_MISMATCH",
            "segmentation artifact failed protected-storage checksum validation",
        )
    try:
        import nibabel as nib
    except ImportError as exc:
        raise PhysicalQuantificationError(
            "NIBABEL_NOT_INSTALLED",
            "NiBabel is required for physical segmentation quantification",
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        with storage.open_read(storage_key) as source, temp_path.open("wb") as target:
            shutil.copyfileobj(source, target)
        image = nib.load(str(temp_path))
        data = np.asanyarray(image.dataobj)
        affine = np.asarray(image.affine, dtype=np.float64)
        return np.asarray(data), affine
    except PhysicalQuantificationError:
        raise
    except Exception as exc:
        raise PhysicalQuantificationError(
            "QUANTIFICATION_MASK_LOAD_FAILED",
            "segmentation mask could not be loaded as a NIfTI volume",
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)


def load_and_validate_mask(
    storage: LocalObjectStore,
    *,
    storage_key: str,
    checksum_sha256: str,
    expected_shape: tuple[int, int, int],
    expected_affine_ras: np.ndarray,
) -> np.ndarray:
    data, affine = _load_nifti_from_storage(storage, storage_key, checksum_sha256)
    if affine.shape != (4, 4) or not np.allclose(
        affine,
        expected_affine_ras,
        rtol=0.0,
        atol=GEOMETRY_ATOL,
    ):
        raise PhysicalQuantificationError(
            "QUANTIFICATION_MASK_AFFINE_MISMATCH",
            "stored segmentation mask no longer matches the persisted physical geometry",
        )
    return validate_binary_mask(data, expected_shape=expected_shape)
