from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from gbm_ai.api.segmentation.bundle_runtime import (
    BUNDLE_INPUT_CHANNEL_ORDER,
    BUNDLE_MODEL_SHA256,
    BUNDLE_NAME,
    BUNDLE_NORMALIZE_CHANNEL_WISE,
    BUNDLE_NORMALIZE_NONZERO,
    BUNDLE_OVERLAP,
    BUNDLE_OUTPUT_THRESHOLD,
    BUNDLE_ROI_SIZE,
    BUNDLE_SW_BATCH_SIZE,
    BUNDLE_VERSION,
)
from gbm_ai.api.storage.local import LocalObjectStore, StoredObject


MODEL_INPUT_VERSION = "phase6_step4_monai_model_input_v1"
MODEL_INPUT_PREPROCESSING_VERSION = "brats_0.5.4_nonzero_channelwise_normalization_v1"
MAX_MODEL_INPUT_SPATIAL_VOXELS = 20_000_000
MIN_NONZERO_VOXELS_PER_CHANNEL = 128
MIN_NONZERO_STD = 1e-6


class SegmentationModelInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SegmentationModelInputError(
            "NIBABEL_NOT_INSTALLED",
            "NiBabel is required for Phase 6 model-input preparation",
        ) from exc
    return nib


def _require_monai_normalizer():
    try:
        from monai.transforms import NormalizeIntensity
    except ImportError as exc:
        raise SegmentationModelInputError(
            "MONAI_NOT_INSTALLED",
            "MONAI is required for bundle-compatible intensity normalization",
        ) from exc
    return NormalizeIntensity


def _load_geometry_channel(
    storage: LocalObjectStore,
    item: dict,
) -> tuple[np.ndarray, np.ndarray]:
    nib = _require_nibabel()
    key = str(item.get("storage_key") or "")
    checksum = str(item.get("checksum_sha256") or "")
    if not key or not checksum:
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_GEOMETRY_ARTIFACT_REFERENCE_INVALID",
            "model-geometry channel has an incomplete protected-storage reference",
        )
    if not storage.verify_checksum(key, checksum):
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_GEOMETRY_ARTIFACT_CHECKSUM_MISMATCH",
            "model-geometry channel checksum verification failed before normalization",
        )

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp:
        temp_path = Path(temp.name)
        with storage.open_read(key) as source:
            shutil.copyfileobj(source, temp)

    try:
        image = nib.load(str(temp_path))
        data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
        affine = np.asarray(image.affine, dtype=np.float64)
    except Exception as exc:
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_GEOMETRY_ARTIFACT_READ_FAILED",
            "model-geometry NIfTI artifact could not be read",
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    if data.ndim != 3 or not bool(np.isfinite(data).all()):
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_CHANNEL_INVALID",
            "model-geometry channel is not a finite 3D float volume",
        )
    if affine.shape != (4, 4) or not bool(np.isfinite(affine).all()):
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_AFFINE_INVALID",
            "model-geometry channel has an invalid affine",
        )
    return data, affine


def _nonzero_stats(data: np.ndarray) -> dict:
    mask = data != 0
    count = int(np.count_nonzero(mask))
    if count < MIN_NONZERO_VOXELS_PER_CHANNEL:
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_NONZERO_SUPPORT_TOO_SMALL",
            "MRI channel has too few non-zero voxels for bundle-compatible normalization",
        )
    values = np.asarray(data[mask], dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= MIN_NONZERO_STD:
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_INTENSITY_VARIANCE_INVALID",
            "MRI channel has insufficient non-zero intensity variance for normalization",
        )
    return {
        "nonzero_voxels": count,
        "mean": round(mean, 6),
        "std": round(std, 6),
    }


def build_normalized_model_input(
    storage: LocalObjectStore,
    model_geometry: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if model_geometry.get("status") != "ready":
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_GEOMETRY_NOT_READY",
            "Phase 6 Step 3 model geometry must be ready before model-input normalization",
        )
    if tuple(model_geometry.get("channel_order") or ()) != BUNDLE_INPUT_CHANNEL_ORDER:
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_CHANNEL_ORDER_MISMATCH",
            "Step 3 channel order is not T1C/T1/T2/FLAIR",
        )

    by_sequence = {
        str(item.get("sequence")): item
        for item in list(model_geometry.get("channels") or [])
        if isinstance(item, dict)
    }
    if set(by_sequence) != set(BUNDLE_INPUT_CHANNEL_ORDER):
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_CHANNEL_SET_INVALID",
            "Step 3 model geometry must contain exactly T1C/T1/T2/FLAIR",
        )

    arrays: list[np.ndarray] = []
    reference_affine: np.ndarray | None = None
    reference_shape: tuple[int, int, int] | None = None
    before_stats: list[dict] = []

    for sequence in BUNDLE_INPUT_CHANNEL_ORDER:
        data, affine = _load_geometry_channel(storage, by_sequence[sequence])
        if reference_shape is None:
            reference_shape = tuple(int(v) for v in data.shape)
            reference_affine = affine
            spatial_voxels = int(np.prod(reference_shape, dtype=np.int64))
            if spatial_voxels > MAX_MODEL_INPUT_SPATIAL_VOXELS:
                raise SegmentationModelInputError(
                    "SEGMENTATION_MODEL_INPUT_MEMORY_GUARD",
                    "1 mm model geometry is too large for the current CPU-friendly normalization safety limit",
                )
        elif tuple(int(v) for v in data.shape) != reference_shape:
            raise SegmentationModelInputError(
                "SEGMENTATION_MODEL_INPUT_SHAPE_MISMATCH",
                "Step 3 model-geometry channels no longer share one spatial shape",
            )
        if reference_affine is not None and not np.allclose(
            affine,
            reference_affine,
            atol=1e-5,
            rtol=0.0,
        ):
            raise SegmentationModelInputError(
                "SEGMENTATION_MODEL_INPUT_AFFINE_MISMATCH",
                "Step 3 model-geometry channels no longer share one affine",
            )

        stats = _nonzero_stats(data)
        stats["sequence"] = sequence
        before_stats.append(stats)
        arrays.append(data)

    stacked = np.stack(arrays, axis=0).astype(np.float32, copy=False)
    NormalizeIntensity = _require_monai_normalizer()
    normalizer = NormalizeIntensity(
        nonzero=BUNDLE_NORMALIZE_NONZERO,
        channel_wise=BUNDLE_NORMALIZE_CHANNEL_WISE,
    )
    try:
        normalized = normalizer(stacked)
        if hasattr(normalized, "detach"):
            normalized = normalized.detach().cpu().numpy()
        normalized = np.asarray(normalized, dtype=np.float32)
    except Exception as exc:
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_NORMALIZATION_FAILED",
            "MONAI bundle-compatible intensity normalization failed",
        ) from exc

    if normalized.shape != stacked.shape or not bool(np.isfinite(normalized).all()):
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_NORMALIZED_ARRAY_INVALID",
            "normalized model input has invalid shape or non-finite values",
        )

    channel_stats: list[dict] = []
    for index, sequence in enumerate(BUNDLE_INPUT_CHANNEL_ORDER):
        after = _nonzero_stats(normalized[index])
        before = before_stats[index]
        channel_stats.append(
            {
                "sequence": sequence,
                "nonzero_voxels": before["nonzero_voxels"],
                "mean_before": before["mean"],
                "std_before": before["std"],
                "mean_after": after["mean"],
                "std_after": after["std"],
            }
        )

    assert reference_affine is not None
    return normalized, reference_affine, channel_stats


def persist_normalized_model_input(
    storage: LocalObjectStore,
    study_uuid,
    *,
    image: np.ndarray,
    affine_ras: np.ndarray,
) -> StoredObject:
    key = storage.generate_study_derived_key(
        study_uuid,
        "segmentation_model_input",
        suffix=".npz",
    )
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        np.savez_compressed(
            temp_path,
            image=np.asarray(image, dtype=np.float32),
            affine_ras=np.asarray(affine_ras, dtype=np.float64),
            channel_order=np.asarray(BUNDLE_INPUT_CHANNEL_ORDER, dtype="U8"),
        )
        with temp_path.open("rb") as source:
            return storage.put_stream(key, source)
    except Exception as exc:
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_STORE_FAILED",
            "normalized four-channel model input could not be stored safely",
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)


def load_prepared_model_input(
    storage: LocalObjectStore,
    summary: dict,
) -> tuple[np.ndarray, np.ndarray]:
    key = str(summary.get("storage_key") or "")
    checksum = str(summary.get("checksum_sha256") or "")
    if not key or not checksum:
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_REFERENCE_INVALID",
            "prepared model input has an incomplete storage reference",
        )
    if not storage.verify_checksum(key, checksum):
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_CHECKSUM_MISMATCH",
            "prepared model input checksum verification failed",
        )

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp:
        temp_path = Path(temp.name)
        with storage.open_read(key) as source:
            shutil.copyfileobj(source, temp)
    try:
        with np.load(temp_path, allow_pickle=False) as payload:
            image = np.asarray(payload["image"], dtype=np.float32)
            affine = np.asarray(payload["affine_ras"], dtype=np.float64)
            channel_order = tuple(str(v) for v in payload["channel_order"].tolist())
    except Exception as exc:
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_ARTIFACT_READ_FAILED",
            "prepared model-input artifact could not be read safely",
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    if image.ndim != 4 or image.shape[0] != 4 or not bool(np.isfinite(image).all()):
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_ARTIFACT_INVALID",
            "prepared model input must be finite [4, X, Y, Z] float data",
        )
    if channel_order != BUNDLE_INPUT_CHANNEL_ORDER:
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_ARTIFACT_CHANNEL_ORDER_INVALID",
            "prepared model input channel order is not T1C/T1/T2/FLAIR",
        )
    if affine.shape != (4, 4) or not bool(np.isfinite(affine).all()):
        raise SegmentationModelInputError(
            "SEGMENTATION_MODEL_INPUT_ARTIFACT_AFFINE_INVALID",
            "prepared model input affine is invalid",
        )
    return image, affine


def model_input_summary(
    stored: StoredObject,
    *,
    image: np.ndarray,
    affine_ras: np.ndarray,
    channel_stats: list[dict],
) -> dict:
    return {
        "version": MODEL_INPUT_VERSION,
        "status": "ready",
        "preprocessing_version": MODEL_INPUT_PREPROCESSING_VERSION,
        "bundle_name": BUNDLE_NAME,
        "bundle_version": BUNDLE_VERSION,
        "bundle_model_sha256": BUNDLE_MODEL_SHA256,
        "channel_order": list(BUNDLE_INPUT_CHANNEL_ORDER),
        "shape": [int(v) for v in image.shape],
        "spatial_shape": [int(v) for v in image.shape[1:]],
        "dtype": "float32",
        "affine_ras": [
            [round(float(value), 6) for value in row]
            for row in affine_ras
        ],
        "storage_key": stored.storage_key,
        "checksum_sha256": stored.sha256,
        "size_bytes": stored.size_bytes,
        "normalization": {
            "transform": "MONAI NormalizeIntensity",
            "nonzero": BUNDLE_NORMALIZE_NONZERO,
            "channel_wise": BUNDLE_NORMALIZE_CHANNEL_WISE,
            "channel_stats": channel_stats,
        },
        "inference_contract": {
            "roi_size": list(BUNDLE_ROI_SIZE),
            "sw_batch_size": BUNDLE_SW_BATCH_SIZE,
            "overlap": BUNDLE_OVERLAP,
            "activation": "sigmoid",
            "threshold": BUNDLE_OUTPUT_THRESHOLD,
        },
        "domain_warnings": [
            "BUNDLE_TRAINING_DOMAIN_BRATS2018",
            "SKULL_STRIPPING_OR_EQUIVALENT_BRATS_PREPROCESSING_NOT_AUTOMATICALLY_VALIDATED",
        ],
        "intensity_normalization_performed": True,
        "crop_pad_performed": False,
        "sliding_window_padding_deferred_to_inferer": True,
        "bundle_runtime_loading_verified_per_environment": False,
        "model_execution_started": False,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,
        "next_step": "phase6_step5_segmentation_inference",
    }
