from __future__ import annotations

import math
import shutil
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gbm_ai.api.segmentation.bundle_runtime import (
    BUNDLE_MODEL_SHA256,
    BUNDLE_NAME,
    BUNDLE_OUTPUT_CHANNEL_ORDER,
    BUNDLE_OUTPUT_THRESHOLD,
    BUNDLE_OVERLAP,
    BUNDLE_ROI_SIZE,
    BUNDLE_SW_BATCH_SIZE,
    BUNDLE_VERSION,
    SegmentationBundleRuntimeError,
    load_frozen_network,
    validate_bundle_layout,
)
from gbm_ai.api.segmentation.model_input import (
    MODEL_INPUT_PREPROCESSING_VERSION,
    MODEL_INPUT_VERSION,
    SegmentationModelInputError,
    load_prepared_model_input,
)
from gbm_ai.api.storage.local import LocalObjectStore, StoredObject


SEGMENTATION_INFERENCE_VERSION = "phase6_step5_guarded_segmentation_inference_v1"
SEGMENTATION_POSTPROCESSING_VERSION = "brats_0.5.4_sigmoid_threshold_0.5_priority_labelmap_v1"


class SegmentationInferenceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SegmentationMaskArtifact:
    storage_key: str
    checksum_sha256: str
    size_bytes: int
    foreground_voxels: int

    @classmethod
    def from_stored(cls, stored: StoredObject, *, foreground_voxels: int) -> "SegmentationMaskArtifact":
        return cls(
            storage_key=stored.storage_key,
            checksum_sha256=stored.sha256,
            size_bytes=stored.size_bytes,
            foreground_voxels=int(foreground_voxels),
        )


@dataclass(frozen=True)
class SegmentationExecutionResult:
    device: str
    amp_enabled: bool
    runtime_seconds: float
    spatial_shape: tuple[int, int, int]
    affine_ras: np.ndarray
    tc: SegmentationMaskArtifact
    wt: SegmentationMaskArtifact
    et: SegmentationMaskArtifact
    labelmap: SegmentationMaskArtifact

    @property
    def voxel_counts(self) -> dict[str, int]:
        return {
            "TC": self.tc.foreground_voxels,
            "WT": self.wt.foreground_voxels,
            "ET": self.et.foreground_voxels,
        }


def resolve_inference_device(preference: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise SegmentationInferenceError(
            "TORCH_NOT_INSTALLED",
            "PyTorch is required for 3D segmentation inference",
        ) from exc

    normalized = str(preference or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise SegmentationInferenceError(
            "SEGMENTATION_INFERENCE_DEVICE_INVALID",
            "segmentation inference device must be auto, cpu, or cuda",
        )
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise SegmentationInferenceError(
                "SEGMENTATION_CUDA_UNAVAILABLE",
                "CUDA inference was requested but CUDA is not available",
            )
        return "cuda:0"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _validate_model_input_gate(model_input: dict, *, max_spatial_voxels: int) -> None:
    if model_input.get("version") != MODEL_INPUT_VERSION or model_input.get("status") != "ready":
        raise SegmentationInferenceError(
            "SEGMENTATION_MODEL_INPUT_NOT_READY",
            "Phase 6 Step 4 model input must be ready before SegResNet inference",
        )
    if model_input.get("bundle_name") != BUNDLE_NAME or model_input.get("bundle_version") != BUNDLE_VERSION:
        raise SegmentationInferenceError(
            "SEGMENTATION_MODEL_INPUT_BUNDLE_MISMATCH",
            "prepared model input does not match the frozen MONAI BraTS bundle",
        )
    if str(model_input.get("bundle_model_sha256") or "").lower() != BUNDLE_MODEL_SHA256:
        raise SegmentationInferenceError(
            "SEGMENTATION_MODEL_INPUT_WEIGHTS_MISMATCH",
            "prepared model input is not bound to the frozen model checksum",
        )
    contract = dict(model_input.get("inference_contract") or {})
    if (
        tuple(contract.get("roi_size") or ()) != BUNDLE_ROI_SIZE
        or int(contract.get("sw_batch_size", -1)) != BUNDLE_SW_BATCH_SIZE
        or not math.isclose(float(contract.get("overlap", -1.0)), BUNDLE_OVERLAP, abs_tol=1e-12)
        or str(contract.get("activation") or "") != "sigmoid"
        or not math.isclose(float(contract.get("threshold", -1.0)), BUNDLE_OUTPUT_THRESHOLD, abs_tol=1e-12)
    ):
        raise SegmentationInferenceError(
            "SEGMENTATION_INFERENCE_CONTRACT_DRIFT",
            "Step 4 inference contract differs from the frozen bundle settings",
        )
    spatial_shape = tuple(int(v) for v in (model_input.get("spatial_shape") or ()))
    if len(spatial_shape) != 3 or any(v <= 0 for v in spatial_shape):
        raise SegmentationInferenceError(
            "SEGMENTATION_MODEL_INPUT_SPATIAL_SHAPE_INVALID",
            "prepared model input has an invalid 3D spatial shape",
        )
    spatial_voxels = int(np.prod(spatial_shape, dtype=np.int64))
    if spatial_voxels > int(max_spatial_voxels):
        raise SegmentationInferenceError(
            "SEGMENTATION_INFERENCE_MEMORY_GUARD",
            "prepared model input exceeds the configured inference spatial-voxel safety limit",
        )


def binary_masks_to_brats_labelmap(tc: np.ndarray, wt: np.ndarray, et: np.ndarray) -> np.ndarray:
    """Apply the frozen bundle's ET > TC > WT priority label mapping.

    BraTS output label values follow the bundle inference config:
    background=0, TC-priority=1, WT-only=2, ET=4.
    """
    tc_bool = np.asarray(tc, dtype=bool)
    wt_bool = np.asarray(wt, dtype=bool)
    et_bool = np.asarray(et, dtype=bool)
    if tc_bool.shape != wt_bool.shape or tc_bool.shape != et_bool.shape or tc_bool.ndim != 3:
        raise SegmentationInferenceError(
            "SEGMENTATION_MASK_SHAPE_INVALID",
            "TC/WT/ET binary masks must share one 3D shape",
        )
    return np.where(
        et_bool,
        np.uint8(4),
        np.where(tc_bool, np.uint8(1), np.where(wt_bool, np.uint8(2), np.uint8(0))),
    ).astype(np.uint8, copy=False)


def _postprocess_logits(logits: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
    except ImportError as exc:
        raise SegmentationInferenceError("TORCH_NOT_INSTALLED", "PyTorch is required for postprocessing") from exc

    if not isinstance(logits, torch.Tensor):
        raise SegmentationInferenceError(
            "SEGMENTATION_INFERENCE_OUTPUT_TYPE_INVALID",
            "SegResNet inference output must be a torch.Tensor",
        )
    if logits.ndim != 5 or logits.shape[0] != 1 or logits.shape[1] != 3:
        raise SegmentationInferenceError(
            "SEGMENTATION_INFERENCE_OUTPUT_SHAPE_INVALID",
            "SegResNet inference output must have shape [1, 3, X, Y, Z]",
        )
    if not bool(torch.isfinite(logits).all()):
        raise SegmentationInferenceError(
            "SEGMENTATION_INFERENCE_OUTPUT_NONFINITE",
            "SegResNet produced non-finite logits",
        )

    probabilities = torch.sigmoid(logits.float())
    masks = probabilities >= float(BUNDLE_OUTPUT_THRESHOLD)
    masks_np = masks[0].to(device="cpu", dtype=torch.uint8).numpy()
    tc = np.asarray(masks_np[0], dtype=np.uint8)
    wt = np.asarray(masks_np[1], dtype=np.uint8)
    et = np.asarray(masks_np[2], dtype=np.uint8)
    labelmap = binary_masks_to_brats_labelmap(tc, wt, et)
    return tc, wt, et, labelmap


def _persist_nifti_uint8(
    storage: LocalObjectStore,
    study_uuid,
    *,
    category: str,
    data: np.ndarray,
    affine_ras: np.ndarray,
) -> StoredObject:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SegmentationInferenceError(
            "NIBABEL_NOT_INSTALLED",
            "NiBabel is required to persist 3D segmentation masks",
        ) from exc

    data = np.asarray(data, dtype=np.uint8)
    affine = np.asarray(affine_ras, dtype=np.float64)
    if data.ndim != 3 or affine.shape != (4, 4) or not bool(np.isfinite(affine).all()):
        raise SegmentationInferenceError(
            "SEGMENTATION_MASK_ARTIFACT_INVALID",
            "segmentation mask artifact geometry is invalid",
        )

    key = storage.generate_study_derived_key(study_uuid, category, suffix=".nii.gz")
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        image = nib.Nifti1Image(data, affine)
        image.set_data_dtype(np.uint8)
        image.set_qform(affine, code=1)
        image.set_sform(affine, code=1)
        nib.save(image, str(temp_path))
        with temp_path.open("rb") as source:
            return storage.put_stream(key, source)
    except SegmentationInferenceError:
        raise
    except Exception as exc:
        raise SegmentationInferenceError(
            "SEGMENTATION_MASK_STORE_FAILED",
            "generated segmentation mask could not be stored safely",
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)


def _safe_delete(storage: LocalObjectStore, keys: list[str]) -> None:
    for key in keys:
        try:
            if storage.exists(key):
                storage.delete(key)
        except Exception:
            pass


def execute_and_persist_segmentation(
    storage: LocalObjectStore,
    study_uuid,
    *,
    model_input: dict,
    bundle_dir: Path,
    device_preference: str,
    max_spatial_voxels: int,
) -> SegmentationExecutionResult:
    """Run the frozen SegResNet with guarded sliding-window inference and persist masks."""

    _validate_model_input_gate(model_input, max_spatial_voxels=max_spatial_voxels)
    try:
        image, affine_ras = load_prepared_model_input(storage, model_input)
    except SegmentationModelInputError as exc:
        raise SegmentationInferenceError(exc.code, str(exc)) from exc
    except Exception as exc:
        raise SegmentationInferenceError(
            "SEGMENTATION_MODEL_INPUT_LOAD_FAILED",
            "prepared Step 4 model-input artifact could not be loaded safely",
        ) from exc

    expected_shape = tuple(int(v) for v in model_input["spatial_shape"])
    if tuple(int(v) for v in image.shape) != (4, *expected_shape):
        raise SegmentationInferenceError(
            "SEGMENTATION_MODEL_INPUT_ARTIFACT_SHAPE_MISMATCH",
            "stored Step 4 model input no longer matches its persisted spatial shape",
        )

    try:
        validation = validate_bundle_layout(bundle_dir)
    except SegmentationBundleRuntimeError as exc:
        raise SegmentationInferenceError(exc.code, str(exc)) from exc
    if validation.model_sha256.lower() != BUNDLE_MODEL_SHA256:
        raise SegmentationInferenceError(
            "SEGMENTATION_BUNDLE_MODEL_CHECKSUM_MISMATCH",
            "frozen segmentation model checksum changed before inference",
        )

    device = resolve_inference_device(device_preference)
    try:
        import torch
        from monai.inferers import SlidingWindowInferer
    except ImportError as exc:
        raise SegmentationInferenceError(
            "SEGMENTATION_RUNTIME_DEPENDENCY_MISSING",
            "PyTorch and MONAI are required for 3D segmentation inference",
        ) from exc

    try:
        network, _ = load_frozen_network(validation.bundle_dir, device=device)
    except Exception as exc:
        code = getattr(exc, "code", "SEGMENTATION_NETWORK_LOAD_FAILED")
        raise SegmentationInferenceError(code, str(exc)) from exc

    compute_device = torch.device(device)
    output_device = torch.device("cpu") if device.startswith("cuda") else compute_device
    inferer = SlidingWindowInferer(
        roi_size=BUNDLE_ROI_SIZE,
        sw_batch_size=BUNDLE_SW_BATCH_SIZE,
        overlap=BUNDLE_OVERLAP,
        sw_device=compute_device,
        device=output_device,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0)
    amp_enabled = device.startswith("cuda")
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with autocast_context:
                logits = inferer(tensor, network)
        runtime_seconds = float(time.perf_counter() - start)
        tc, wt, et, labelmap = _postprocess_logits(logits)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message:
            raise SegmentationInferenceError(
                "SEGMENTATION_INFERENCE_OUT_OF_MEMORY",
                "SegResNet inference ran out of memory; no segmentation result was accepted",
            ) from exc
        raise SegmentationInferenceError(
            "SEGMENTATION_INFERENCE_RUNTIME_FAILED",
            "SegResNet sliding-window inference failed",
        ) from exc
    except SegmentationInferenceError:
        raise
    except Exception as exc:
        raise SegmentationInferenceError(
            "SEGMENTATION_INFERENCE_RUNTIME_FAILED",
            "SegResNet sliding-window inference failed",
        ) from exc
    finally:
        try:
            del network
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    for mask in (tc, wt, et, labelmap):
        if tuple(int(v) for v in mask.shape) != expected_shape:
            raise SegmentationInferenceError(
                "SEGMENTATION_MASK_SHAPE_MISMATCH",
                "generated segmentation mask shape does not match the Step 4 model input",
            )

    stored_keys: list[str] = []
    try:
        tc_stored = _persist_nifti_uint8(storage, study_uuid, category="segmentation_tc", data=tc, affine_ras=affine_ras)
        stored_keys.append(tc_stored.storage_key)
        wt_stored = _persist_nifti_uint8(storage, study_uuid, category="segmentation_wt", data=wt, affine_ras=affine_ras)
        stored_keys.append(wt_stored.storage_key)
        et_stored = _persist_nifti_uint8(storage, study_uuid, category="segmentation_et", data=et, affine_ras=affine_ras)
        stored_keys.append(et_stored.storage_key)
        label_stored = _persist_nifti_uint8(
            storage,
            study_uuid,
            category="segmentation_brats_labelmap",
            data=labelmap,
            affine_ras=affine_ras,
        )
        stored_keys.append(label_stored.storage_key)
    except Exception:
        _safe_delete(storage, stored_keys)
        raise

    return SegmentationExecutionResult(
        device=device,
        amp_enabled=amp_enabled,
        runtime_seconds=runtime_seconds,
        spatial_shape=expected_shape,
        affine_ras=np.asarray(affine_ras, dtype=np.float64),
        tc=SegmentationMaskArtifact.from_stored(tc_stored, foreground_voxels=int(np.count_nonzero(tc))),
        wt=SegmentationMaskArtifact.from_stored(wt_stored, foreground_voxels=int(np.count_nonzero(wt))),
        et=SegmentationMaskArtifact.from_stored(et_stored, foreground_voxels=int(np.count_nonzero(et))),
        labelmap=SegmentationMaskArtifact.from_stored(
            label_stored,
            foreground_voxels=int(np.count_nonzero(labelmap)),
        ),
    )
