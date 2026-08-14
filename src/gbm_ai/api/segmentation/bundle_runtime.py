from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


BUNDLE_RUNTIME_VERSION = "phase6_step4_bundle_runtime_v1"
BUNDLE_REPOSITORY = "MONAI/brats_mri_segmentation"
BUNDLE_NAME = "brats_mri_segmentation"
BUNDLE_VERSION = "0.5.4"
BUNDLE_MODEL_RELATIVE_PATH = Path("models/model.pt")
BUNDLE_METADATA_RELATIVE_PATH = Path("configs/metadata.json")
BUNDLE_INFERENCE_RELATIVE_PATH = Path("configs/inference.json")
BUNDLE_MODEL_SHA256 = "860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae"

BUNDLE_INPUT_CHANNEL_ORDER = ("T1C", "T1", "T2", "FLAIR")
BUNDLE_OUTPUT_CHANNEL_ORDER = ("TC", "WT", "ET")
BUNDLE_NETWORK_BLOCKS_DOWN = (1, 2, 2, 4)
BUNDLE_NETWORK_BLOCKS_UP = (1, 1, 1)
BUNDLE_NETWORK_INIT_FILTERS = 16
BUNDLE_NETWORK_DROPOUT = 0.2
BUNDLE_ROI_SIZE = (240, 240, 160)
BUNDLE_SW_BATCH_SIZE = 1
BUNDLE_OVERLAP = 0.5
BUNDLE_OUTPUT_THRESHOLD = 0.5
BUNDLE_NORMALIZE_NONZERO = True
BUNDLE_NORMALIZE_CHANNEL_WISE = True

DOWNLOAD_ALLOW_PATTERNS = (
    "models/model.pt",
    "configs/inference.json",
    "configs/metadata.json",
    "LICENSE",
    "docs/README.md",
    "docs/data_license.txt",
)


class SegmentationBundleRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class BundleValidationResult:
    bundle_dir: Path
    version: str
    model_sha256: str
    metadata_monai_version: str | None
    metadata_pytorch_version: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime_version": BUNDLE_RUNTIME_VERSION,
            "bundle_name": BUNDLE_NAME,
            "bundle_version": self.version,
            "bundle_repository": BUNDLE_REPOSITORY,
            "bundle_dir": str(self.bundle_dir),
            "model_sha256": self.model_sha256,
            "metadata_monai_version": self.metadata_monai_version,
            "metadata_pytorch_version": self.metadata_pytorch_version,
            "input_channel_order": list(BUNDLE_INPUT_CHANNEL_ORDER),
            "output_channel_order": list(BUNDLE_OUTPUT_CHANNEL_ORDER),
            "roi_size": list(BUNDLE_ROI_SIZE),
            "sw_batch_size": BUNDLE_SW_BATCH_SIZE,
            "overlap": BUNDLE_OVERLAP,
            "output_threshold": BUNDLE_OUTPUT_THRESHOLD,
            "normalize_nonzero": BUNDLE_NORMALIZE_NONZERO,
            "normalize_channel_wise": BUNDLE_NORMALIZE_CHANNEL_WISE,
        }


def frozen_bundle_dir(bundle_root: Path) -> Path:
    return Path(bundle_root).expanduser().resolve() / BUNDLE_NAME


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SegmentationBundleRuntimeError(
            code,
            f"could not read valid JSON from {path.name}",
        ) from exc
    if not isinstance(payload, dict):
        raise SegmentationBundleRuntimeError(code, f"{path.name} must contain a JSON object")
    return payload


def _normalize_channel_def(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    ordered: list[str] = []
    for index in range(len(value)):
        item = value.get(str(index), value.get(index))
        if item is None:
            return ()
        ordered.append(str(item))
    return tuple(ordered)


def validate_bundle_layout(
    bundle_dir: Path,
    *,
    expected_model_sha256: str = BUNDLE_MODEL_SHA256,
) -> BundleValidationResult:
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    model_path = bundle_dir / BUNDLE_MODEL_RELATIVE_PATH
    metadata_path = bundle_dir / BUNDLE_METADATA_RELATIVE_PATH
    inference_path = bundle_dir / BUNDLE_INFERENCE_RELATIVE_PATH

    for path, code in (
        (model_path, "SEGMENTATION_BUNDLE_MODEL_MISSING"),
        (metadata_path, "SEGMENTATION_BUNDLE_METADATA_MISSING"),
        (inference_path, "SEGMENTATION_BUNDLE_INFERENCE_CONFIG_MISSING"),
    ):
        if not path.is_file():
            raise SegmentationBundleRuntimeError(code, f"required bundle file is missing: {path}")

    model_sha256 = _sha256_file(model_path)
    if model_sha256.lower() != expected_model_sha256.lower():
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_MODEL_CHECKSUM_MISMATCH",
            "MONAI BraTS model.pt checksum does not match the frozen project artifact",
        )

    metadata = _read_json(metadata_path, code="SEGMENTATION_BUNDLE_METADATA_INVALID")
    if str(metadata.get("version")) != BUNDLE_VERSION:
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_VERSION_MISMATCH",
            f"bundle metadata version must be {BUNDLE_VERSION}",
        )

    network_data_format = dict(metadata.get("network_data_format") or {})
    input_def = (
        dict(network_data_format.get("inputs") or {})
        .get("image", {})
        .get("channel_def")
    )
    output_def = (
        dict(network_data_format.get("outputs") or {})
        .get("pred", {})
        .get("channel_def")
    )
    metadata_inputs = _normalize_channel_def(input_def)
    metadata_outputs = _normalize_channel_def(output_def)

    expected_metadata_inputs = ("T1c", "T1", "T2", "FLAIR")
    expected_metadata_outputs = ("Tumor core", "Whole tumor", "Enhancing tumor")
    if metadata_inputs != expected_metadata_inputs:
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_INPUT_CONTRACT_MISMATCH",
            "bundle metadata no longer declares T1c/T1/T2/FLAIR in the frozen four-channel order",
        )
    if metadata_outputs != expected_metadata_outputs:
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_OUTPUT_CONTRACT_MISMATCH",
            "bundle metadata no longer declares TC/WT/ET in the frozen three-channel order",
        )

    inference = _read_json(inference_path, code="SEGMENTATION_BUNDLE_INFERENCE_CONFIG_INVALID")
    network_def = dict(inference.get("network_def") or {})
    if (
        str(network_def.get("_target_")) != "SegResNet"
        or tuple(network_def.get("blocks_down") or ()) != BUNDLE_NETWORK_BLOCKS_DOWN
        or tuple(network_def.get("blocks_up") or ()) != BUNDLE_NETWORK_BLOCKS_UP
        or int(network_def.get("init_filters", -1)) != BUNDLE_NETWORK_INIT_FILTERS
        or int(network_def.get("in_channels", -1)) != 4
        or int(network_def.get("out_channels", -1)) != 3
        or float(network_def.get("dropout_prob", -1.0)) != BUNDLE_NETWORK_DROPOUT
    ):
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_NETWORK_CONFIG_MISMATCH",
            "bundle SegResNet network definition differs from the frozen project contract",
        )

    preprocessing = dict(inference.get("preprocessing") or {})
    transforms = list(preprocessing.get("transforms") or [])
    normalize = next(
        (
            item
            for item in transforms
            if isinstance(item, dict) and item.get("_target_") == "NormalizeIntensityd"
        ),
        None,
    )
    if not isinstance(normalize, dict) or normalize.get("nonzero") is not True or normalize.get("channel_wise") is not True:
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_NORMALIZATION_CONFIG_MISMATCH",
            "bundle inference normalization must remain nonzero=True and channel_wise=True",
        )

    inferer = dict(inference.get("inferer") or {})
    if (
        str(inferer.get("_target_")) != "SlidingWindowInferer"
        or tuple(inferer.get("roi_size") or ()) != BUNDLE_ROI_SIZE
        or int(inferer.get("sw_batch_size", -1)) != BUNDLE_SW_BATCH_SIZE
        or float(inferer.get("overlap", -1.0)) != BUNDLE_OVERLAP
    ):
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_INFERER_CONFIG_MISMATCH",
            "bundle sliding-window inference settings differ from the frozen project contract",
        )

    postprocessing = dict(inference.get("postprocessing") or {})
    post_transforms = list(postprocessing.get("transforms") or [])
    sigmoid = next(
        (
            item
            for item in post_transforms
            if isinstance(item, dict) and item.get("_target_") == "Activationsd"
        ),
        None,
    )
    discretize = next(
        (
            item
            for item in post_transforms
            if isinstance(item, dict) and item.get("_target_") == "AsDiscreted"
        ),
        None,
    )
    if not isinstance(sigmoid, dict) or sigmoid.get("sigmoid") is not True:
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_POSTPROCESSING_MISMATCH",
            "bundle output activation must remain sigmoid",
        )
    if not isinstance(discretize, dict) or float(discretize.get("threshold", -1.0)) != BUNDLE_OUTPUT_THRESHOLD:
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_THRESHOLD_MISMATCH",
            f"bundle output threshold must remain {BUNDLE_OUTPUT_THRESHOLD}",
        )

    return BundleValidationResult(
        bundle_dir=bundle_dir,
        version=BUNDLE_VERSION,
        model_sha256=model_sha256,
        metadata_monai_version=(
            str(metadata.get("monai_version")) if metadata.get("monai_version") is not None else None
        ),
        metadata_pytorch_version=(
            str(metadata.get("pytorch_version")) if metadata.get("pytorch_version") is not None else None
        ),
    )


def download_frozen_bundle(bundle_root: Path) -> BundleValidationResult:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SegmentationBundleRuntimeError(
            "HUGGINGFACE_HUB_NOT_INSTALLED",
            "huggingface-hub is required to download the frozen MONAI bundle",
        ) from exc

    target = frozen_bundle_dir(bundle_root)
    target.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=BUNDLE_REPOSITORY,
            revision=BUNDLE_VERSION,
            local_dir=target,
            allow_patterns=list(DOWNLOAD_ALLOW_PATTERNS),
        )
    except Exception as exc:
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_DOWNLOAD_FAILED",
            "could not download the frozen MONAI BraTS bundle from its official Hugging Face repository",
        ) from exc
    return validate_bundle_layout(target)


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_CHECKPOINT_FORMAT_INVALID",
            "model.pt did not contain a supported state-dictionary mapping",
        )

    for key in ("model", "state_dict", "network"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    if checkpoint and all(isinstance(key, str) for key in checkpoint):
        return checkpoint

    raise SegmentationBundleRuntimeError(
        "SEGMENTATION_BUNDLE_CHECKPOINT_FORMAT_INVALID",
        "model.pt did not contain a recognized model state dictionary",
    )


def _strip_uniform_prefix(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    keys = list(state_dict)
    if keys and all(key.startswith(prefix) for key in keys):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return dict(state_dict)


def load_frozen_network(bundle_dir: Path, *, device: str = "cpu"):
    validation = validate_bundle_layout(bundle_dir)
    try:
        import torch
        from monai.networks.nets import SegResNet
    except ImportError as exc:
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_RUNTIME_DEPENDENCY_MISSING",
            "PyTorch and MONAI are required to construct the frozen SegResNet runtime",
        ) from exc

    network = SegResNet(
        blocks_down=BUNDLE_NETWORK_BLOCKS_DOWN,
        blocks_up=BUNDLE_NETWORK_BLOCKS_UP,
        init_filters=BUNDLE_NETWORK_INIT_FILTERS,
        in_channels=4,
        out_channels=3,
        dropout_prob=BUNDLE_NETWORK_DROPOUT,
    )

    model_path = validation.bundle_dir / BUNDLE_MODEL_RELATIVE_PATH
    try:
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_CHECKPOINT_LOAD_FAILED",
            "frozen model.pt could not be read safely by PyTorch with weights_only=True",
        ) from exc

    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_uniform_prefix(state_dict, "module.")
    state_dict = _strip_uniform_prefix(state_dict, "model.")

    try:
        network.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_BUNDLE_STATE_DICT_MISMATCH",
            "frozen model.pt is not compatible with the frozen SegResNet architecture",
        ) from exc

    try:
        network = network.to(torch.device(device))
    except Exception as exc:
        raise SegmentationBundleRuntimeError(
            "SEGMENTATION_RUNTIME_DEVICE_INVALID",
            f"SegResNet could not be moved to requested device {device!r}",
        ) from exc
    network.eval()
    return network, validation
