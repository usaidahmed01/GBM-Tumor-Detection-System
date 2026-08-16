from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from typing import Any


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def collect_preflight() -> dict[str, Any]:
    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64"}:
        raise RuntimeError(
            f"ORACLE_ARM64_PLATFORM_MISMATCH: expected linux/arm64, got {machine!r}"
        )

    import numpy as np
    import scipy
    import sklearn
    import torch
    import torchvision
    import monai
    import SimpleITK as sitk
    import nibabel as nib
    import nilearn
    import pydicom
    import psycopg
    import sqlalchemy
    import fastapi
    import huggingface_hub
    import templateflow
    from PIL import Image
    from monai.networks.nets import SegResNet
    from torchvision.models import efficientnet_v2_s

    from gbm_ai.api.segmentation.bundle_runtime import (
        BUNDLE_NETWORK_BLOCKS_DOWN,
        BUNDLE_NETWORK_BLOCKS_UP,
        BUNDLE_NETWORK_DROPOUT,
        BUNDLE_NETWORK_INIT_FILTERS,
    )

    # Oracle Ampere A1 is a CPU-only deployment target. A generic PyPI
    # ARM64 torch wheel may pull CUDA/NVIDIA runtime packages. The deployment
    # image must instead use PyTorch's official +cpu wheels.
    installed_distribution_names = {
        str(dist.metadata.get("Name", "")).strip().lower()
        for dist in metadata.distributions()
    }
    forbidden_gpu_packages = sorted(
        name
        for name in installed_distribution_names
        if name.startswith("nvidia-") or name in {"cuda-toolkit", "cuda-bindings"}
    )
    if forbidden_gpu_packages:
        raise RuntimeError(
            "ORACLE_ARM64_GPU_PACKAGES_PRESENT: " + ", ".join(forbidden_gpu_packages)
        )
    if "+cpu" not in _version(torch):
        raise RuntimeError(
            f"ORACLE_ARM64_NON_CPU_TORCH: expected official +cpu wheel, got {_version(torch)!r}"
        )
    if getattr(torch.version, "cuda", None) is not None or torch.cuda.is_available():
        raise RuntimeError("ORACLE_ARM64_CUDA_RUNTIME_PRESENT: CPU-only image required")

    # Exercise native scientific-imaging bindings without touching the network
    # or requiring patient/model assets.
    itk_image = sitk.Image([8, 8, 8], sitk.sitkFloat32)
    if itk_image.GetSize() != (8, 8, 8):
        raise RuntimeError("SimpleITK basic ARM64 runtime check failed")

    nifti = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    if nifti.shape != (2, 2, 2):
        raise RuntimeError("NiBabel basic ARM64 runtime check failed")

    pil_image = Image.new("RGB", (8, 8))
    if pil_image.size != (8, 8):
        raise RuntimeError("Pillow basic ARM64 runtime check failed")

    classifier = efficientnet_v2_s(weights=None)
    classifier_parameters = sum(int(parameter.numel()) for parameter in classifier.parameters())
    del classifier

    segmenter = SegResNet(
        blocks_down=BUNDLE_NETWORK_BLOCKS_DOWN,
        blocks_up=BUNDLE_NETWORK_BLOCKS_UP,
        init_filters=BUNDLE_NETWORK_INIT_FILTERS,
        in_channels=4,
        out_channels=3,
        dropout_prob=BUNDLE_NETWORK_DROPOUT,
    )
    segmentation_parameters = sum(int(parameter.numel()) for parameter in segmenter.parameters())
    del segmenter

    # Import the actual FastAPI application module to catch deployment-only
    # import/linker failures. This does not connect to the database or run AI.
    from gbm_ai.api.main import app

    if app is None:
        raise RuntimeError("FastAPI application import failed")

    return {
        "preflight_version": "phase10_step5a_oracle_arm64_preflight_v1",
        "platform": {
            "system": platform.system(),
            "machine": machine,
            "python": platform.python_version(),
        },
        "runtime": {
            "torch": _version(torch),
            "torchvision": _version(torchvision),
            "monai": _version(monai),
            "numpy": _version(np),
            "scipy": _version(scipy),
            "scikit_learn": _version(sklearn),
            "simpleitk": str(sitk.Version_VersionString()),
            "nibabel": _version(nib),
            "nilearn": _version(nilearn),
            "pydicom": _version(pydicom),
            "psycopg": _version(psycopg),
            "sqlalchemy": _version(sqlalchemy),
            "fastapi": _version(fastapi),
            "huggingface_hub": _version(huggingface_hub),
            "templateflow": _version(templateflow),
        },
        "cpu": {
            "torch_cpu_wheel": "+cpu" in _version(torch),
            "torch_compiled_cuda": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "forbidden_gpu_packages": forbidden_gpu_packages,
            "deployment_device": "cpu",
        },
        "model_architecture_smoke": {
            "efficientnetv2_s_constructed": True,
            "efficientnetv2_s_parameters": classifier_parameters,
            "segresnet_constructed": True,
            "segresnet_parameters": segmentation_parameters,
            "checkpoint_weights_loaded": False,
            "full_forward_inference_executed": False,
        },
        "asset_policy": {
            "patient_data_in_image": False,
            "classifier_checkpoints_in_image": False,
            "monai_bundle_in_image": False,
            "localization_atlas_in_image": False,
        },
        "clinical_validation_claimed": False,
        "status": "ARM64_RUNTIME_COMPATIBLE",
    }


def main() -> None:
    payload = collect_preflight()
    print("PHASE 10 STEP 5A — ORACLE AMPERE A1 / ARM64 DOCKER PREFLIGHT")
    print("=" * 82)
    print(f"Architecture:                    {payload['platform']['machine']}")
    print(f"Python:                          {payload['platform']['python']}")
    print(f"PyTorch:                         {payload['runtime']['torch']}")
    print(f"TorchVision:                     {payload['runtime']['torchvision']}")
    print(f"MONAI:                           {payload['runtime']['monai']}")
    print(f"SimpleITK:                       {payload['runtime']['simpleitk']}")
    print("EfficientNetV2-S construction:   PASS")
    print("SegResNet construction:          PASS")
    print("FastAPI application import:      PASS")
    print("PyTorch CPU wheel:                PASS")
    print("CUDA/NVIDIA runtime packages:     NONE")
    print("Deployment inference device:      CPU")
    print("Checkpoint weights loaded:       NO — RUNTIME ASSETS REMAIN PRIVATE")
    print("Full MRI inference executed:     NO — PREFLIGHT ONLY")
    print("Clinical validation claimed:     NO")
    print("ARM64 Docker compatibility:      PASS")
    print("Next step:                       CREATE ORACLE ALWAYS FREE A1 INSTANCE")
    print("\nJSON_RESULT=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ARM64 Docker preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
