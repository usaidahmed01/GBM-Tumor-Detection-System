from __future__ import annotations

import argparse

from gbm_ai.api.config import get_settings
from gbm_ai.api.segmentation.bundle_runtime import (
    BUNDLE_INPUT_CHANNEL_ORDER,
    BUNDLE_MODEL_SHA256,
    BUNDLE_NAME,
    BUNDLE_OUTPUT_CHANNEL_ORDER,
    BUNDLE_OVERLAP,
    BUNDLE_ROI_SIZE,
    BUNDLE_VERSION,
    SegmentationBundleRuntimeError,
    download_frozen_bundle,
    frozen_bundle_dir,
    load_frozen_network,
    validate_bundle_layout,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the frozen MONAI bundle from the official Hugging Face repository if needed.",
    )
    args = parser.parse_args()

    settings = get_settings()
    bundle_dir = frozen_bundle_dir(settings.segmentation_bundle_root_resolved)

    if args.download:
        validation = download_frozen_bundle(settings.segmentation_bundle_root_resolved)
    else:
        validation = validate_bundle_layout(bundle_dir)

    network, validation = load_frozen_network(validation.bundle_dir, device="cpu")
    parameter_count = sum(int(parameter.numel()) for parameter in network.parameters())

    import monai
    import torch

    print("PHASE 6 STEP 4 — MONAI BUNDLE & MODEL-INPUT RUNTIME CHECK")
    print("=" * 76)
    print(f"MONAI installed:              {monai.__version__}")
    print(f"PyTorch installed:            {torch.__version__}")
    print(f"Bundle:                       {BUNDLE_NAME}")
    print(f"Frozen bundle version:        {BUNDLE_VERSION}")
    print(f"Bundle metadata MONAI:        {validation.metadata_monai_version}")
    print(f"Bundle metadata PyTorch:      {validation.metadata_pytorch_version}")
    print(f"Model SHA-256 verified:       YES ({BUNDLE_MODEL_SHA256[:12]}...)")
    print("Input order:                  " + " -> ".join(BUNDLE_INPUT_CHANNEL_ORDER))
    print("Output order:                 " + " / ".join(BUNDLE_OUTPUT_CHANNEL_ORDER))
    print("Normalization:                nonzero=True, channel_wise=True")
    print("Sliding-window ROI:           " + " x ".join(str(v) for v in BUNDLE_ROI_SIZE))
    print(f"Sliding-window overlap:       {BUNDLE_OVERLAP}")
    print(f"Network state load:           STRICT PASS ({parameter_count:,} parameters)")
    print("Runtime verification device:  CPU")
    print("Forward inference executed:   NO")
    print("WT/TC/ET masks generated:     NO")
    print("Physical volume:              NOT GENERATED")
    print("Anatomical localization:      NOT GENERATED")
    print("Clinical validation claimed:  NO")
    print("Phase 6 Step 4 runtime:       READY")


if __name__ == "__main__":
    try:
        main()
    except SegmentationBundleRuntimeError as exc:
        print(f"ERROR [{exc.code}]: {exc}")
        raise SystemExit(1)
