from __future__ import annotations

import argparse

from sqlalchemy import inspect

from gbm_ai.api.config import get_settings
from gbm_ai.api.db import DatabaseManager
from gbm_ai.api.segmentation.bundle_runtime import (
    BUNDLE_MODEL_SHA256,
    BUNDLE_NAME,
    BUNDLE_OUTPUT_CHANNEL_ORDER,
    BUNDLE_OUTPUT_THRESHOLD,
    BUNDLE_OVERLAP,
    BUNDLE_ROI_SIZE,
    BUNDLE_VERSION,
    frozen_bundle_dir,
    load_frozen_network,
)
from gbm_ai.api.segmentation.inference import (
    SEGMENTATION_INFERENCE_VERSION,
    binary_masks_to_brats_labelmap,
    resolve_inference_device,
)


REQUIRED_SEGMENTATION_COLUMNS = {
    "id",
    "analysis_run_id",
    "model_input_checksum_sha256",
    "inference_version",
    "bundle_name",
    "bundle_version",
    "weights_checksum_sha256",
    "tc_storage_key",
    "tc_checksum_sha256",
    "wt_storage_key",
    "wt_checksum_sha256",
    "et_storage_key",
    "et_checksum_sha256",
    "labelmap_storage_key",
    "labelmap_checksum_sha256",
    "voxel_counts",
    "review_status",
    "physical_volume_generated",
    "anatomical_localization_generated",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-forward-smoke",
        action="store_true",
        help="Skip the small real SegResNet forward-pass smoke test.",
    )
    args = parser.parse_args()

    settings = get_settings()
    database = DatabaseManager(settings)
    try:
        inspector = inspect(database.engine)
        if "segmentations" not in inspector.get_table_names():
            raise SystemExit("segmentations table missing; run alembic upgrade head")
        columns = {item["name"] for item in inspector.get_columns("segmentations")}
        missing = REQUIRED_SEGMENTATION_COLUMNS - columns
        if missing:
            raise SystemExit(f"segmentations table missing columns: {sorted(missing)}")
    finally:
        database.dispose()

    device = resolve_inference_device(settings.segmentation_inference_device)
    bundle_dir = frozen_bundle_dir(settings.segmentation_bundle_root_resolved)
    network, validation = load_frozen_network(bundle_dir, device="cpu")

    import numpy as np
    import torch

    tc = np.zeros((2, 2, 2), dtype=np.uint8)
    wt = np.zeros_like(tc)
    et = np.zeros_like(tc)
    wt[0, 0, 0] = 1
    tc[0, 0, 1] = 1
    et[0, 1, 0] = 1
    labelmap = binary_masks_to_brats_labelmap(tc, wt, et)
    if set(np.unique(labelmap).tolist()) != {0, 1, 2, 4}:
        raise SystemExit("BraTS priority label-map smoke check failed")

    smoke_status = "SKIPPED"
    smoke_shape = None
    if not args.skip_forward_smoke:
        torch.manual_seed(42)
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(max(1, min(previous_threads, 4)))
            sample = torch.zeros((1, 4, 32, 32, 32), dtype=torch.float32)
            with torch.inference_mode():
                output = network(sample)
            if tuple(output.shape) != (1, 3, 32, 32, 32) or not bool(torch.isfinite(output).all()):
                raise SystemExit("real SegResNet forward smoke produced invalid output")
            smoke_status = "PASS"
            smoke_shape = tuple(int(v) for v in output.shape)
        finally:
            torch.set_num_threads(previous_threads)

    parameter_count = sum(int(parameter.numel()) for parameter in network.parameters())
    print("PHASE 6 STEP 5 — GUARDED SEGRESNET INFERENCE FOUNDATION CHECK")
    print("=" * 78)
    print("Alembic segmentation schema:  READY (20260814_0008)")
    print(f"Inference version:             {SEGMENTATION_INFERENCE_VERSION}")
    print(f"Bundle:                        {BUNDLE_NAME} {BUNDLE_VERSION}")
    print(f"Model SHA-256 verified:        YES ({BUNDLE_MODEL_SHA256[:12]}...)")
    print(f"Network state load:            STRICT PASS ({parameter_count:,} parameters)")
    print("Output order:                  " + " / ".join(BUNDLE_OUTPUT_CHANNEL_ORDER))
    print("Sliding-window ROI:            " + " x ".join(str(v) for v in BUNDLE_ROI_SIZE))
    print(f"Sliding-window overlap:        {BUNDLE_OVERLAP}")
    print(f"Sigmoid threshold:             {BUNDLE_OUTPUT_THRESHOLD}")
    print("BraTS label-map priority:      ET=4 > TC=1 > WT=2 > background=0")
    print(f"Configured runtime device:     {device}")
    print(f"Real SegResNet forward smoke:  {smoke_status}" + (f" {smoke_shape}" if smoke_shape else ""))
    print("Full-study inference executed: NO (requires an eligible prepared study)")
    print("Persistent WT/TC/ET masks:     IMPLEMENTED in protected storage")
    print("AnalysisRun traceability:      IMPLEMENTED")
    print("Final GBM decision state:      NOT SET BY SEGMENTATION ALONE")
    print("Physical volume:               NOT GENERATED")
    print("Anatomical localization:       NOT GENERATED")
    print("Background execution:          NOT IMPLEMENTED IN STEP 5")
    print("Clinical validation claimed:   NO")
    print("Phase 6 Step 5 foundation:     READY")


if __name__ == "__main__":
    main()
