from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gbm_ai.api.config import get_settings
from gbm_ai.api.localization import (
    ANATOMICAL_LOCALIZATION_VERSION,
    ATLAS_LICENSE,
    ATLAS_NAME,
    ATLAS_VERSION,
    REGISTRATION_METHOD,
    REGISTRATION_SUPPORT_DICE_MIN,
    STANDARD_SPACE,
    STANDARD_TEMPLATE_NAME,
    centroid_world_mm,
    compute_region_overlaps,
    hemisphere_from_standard_mask,
)
from gbm_ai.api.localization_assets import (
    load_and_verify_localization_assets,
    prepare_localization_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-assets", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    root: Path = settings.localization_atlas_root_resolved

    if args.download_assets:
        prepare_localization_assets(root)
    manifest, paths = load_and_verify_localization_assets(root)

    # Deterministic overlap/laterality smoke independent of patient data.
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[0:2, 2:4, 2:4] = 1
    affine = np.eye(4, dtype=np.float64)
    affine[0, 3] = -6.0
    atlas = np.zeros_like(mask, dtype=np.int16)
    atlas[0:2, 2:4, 2:4] = 1
    overlaps = compute_region_overlaps(mask, atlas, ["Background", "Synthetic Region"], atlas_name="synthetic")
    hemisphere = hemisphere_from_standard_mask(mask, affine)
    centroid = centroid_world_mm(mask, affine)
    assert overlaps and overlaps[0].overlap_voxels == 8
    assert hemisphere == "left"

    try:
        import nilearn
        nilearn_version = nilearn.__version__
    except Exception:
        nilearn_version = "unavailable"
    try:
        import templateflow
        templateflow_version = templateflow.__version__
    except Exception:
        templateflow_version = "unavailable"

    print("PHASE 7 STEP 2 — ATLAS-BASED ANATOMICAL LOCALIZATION CHECK")
    print("=" * 78)
    print(f"Localization version:          {ANATOMICAL_LOCALIZATION_VERSION}")
    print(f"Standard space:                {STANDARD_SPACE}")
    print(f"Fixed template:                {STANDARD_TEMPLATE_NAME}")
    print(f"Atlas:                         {ATLAS_NAME}")
    print(f"Frozen atlas variant:          {ATLAS_VERSION}")
    print(f"Atlas license:                 {ATLAS_LICENSE}")
    print(f"Nilearn installed:             {nilearn_version}")
    print(f"TemplateFlow installed:        {templateflow_version}")
    print(f"Asset manifest verified:       YES ({manifest['manifest_checksum_sha256'][:12]}...)")
    print(f"Template checksum verified:    YES ({manifest['files']['template']['sha256'][:12]}...)")
    print(f"Cortical atlas verified:       YES ({manifest['files']['cortical']['sha256'][:12]}...)")
    print(f"Subcortical atlas verified:    YES ({manifest['files']['subcortical']['sha256'][:12]}...)")
    print(f"Registration method:           {REGISTRATION_METHOD}")
    print(f"Registration QC threshold:     support Dice >= {REGISTRATION_SUPPORT_DICE_MIN:.2f} (engineering gate)")
    print("Left/right consistency gate:   REQUIRED")
    print("Primary region by overlap:     IMPLEMENTED")
    print("Secondary overlap regions:     IMPLEMENTED")
    print(f"Deterministic laterality smoke:{' ' * 5}PASS ({hemisphere}, centroid={centroid})")
    print("Clinician verification:        REQUIRED")
    print("Functional deficit prediction: NO")
    print("Segmentation equals diagnosis: NO")
    print("Clinical validation claimed:   NO")
    print("Next roadmap phase:            PHASE 8 — CLINICAL VIEWER")
    print("Phase 7 Step 2 foundation:     READY")


if __name__ == "__main__":
    main()
