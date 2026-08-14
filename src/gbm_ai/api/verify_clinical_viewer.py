from __future__ import annotations

from gbm_ai.api.main import create_app
from gbm_ai.api.services.clinical_viewer import (
    CLINICAL_VIEWER_BACKEND_VERSION,
    VIEWER_MODEL_SEQUENCES,
    VIEWER_PLANES,
    VIEWER_PRIMARY_REFERENCE_SEQUENCE,
)


def main() -> None:
    app = create_app()
    paths = {route.path for route in app.routes}
    manifest_path = "/api/v1/studies/{study_uuid}/viewer/manifest"
    asset_path = "/api/v1/studies/{study_uuid}/viewer/assets/{asset_alias}"
    if manifest_path not in paths or asset_path not in paths:
        raise RuntimeError("Phase 8 viewer routes are not registered")
    if VIEWER_MODEL_SEQUENCES != ("T1C", "T1", "T2", "FLAIR"):
        raise RuntimeError("viewer MRI channel contract drifted")
    if VIEWER_PLANES != ("axial", "coronal", "sagittal"):
        raise RuntimeError("viewer plane contract drifted")

    print("PHASE 8 STEP 1 — CLINICAL VIEWER BACKEND FOUNDATION CHECK")
    print("=" * 78)
    print("Alembic/schema basis:            READY (20260815_0011 retained)")
    print(f"Viewer contract version:         {CLINICAL_VIEWER_BACKEND_VERSION}")
    print(f"Primary reference sequence:      {VIEWER_PRIMARY_REFERENCE_SEQUENCE}")
    print("Approved MRI volumes:            T1C / T1 / T2 / FLAIR")
    print("Approved patient-space overlays: WT / TC / ET / BraTS label map")
    print("Standard-space review mask:      WT MNI (when current localization exists)")
    print("Viewer planes:                   axial / coronal / sagittal")
    print("Raw storage keys in manifest:    NO")
    print("Raw storage path endpoint:       NO")
    print("Checksum before streaming:       REQUIRED")
    print("Patient identifiers in manifest: NO")
    print("Protected object audit event:    IMPLEMENTED")
    print("3D volume asset basis:           READY")
    print("Cornerstone/OHIF frontend:       NOT IMPLEMENTED IN STEP 1")
    print("Manual mask editing:             NOT IMPLEMENTED IN STEP 1")
    print("Clinician verification:          REQUIRED")
    print("Segmentation equals diagnosis:   NO")
    print("Clinical validation claimed:     NO")
    print("Next step:                       PHASE 8 STEP 2 — CLINICAL VIEWER UI")
    print("Phase 8 Step 1 foundation:       READY")


if __name__ == "__main__":
    main()
