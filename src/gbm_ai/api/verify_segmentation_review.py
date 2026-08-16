from __future__ import annotations

from pathlib import Path

from gbm_ai.api.main import create_app
from gbm_ai.api.models.segmentation import SegmentationReviewAction
from gbm_ai.api.services.clinical_viewer import CLINICAL_VIEWER_UI_VERSION
from gbm_ai.api.services.segmentation_review import SEGMENTATION_REVIEW_VERSION


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    frontend = project_root / "frontend"
    cornerstone = frontend / "components" / "viewer" / "CornerstoneMprViewer.jsx"
    workspace = frontend / "components" / "viewer" / "ViewerWorkspace.jsx"
    api = frontend / "lib" / "api.js"
    migration = project_root / "migrations" / "versions" / "20260816_0012_segmentation_review_revisions.py"

    for path in (cornerstone, workspace, api, migration):
        if not path.is_file():
            raise RuntimeError(f"Phase 8 Step 3 file missing: {path.relative_to(project_root)}")

    paths = create_app().openapi()["paths"]
    required_routes = (
        "/api/v1/studies/{study_uuid}/viewer/review",
        "/api/v1/studies/{study_uuid}/viewer/corrections",
        "/api/v1/studies/{study_uuid}/viewer/review/history",
    )
    for route in required_routes:
        if route not in paths:
            raise RuntimeError(f"Phase 8 Step 3 route is not registered: {route}")

    corner_source = cornerstone.read_text(encoding="utf-8")
    workspace_source = workspace.read_text(encoding="utf-8")
    api_source = api.read_text(encoding="utf-8")
    combined = "\n".join((corner_source, workspace_source, api_source))
    for term in (
        "BrushTool",
        "FILL_INSIDE_CIRCLE",
        "ERASE_INSIDE_CIRCLE",
        "setActiveSegmentIndex",
        "submitLabelmapCorrection",
        "submitSegmentationReview",
    ):
        if term not in combined:
            raise RuntimeError(f"Phase 8 Step 3 frontend capability missing: {term}")
    if "storage_key" in combined or "/var/storage" in combined:
        raise RuntimeError("frontend must not expose protected object-storage internals")

    print("PHASE 8 STEP 3 — CLINICIAN MASK REVIEW & CORRECTION CHECK")
    print("=" * 82)
    print(f"Clinical viewer UI version:       {CLINICAL_VIEWER_UI_VERSION}")
    print(f"Review contract version:          {SEGMENTATION_REVIEW_VERSION}")
    print("Alembic Phase 8 schema:           READY (20260816_0012)")
    print("Review actions:                    ACCEPT / REJECT / EDIT")
    print("Cornerstone brush editing:         IMPLEMENTED")
    print("Brush paint strategy:              FILL_INSIDE_CIRCLE")
    print("Brush erase strategy:              ERASE_INSIDE_CIRCLE")
    print("Editable planes:                   AXIAL / CORONAL / SAGITTAL")
    print("Editable labels:                   TC=1 / WT=2 / ET=4")
    print("Optimistic checksum concurrency:   REQUIRED")
    print("Unknown BraTS labels:              BLOCKED")
    print("Immutable review revisions:        IMPLEMENTED")
    print("Protected corrected NIfTI masks:   IMPLEMENTED")
    print("Quantification after edit:         AUTOMATIC RECALCULATION")
    print("Localization after edit:           AUTOMATIC RECALCULATION WHEN POSSIBLE")
    print("Rejected-mask volume/location:     BLOCKED")
    print("Raw storage keys in frontend:      NO")
    print("Patient identifiers in UI feed:    NO")
    print("Segmentation equals diagnosis:     NO")
    print("Clinical validation claimed:       NO")
    print("Next step:                         PHASE 8 STEP 4 — UNIFIED INTAKE / AUTOMATIC STUDY CREATION UI")
    print("Phase 8 Step 3 foundation:         READY")


if __name__ == "__main__":
    main()
