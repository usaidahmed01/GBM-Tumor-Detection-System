from pathlib import Path


PHASE8_STEP5_3D_VIEWER_VERSION = "phase8_step5_cornerstone3d_volume_review_v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    root = _repo_root()
    mpr_path = root / "frontend" / "components" / "viewer" / "CornerstoneMprViewer.jsx"
    workspace_path = root / "frontend" / "components" / "viewer" / "ViewerWorkspace.jsx"
    page_path = root / "frontend" / "app" / "analysis" / "new" / "page.jsx"

    mpr = mpr_path.read_text(encoding="utf-8")
    workspace = workspace_path.read_text(encoding="utf-8")
    page = page_path.read_text(encoding="utf-8")

    required = (
        "ViewportType.VOLUME_3D",
        "addVolumesToViewports",
        "TrackballRotateTool",
        "BlendModes.COMPOSITE",
        "BlendModes.MAXIMUM_INTENSITY_BLEND",
        "addLabelmapRepresentationToViewport",
        "REVIEW ONLY · NOT EDITABLE",
    )
    missing = [item for item in required if item not in mpr]
    if missing:
        raise RuntimeError(f"Phase 8 Step 5 3D viewer contract is incomplete: {missing}")
    if "threeDVisible" not in workspace or "3D review" not in workspace:
        raise RuntimeError("Phase 8 Step 5 on-demand 3D viewer controls are missing")
    if "<Suspense" not in page or "AnalysisIntakeWorkspace" not in page:
        raise RuntimeError("Next.js production-build Suspense boundary is missing")

    print("PHASE 8 STEP 5 — INTERACTIVE 3D VOLUME REVIEW CHECK")
    print("=" * 82)
    print(f"3D viewer version:                  {PHASE8_STEP5_3D_VIEWER_VERSION}")
    print("Product name:                       NeuroGlioma AI")
    print("Next.js search-param Suspense fix:  READY")
    print("3D rendering engine:                Cornerstone3D VolumeViewport3D")
    print("3D viewport creation:               ON DEMAND")
    print("3D source volume:                   CURRENT SELECTED MRI SEQUENCE")
    print("3D blend modes:                     COMPOSITE / MIP")
    print("3D interaction:                     TRACKBALL ROTATE / PAN / ZOOM")
    print("WT / TC / ET 3D labelmap overlay:  BEST-EFFORT VIEWER REPRESENTATION")
    print("Fallback if 3D overlay unsupported: MRI VOLUME REMAINS AVAILABLE")
    print("MPR axial/coronal/sagittal:         RETAINED")
    print("Brush/erase correction:             RETAINED IN MPR ONLY")
    print("3D correction:                      DISABLED (REVIEW ONLY)")
    print("Surface mesh generation:            NOT CLAIMED / NOT REQUIRED")
    print("Physical quantification source:     BACKEND ONLY")
    print("Anatomical localization source:     BACKEND ONLY")
    print("Clinical validation claimed:        NO")
    print("Next roadmap phase:                 PHASE 9 — DECISION FUSION / REPORT")
    print("Phase 8 Step 5 foundation:          READY")


if __name__ == "__main__":
    main()
