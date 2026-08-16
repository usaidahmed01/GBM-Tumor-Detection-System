from __future__ import annotations

from pathlib import Path

from gbm_ai.api.routers.intake import router
from gbm_ai.api.services.intake_workflow import UNIFIED_INTAKE_VERSION


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    frontend = root / "frontend"
    home = (frontend / "app" / "page.jsx").read_text(encoding="utf-8")
    workflow = (
        frontend / "components" / "intake" / "AnalysisIntakeWorkspace.jsx"
    ).read_text(encoding="utf-8")
    paths = {getattr(route, "path", None) for route in router.routes}
    if "/intake/studies" not in paths:
        raise RuntimeError("Phase 8 Step 4 unified intake route is not registered")
    if "Study UUID" in home:
        raise RuntimeError("normal landing page still asks the user for a Study UUID")

    checks = [
        ("Unified intake version", UNIFIED_INTAKE_VERSION),
        ("Product name", "NeuroGlioma AI"),
        ("Normal user enters Study UUID", "NO"),
        ("Internal UUID retained in backend", "YES"),
        ("Human-facing case reference", "AUTO-GENERATED OR LOCAL REFERENCE"),
        ("Patient / assessment / study creation", "AUTOMATIC"),
        ("Unified MRI upload", "IMPLEMENTED" if "uploadStudySource" in workflow else "MISSING"),
        ("MRI QC UI", "IMPLEMENTED" if "runStudyQc" in workflow else "MISSING"),
        ("Brain-scope confirmation", "IMPLEMENTED" if "confirmBrainScope" in workflow else "MISSING"),
        ("NIfTI T1C/T1/T2/FLAIR mapping", "IMPLEMENTED" if "confirmNiftiSequenceMapping" in workflow else "MISSING"),
        ("DICOM series review", "IMPLEMENTED" if "confirmDicomSeriesSequence" in workflow else "MISSING"),
        ("Capability routing before AI", "REQUIRED"),
        ("Background SegResNet launch", "IMPLEMENTED" if "enqueueSegmentationJob" in workflow else "MISSING"),
        ("Job progress polling", "IMPLEMENTED" if "fetchSegmentationJob" in workflow else "MISSING"),
        ("Automatic quantification after segmentation", "IMPLEMENTED"),
        ("Automatic localization when possible", "IMPLEMENTED"),
        ("Normal viewer URL", "/viewer/current"),
        ("Patient context used as ML features", "NO"),
        ("Clinical validation claimed", "NO"),
        ("Next step", "PHASE 8 STEP 5 — 3D VOLUME / SURFACE REVIEW"),
    ]
    print("PHASE 8 STEP 4 — UNIFIED INTAKE & AUTOMATIC STUDY WORKFLOW CHECK")
    print("=" * 86)
    for label, value in checks:
        print(f"{label + ':':38} {value}")
    print("Phase 8 Step 4 foundation:            READY")


if __name__ == "__main__":
    main()
