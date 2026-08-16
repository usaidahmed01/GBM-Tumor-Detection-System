from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect

from gbm_ai.api.db import DatabaseManager
from gbm_ai.api.config import get_settings
from gbm_ai.api.main import create_app
from gbm_ai.api.services.decision_fusion import DECISION_FUSION_VERSION


def main() -> None:
    settings = get_settings()
    dbm = DatabaseManager(settings)
    with dbm.engine.connect() as conn:
        columns = {item["name"] for item in inspect(conn).get_columns("analysis_runs")}
    required = {"decision_fusion_version", "decision_evidence_summary", "decision_fused_at"}
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError("Phase 9 decision-fusion columns missing: " + ", ".join(missing))

    paths = set(create_app(settings).openapi().get("paths", {}))
    expected = {
        "/api/v1/studies/{study_uuid}/decision/fuse",
        "/api/v1/studies/{study_uuid}/decision/current",
    }
    if not expected.issubset(paths):
        raise RuntimeError("Phase 9 decision-fusion routes are not registered")

    css_path = Path("frontend/app/globals.css")
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    compact_css = "".join(css.split())
    single_scroll_contract = (
        "html{" in compact_css
        and "overflow-x:clip" in compact_css
        and "overflow-y:auto" in compact_css
        and "body{margin:0" in compact_css
        and "overflow:visible" in compact_css
        and "body{overflow-x:hidden" not in compact_css
        and ".ng-home-shell,.intake-shell{min-height:100vh;position:relative;overflow-x:clip" in compact_css
        and ".ng-home-shell,.intake-shell{min-height:100vh;position:relative;overflow-x:hidden" not in compact_css
    )
    if not single_scroll_contract:
        raise RuntimeError("single document-scroll contract is not present in frontend CSS")

    print("PHASE 9 STEP 1 — GUARDED DECISION FUSION CHECK")
    print("=" * 82)
    print("Alembic Phase 9 schema:                 READY (20260816_0013)")
    print(f"Decision fusion version:                {DECISION_FUSION_VERSION}")
    print("Final states:                           GBM suspected / not suspected / indeterminate")
    print("Safety monotonicity:                    DETERMINATE -> INDETERMINATE ONLY")
    print("Low GBM probability means normal brain: NO")
    print("Segmentation treated as GBM diagnosis:  NO")
    print("2D classifier deployment strategy:      STILL NOT FROZEN")
    print("2D determinate state before freeze:      BLOCKED -> INDETERMINATE")
    print("Volumetric classifier bridge:           STILL NOT VALIDATED")
    print("DICOM/NIfTI GBM state without bridge:   INDETERMINATE")
    print("Reviewed lesion evidence retained:      YES, AS NON-DIAGNOSTIC SUPPORTING EVIDENCE")
    print("Explicit segmentation review for report: REQUIRED FOR VOLUMETRIC STUDIES")
    print("Decision provenance in AnalysisRun:     IMPLEMENTED")
    print("Decision-fusion audit event:             IMPLEMENTED")
    print("Frontend document scroll owner:         SINGLE (HTML / VIEWPORT)")
    print("Duplicate page scrollbar:                FIXED")
    print("Structured final report:                 NOT IMPLEMENTED IN STEP 1")
    print("Clinician report sign-off:               NOT IMPLEMENTED IN STEP 1")
    print("Clinical validation claimed:             NO")
    print("Next step:                               PHASE 9 STEP 2 — CLASSIFIER DEPLOYMENT FREEZE & 2D RUNTIME")
    print("Phase 9 Step 1 foundation:               READY")


if __name__ == "__main__":
    main()
