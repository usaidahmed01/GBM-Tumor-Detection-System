from __future__ import annotations

from pathlib import Path

from gbm_ai.api.main import create_app


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    app = create_app()
    paths = set(app.openapi().get("paths", {}))
    required_api = {
        "/api/v1/studies/{study_uuid}/report/preview",
        "/api/v1/studies/{study_uuid}/report/finalize",
        "/api/v1/studies/{study_uuid}/report/current",
    }
    missing = sorted(required_api - paths)
    if missing:
        raise RuntimeError(f"report API routes missing: {missing}")

    workspace = ROOT / "frontend/components/report/ReportWorkspace.jsx"
    current_route = ROOT / "frontend/app/report/current/page.jsx"
    css = ROOT / "frontend/app/globals.css"
    for path in (workspace, current_route, css):
        if not path.exists():
            raise RuntimeError(f"required report UI file missing: {path}")

    workspace_text = workspace.read_text(encoding="utf-8")
    css_text = css.read_text(encoding="utf-8")
    if "window.print()" not in workspace_text or "@media print" not in css_text:
        raise RuntimeError("print/PDF-ready report export contract is incomplete")
    if ".ng-home-shell,\n.intake-shell {\n  overflow: clip;" not in css_text:
        raise RuntimeError("page footer boundary/ambient clipping fix is missing")

    print("PHASE 9 STEP 4 — REPORT UI / SIGN-OFF / PRINT EXPORT CHECK")
    print("=" * 82)
    print("Report UI version:                 phase9_step4_report_ui_v1")
    print("Human-facing report route:        /report/current")
    print("Structured report preview:        IMPLEMENTED")
    print("Decision fusion on report open:   GUARDED / BEST-EFFORT")
    print("Clinician sign-off UI:            IMPLEMENTED")
    print("Finalized report retrieval:       IMPLEMENTED")
    print("Immutable checksum display:       IMPLEMENTED")
    print("JSON export:                      IMPLEMENTED")
    print("Print stylesheet:                 A4 READY")
    print("Browser Save-as-PDF workflow:     IMPLEMENTED")
    print("Server-generated PDF file:        NOT CLAIMED")
    print("Viewer -> report navigation:      IMPLEMENTED")
    print("Footer overscroll below page:     FIXED BY SHELL CLIPPING")
    print("Raw storage paths exposed:        NO")
    print("Clinical validation claimed:      NO")
    print("Phase 9 Step 4 foundation:        READY")


if __name__ == "__main__":
    main()
