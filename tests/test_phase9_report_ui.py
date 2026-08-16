from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_report_ui_routes_and_workspace_exist():
    assert (ROOT / "frontend/app/report/current/page.jsx").exists()
    assert (ROOT / "frontend/app/report/[studyUuid]/page.jsx").exists()
    workspace = (ROOT / "frontend/components/report/ReportWorkspace.jsx").read_text(encoding="utf-8")
    assert "phase9_step4_report_ui_v1" not in workspace
    assert "Case {safeCaseReference}" in workspace
    assert "Print / Save PDF" in workspace
    assert "window.print()" in workspace
    assert "Finalize report" in workspace


def test_report_frontend_api_contract_is_wired():
    source = (ROOT / "frontend/lib/api.js").read_text(encoding="utf-8")
    assert "/decision/fuse" in source
    assert "/report/preview" in source
    assert "/report/finalize" in source
    assert "/report/current" in source


def test_report_ui_does_not_expose_storage_paths():
    source = "\n".join(
        p.read_text(encoding="utf-8")
        for p in [
            ROOT / "frontend/components/report/ReportWorkspace.jsx",
            ROOT / "frontend/app/report/current/page.jsx",
            ROOT / "frontend/app/report/[studyUuid]/page.jsx",
        ]
    )
    assert "/var/storage" not in source
    assert "storage_key" not in source


def test_print_styles_and_footer_boundary_are_present():
    css = (ROOT / "frontend/app/globals.css").read_text(encoding="utf-8")
    assert "@media print" in css
    assert "@page{size:A4" in css
    assert ".ng-home-shell,\n.intake-shell {\n  overflow: clip;" in css
    assert ".report-layout" in css
    assert ".report-paper" in css


def test_viewer_has_report_navigation():
    source = (ROOT / "frontend/components/viewer/ViewerWorkspace.jsx").read_text(encoding="utf-8")
    assert 'href="/report/current"' in source
    assert "viewer-report-button" in source
