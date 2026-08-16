from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_homepage_removes_internal_or_prototype_copy():
    content = (REPO_ROOT / "frontend/app/page.jsx").read_text(encoding="utf-8")
    assert "Research prototype" not in content
    assert "No UUID entry required" not in content
    assert "Internal study IDs are created automatically" not in content


def test_homepage_css_contains_large_screen_and_mobile_breakpoints():
    css = (REPO_ROOT / "frontend/app/globals.css").read_text(encoding="utf-8")
    assert "max-width:1680px" in css
    assert "@media(max-width:1280px)" in css
    assert ".ng-nav-button{display:grid" in css


def test_intake_uses_clinical_copy_and_large_screen_density_uplift():
    intake = (REPO_ROOT / "frontend/components/intake/AnalysisIntakeWorkspace.jsx").read_text(encoding="utf-8")
    css = (REPO_ROOT / "frontend/app/globals.css").read_text(encoding="utf-8")
    assert "No study UUID is required" not in intake
    assert "technical identifiers automatically" not in intake
    assert "Phase 9 Step 3 — responsive production density uplift" in css
    assert ".intake-stage-wrap { max-width: 1380px" in css
    assert ".intake-stage-heading h1 { font-size: clamp(30px" in css
    assert ".workspace-main { grid-template-columns: minmax(0, 1fr) 370px" in css
