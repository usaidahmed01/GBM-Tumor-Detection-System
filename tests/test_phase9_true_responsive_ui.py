from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "frontend" / "app" / "globals.css"


def test_responsive_baseline_is_not_hidden_behind_large_desktop_media_query():
    css = CSS.read_text(encoding="utf-8")
    marker = "Phase 9 Step 3 Hotfix — true viewport-responsive production sizing"
    assert marker in css
    section = css.split(marker, 1)[1]
    assert "width: min(92vw, 1500px)" in section
    assert "font-size: clamp(12px, .8vw, 14px)" in section
    assert "grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr)" in section


def test_footer_uses_viewport_fill_layout():
    css = CSS.read_text(encoding="utf-8")
    section = css.split("Phase 9 Step 3 Hotfix — true viewport-responsive production sizing", 1)[1]
    assert "min-height: 100dvh" in section
    assert "margin-top: auto" in section


def test_mobile_and_mid_size_breakpoints_are_present():
    css = CSS.read_text(encoding="utf-8")
    section = css.split("Phase 9 Step 3 Hotfix — true viewport-responsive production sizing", 1)[1]
    assert "@media (max-width: 1180px)" in section
    assert "@media (max-width: 900px)" in section
    assert "@media (max-width: 640px)" in section
