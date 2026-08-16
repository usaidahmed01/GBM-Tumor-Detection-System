from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding='utf-8')


def test_cornerstone_style_api_supports_nested_v2plus_namespace():
    src = read('frontend/components/viewer/CornerstoneMprViewer.jsx')
    assert "segmentation?.config?.style?.setStyle" in src
    assert "segmentation.config.style.setStyle(specifier, style)" in src
    assert "setLabelmapStyle(" in src


def test_overlay_style_no_longer_directly_calls_missing_top_level_api():
    src = read('frontend/components/viewer/CornerstoneMprViewer.jsx')
    assert "tools.segmentation.setStyle(" not in src
    assert "overlay rendering will use Cornerstone defaults" in src


def test_cornerstone_color_api_supports_current_nested_config_namespace():
    src = read('frontend/components/viewer/CornerstoneMprViewer.jsx')
    assert "segmentation?.config?.color" in src
    assert "setSegmentIndexColor" in src
    assert "configureLabelmapColors(" in src


def test_viewer_loading_has_multiple_real_stages_not_fake_percentage():
    src = read('frontend/components/viewer/CornerstoneMprViewer.jsx')
    assert "Loading ${sequence} imaging data" in src
    assert "Preparing multiplanar MRI views" in src
    assert "Applying AI segmentation overlay" in src
    assert "aria-busy={state.status === 'loading'}" in src


def test_clinician_viewer_has_keyboard_shortcuts_and_accessible_pressed_states():
    cs = read('frontend/components/viewer/CornerstoneMprViewer.jsx')
    workspace = read('frontend/components/viewer/ViewerWorkspace.jsx')
    assert "TOOL_SHORTCUTS" in cs
    assert "aria-keyshortcuts={TOOL_SHORTCUTS[key]}" in cs
    assert "aria-pressed={activeTool === key" in cs
    assert "SEQUENCE_SHORTCUTS" in workspace
    assert 'aria-keyshortcuts="O"' in workspace
    assert 'aria-keyshortcuts="V"' in workspace


def test_viewer_controls_have_focus_and_active_feedback():
    css = read('frontend/app/globals.css')
    assert ".tool-button:focus-visible" in css
    assert ".viewer-interaction-status__active" in css
    assert "viewer-loading-scan" in css


def test_next_warning_filter_covers_cornerstone_compute_runtime_variant():
    config = read('frontend/next.config.mjs')
    assert "webpack(?:-runtime)?" in config
    assert "typeof warning === 'string'" in config


def test_sidebar_allows_safe_retry_of_missing_derived_outputs():
    workspace = read('frontend/components/viewer/ViewerWorkspace.jsx')
    css = read('frontend/app/globals.css')
    assert "retryDerivedOutput" in workspace
    assert "Retry localization" in workspace
    assert "Recalculate" in workspace
    assert ".sidebar-inline-action" in css
