from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
MPR = FRONTEND / "components" / "viewer" / "CornerstoneMprViewer.jsx"
WORKSPACE = FRONTEND / "components" / "viewer" / "ViewerWorkspace.jsx"
CSS = FRONTEND / "app" / "globals.css"


def test_3d_volume_viewport_is_real_cornerstone_volume3d_not_a_mock_panel():
    source = MPR.read_text(encoding="utf-8")
    assert "ViewportType.VOLUME_3D" in source
    assert "addVolumesToViewports" in source
    assert "TrackballRotateTool" in source
    assert "THREE_D_VIEWPORT_ID" in source


def test_3d_review_supports_composite_and_mip_without_surface_mesh_claim():
    source = MPR.read_text(encoding="utf-8")
    assert "BlendModes.COMPOSITE" in source
    assert "BlendModes.MAXIMUM_INTENSITY_BLEND" in source
    assert "Composite" in source
    assert ">MIP<" in source
    assert "surface mesh" not in source.lower()


def test_3d_review_is_on_demand_and_keeps_mpr_correction_boundary():
    workspace = WORKSPACE.read_text(encoding="utf-8")
    source = MPR.read_text(encoding="utf-8")
    assert "threeDVisible" in workspace
    assert "3D review" in workspace
    assert "disabled={correctionOpen}" in workspace
    assert "REVIEW ONLY · NOT EDITABLE" in source
    for plane in ("AXIAL", "CORONAL", "SAGITTAL"):
        assert plane in source


def test_3d_labelmap_overlay_is_best_effort_and_mri_volume_remains_available():
    source = MPR.read_text(encoding="utf-8")
    assert "addLabelmapRepresentationToViewport" in source
    assert "3D segmentation overlay unavailable; MRI volume rendering remains enabled." in source
    assert "MRI volume only · 3D labelmap overlay unavailable" in source


def test_3d_review_medical_ui_styles_are_present():
    css = CSS.read_text(encoding="utf-8")
    for token in (
        ".three-d-toggle",
        ".mpr-grid--fourup",
        ".mpr-panel--3d",
        ".three-d-controls",
        ".three-d-overlay-state",
    ):
        assert token in css
