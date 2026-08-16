from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
FIRST_PARTY = [FRONTEND / "app", FRONTEND / "components", FRONTEND / "lib"]


def source_text() -> str:
    files = []
    for root in FIRST_PARTY:
        if not root.exists():
            continue
        files.extend(root.rglob("*.jsx"))
        files.extend(root.rglob("*.js"))
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


def test_neuroglioma_brand_and_new_analysis_entrypoint_exist():
    home = (FRONTEND / "app" / "page.jsx").read_text(encoding="utf-8")
    assert "NeuroGlioma AI" in home
    assert 'href="/analysis/new"' in home
    assert "Study UUID" not in home
    assert "Open viewer" not in home


def test_user_flow_manages_uuid_internally_instead_of_requesting_it():
    source = source_text()
    assert "createUnifiedIntake" in source
    assert "Study UUID" not in (FRONTEND / "app/page.jsx").read_text(encoding="utf-8")
    assert "sessionStorage" in source
    assert "localStorage" in source
    assert "/viewer/current" in source
    assert "Study UUID" not in (FRONTEND / "app" / "page.jsx").read_text(encoding="utf-8")


def test_unified_ui_keeps_qc_mapping_and_capability_gates_before_segmentation():
    source = source_text()
    for token in (
        "runStudyQc",
        "confirmBrainScope",
        "confirmNiftiSequenceMapping",
        "routeStudyCapabilities",
        "runSegmentationPreflight",
        "prepareSegmentationVolumes",
        "prepareSegmentationGeometry",
        "prepareSegmentationModelInput",
        "enqueueSegmentationJob",
    ):
        assert token in source


def test_current_viewer_route_hides_technical_identifier_from_normal_navigation():
    current = (FRONTEND / "app" / "viewer" / "current" / "page.jsx").read_text(encoding="utf-8")
    viewer = (FRONTEND / "components" / "viewer" / "ViewerWorkspace.jsx").read_text(encoding="utf-8")
    assert "getActiveCase" in current
    assert "caseReference" in viewer
    assert "shortStudyId" not in viewer
