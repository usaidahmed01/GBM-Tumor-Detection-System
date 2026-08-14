from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import uuid

from gbm_ai.api.main import create_app
from gbm_ai.api.models.analysis import SourceFormat
from gbm_ai.api.services.clinical_viewer import (
    CLINICAL_VIEWER_BACKEND_VERSION,
    CLINICAL_VIEWER_UI_VERSION,
    build_viewer_assets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"


def _study():
    channels = []
    for sequence in ("T1C", "T1", "T2", "FLAIR"):
        channels.append(
            {
                "sequence": sequence,
                "storage_key": f"studies/s/derived/model_geometry/{sequence.lower()}.nii.gz",
                "checksum_sha256": (sequence[0].lower() if sequence != "FLAIR" else "f") * 64,
                "size_bytes": 100 + len(sequence),
            }
        )
    return SimpleNamespace(
        id=uuid.uuid4(),
        source_format=SourceFormat.NIFTI,
        segmentation_preparation_summary={
            "model_geometry": {"status": "ready", "channels": channels}
        },
    )


def _segmentation():
    return SimpleNamespace(
        id=uuid.uuid4(),
        wt_storage_key="studies/s/derived/wt/wt.nii.gz",
        wt_checksum_sha256="a" * 64,
        wt_size_bytes=101,
        tc_storage_key="studies/s/derived/tc/tc.nii.gz",
        tc_checksum_sha256="b" * 64,
        tc_size_bytes=102,
        et_storage_key="studies/s/derived/et/et.nii.gz",
        et_checksum_sha256="c" * 64,
        et_size_bytes=103,
        labelmap_storage_key="studies/s/derived/label/label.nii.gz",
        labelmap_checksum_sha256="d" * 64,
        labelmap_size_bytes=104,
        review_status=SimpleNamespace(value="unreviewed"),
        clinician_modified=False,
    )


def test_phase8_step2_versions_and_loader_safe_route_are_registered():
    assert CLINICAL_VIEWER_BACKEND_VERSION == "phase8_step1_clinical_viewer_backend_v1"
    assert CLINICAL_VIEWER_UI_VERSION == "phase8_step2_cornerstone3d_readonly_ui_v1"
    paths = create_app().openapi()["paths"]
    assert "/api/v1/studies/{study_uuid}/viewer/manifest" in paths
    assert "/api/v1/studies/{study_uuid}/viewer/assets/{asset_alias}" in paths
    loader_path = "/api/v1/studies/{study_uuid}/viewer/assets/{asset_alias}/{filename}"
    assert loader_path in paths
    assert "get" in paths[loader_path]


def test_loader_url_preserves_safe_alias_boundary_and_ends_with_nifti_filename():
    study = _study()
    assets = build_viewer_assets(
        study,
        _segmentation(),
        localization=None,
        quantification=None,
    )
    payload = assets["mri_t1c"].public_payload(
        study_uuid=study.id,
        api_prefix="/api/v1",
    )
    assert payload["loader_url"].endswith(
        "/viewer/assets/mri_t1c/t1c_model_space.nii.gz"
    )
    assert "studies/s/derived" not in payload["loader_url"]
    assert "storage_key" not in payload


def test_frontend_package_uses_next_react_cornerstone_and_motion_with_webpack():
    package = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
    dependencies = package["dependencies"]
    assert "next" in dependencies
    assert "react" in dependencies
    assert "@cornerstonejs/core" in dependencies
    assert "@cornerstonejs/tools" in dependencies
    assert "@cornerstonejs/nifti-volume-loader" in dependencies
    assert "motion" in dependencies
    assert package["scripts"]["dev"].endswith("--webpack")
    assert package["scripts"]["build"].endswith("--webpack")
    next_config = (FRONTEND / "next.config.mjs").read_text(encoding="utf-8")
    assert "fs: false" in next_config
    assert "/gbm-api/:path*" in next_config


def test_clinical_viewer_ui_has_mpr_sequences_overlay_tools_and_medical_safety_copy():
    workspace = (FRONTEND / "components/viewer/ViewerWorkspace.jsx").read_text(encoding="utf-8")
    cornerstone = (FRONTEND / "components/viewer/CornerstoneMprViewer.jsx").read_text(encoding="utf-8")
    assets = (FRONTEND / "lib/viewerAssets.js").read_text(encoding="utf-8")
    all_source = "\n".join((workspace, cornerstone, assets))

    for term in ("T1C", "T1", "T2", "FLAIR"):
        assert term in all_source
    for term in ("AXIAL", "CORONAL", "SAGITTAL"):
        assert term in cornerstone
    for term in ("WT", "TC", "ET", "mask_labelmap"):
        assert term in all_source
    for term in ("WindowLevelTool", "PanTool", "ZoomTool", "StackScrollTool"):
        assert term in cornerstone
    assert "overlayOpacity" in all_source
    assert "AI-assisted imaging review only" in workspace
    assert "not a definitive GBM diagnosis" in workspace
    assert "clinician verification" in workspace.lower()


def test_step2_is_read_only_and_does_not_claim_manual_mask_correction():
    frontend_files = list(FRONTEND.rglob("*.jsx")) + list(FRONTEND.rglob("*.js"))
    source = "\n".join(path.read_text(encoding="utf-8") for path in frontend_files)
    forbidden_runtime_tools = (
        "BrushTool",
        "SphereScissorsTool",
        "RectangleScissorsTool",
        "CircleScissorsTool",
    )
    for name in forbidden_runtime_tools:
        assert name not in source
    assert "Manual brush/erase editing" in (FRONTEND / "README.md").read_text(encoding="utf-8")


def test_frontend_does_not_expose_internal_storage_paths_or_patient_identifiers():
    frontend_files = [
        path for path in FRONTEND.rglob("*")
        if path.is_file() and path.suffix in {".js", ".jsx", ".css", ".md", ".mjs", ".json"}
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in frontend_files)
    assert "/var/storage" not in source
    assert "storage_key" not in source
    assert "patient_name" not in source
    assert "patient_id" not in source


def test_motion_reduced_motion_and_responsive_medical_theme_are_present():
    workspace = (FRONTEND / "components/viewer/ViewerWorkspace.jsx").read_text(encoding="utf-8")
    launch = (FRONTEND / "app/page.jsx").read_text(encoding="utf-8")
    styles = (FRONTEND / "app/globals.css").read_text(encoding="utf-8")
    assert "MotionConfig" in workspace
    assert "useReducedMotion" in launch
    assert "prefers-reduced-motion" in styles
    assert "@media (max-width:" in styles
    assert "--cyan:" in styles
