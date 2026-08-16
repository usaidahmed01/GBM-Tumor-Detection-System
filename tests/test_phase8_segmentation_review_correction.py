from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gbm_ai.api.main import create_app
from gbm_ai.api.models.segmentation import (
    SegmentationReviewAction,
    SegmentationReviewStatus,
)
from gbm_ai.api.services.clinical_viewer import CLINICAL_VIEWER_UI_VERSION
from gbm_ai.api.services.segmentation_review import (
    SEGMENTATION_REVIEW_VERSION,
    SegmentationReviewServiceError,
    brats_binary_masks_from_labelmap,
    decode_clinician_labelmap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"


def test_phase8_step3_contract_and_review_routes_are_registered():
    assert CLINICAL_VIEWER_UI_VERSION == "phase8_step3_clinician_mask_review_v1"
    assert SEGMENTATION_REVIEW_VERSION == "phase8_step3_clinician_mask_review_v1"
    paths = create_app().openapi()["paths"]
    assert "/api/v1/studies/{study_uuid}/viewer/review" in paths
    assert "/api/v1/studies/{study_uuid}/viewer/corrections" in paths
    assert "/api/v1/studies/{study_uuid}/viewer/review/history" in paths
    assert "post" in paths["/api/v1/studies/{study_uuid}/viewer/corrections"]


def test_review_states_keep_explicit_accept_reject_edit_semantics():
    assert SegmentationReviewAction.ACCEPT.value == "accept"
    assert SegmentationReviewAction.REJECT.value == "reject"
    assert SegmentationReviewAction.EDIT.value == "edit"
    assert SegmentationReviewStatus.ACCEPTED.value == "accepted"
    assert SegmentationReviewStatus.REJECTED.value == "rejected"
    assert SegmentationReviewStatus.EDITED.value == "edited"


def test_raw_cornerstone_labelmap_contract_uses_uint8_fortran_voxel_order():
    # x-fastest / NIfTI-compatible byte ordering: reshape(order="F").
    source = np.zeros((2, 3, 2), dtype=np.uint8)
    source[1, 0, 0] = 2
    source[0, 2, 1] = 4
    raw = source.reshape(-1, order="F").tobytes()
    decoded = decode_clinician_labelmap(raw, source.shape)
    assert np.array_equal(decoded, source)


def test_correction_label_contract_rejects_wrong_byte_count_and_unknown_labels():
    with pytest.raises(SegmentationReviewServiceError, match="exactly 8 uint8 voxels"):
        decode_clinician_labelmap(b"\x00" * 7, (2, 2, 2))
    bad = bytearray(b"\x00" * 8)
    bad[4] = 3
    with pytest.raises(SegmentationReviewServiceError, match="unsupported label"):
        decode_clinician_labelmap(bytes(bad), (2, 2, 2))


def test_brats_labelmap_rebuild_preserves_nested_wt_tc_et_semantics():
    labelmap = np.array([0, 1, 2, 4], dtype=np.uint8).reshape((2, 2, 1), order="F")
    tc, wt, et = brats_binary_masks_from_labelmap(labelmap)
    assert int(np.count_nonzero(et)) == 1
    assert int(np.count_nonzero(tc)) == 2  # label 1 + ET label 4
    assert int(np.count_nonzero(wt)) == 3  # every non-background BraTS label


def test_step3_frontend_enables_brush_review_and_safe_server_persistence():
    cornerstone = (FRONTEND / "components/viewer/CornerstoneMprViewer.jsx").read_text(encoding="utf-8")
    workspace = (FRONTEND / "components/viewer/ViewerWorkspace.jsx").read_text(encoding="utf-8")
    api = (FRONTEND / "lib/api.js").read_text(encoding="utf-8")
    source = "\n".join((cornerstone, workspace, api))
    for term in (
        "BrushTool",
        "FILL_INSIDE_CIRCLE",
        "ERASE_INSIDE_CIRCLE",
        "setActiveSegmentIndex",
        "exportRawLabelmap",
        "submitLabelmapCorrection",
        "submitSegmentationReview",
        "Accept",
        "Reject",
        "Correct mask",
    ):
        assert term in source
    assert "tools.segmentation.setActiveSegmentation" in cornerstone
    assert "activeSegmentation.setActiveSegmentation" not in cornerstone
    assert "source_checksum_sha256" in api
    assert "storage_key" not in source
    assert "/var/storage" not in source


def test_step3_medical_ui_keeps_motion_accessibility_and_correction_polish():
    styles = (FRONTEND / "app/globals.css").read_text(encoding="utf-8")
    workspace = (FRONTEND / "components/viewer/ViewerWorkspace.jsx").read_text(encoding="utf-8")
    assert "correction-deck" in styles
    assert "edit-live-indicator" in styles
    assert "review-toast" in styles
    assert "prefers-reduced-motion" in styles
    assert "MotionConfig" in workspace
    assert "AI-assisted imaging review only" in workspace
    assert "not a definitive GBM diagnosis" in workspace


def test_migration_chain_adds_append_only_review_revision_table():
    migration = (PROJECT_ROOT / "migrations/versions/20260816_0012_segmentation_review_revisions.py").read_text(encoding="utf-8")
    assert 'revision: str = "20260816_0012"' in migration
    assert '"20260815_0011"' in migration
    assert '"segmentation_review_revisions"' in migration
    assert '"revision_number"' in migration
    assert '"source_artifacts"' in migration
    assert '"result_artifacts"' in migration
