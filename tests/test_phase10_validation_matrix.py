from __future__ import annotations

import json
from pathlib import Path

from gbm_ai.validation.matrix import (
    PROJECT_ROOT,
    load_validation_matrix,
    validate_matrix_files_exist,
)


def test_phase10_validation_matrix_covers_all_major_system_layers():
    matrix = load_validation_matrix()
    group_ids = {item["id"] for item in matrix["automated_groups"]}
    assert {
        "foundation_security",
        "upload_qc_routing",
        "classifier_safety",
        "segmentation_pipeline",
        "quantification_localization",
        "viewer_review_report",
    }.issubset(group_ids)


def test_validation_matrix_does_not_reference_missing_test_files():
    assert validate_matrix_files_exist() == []


def test_critical_failure_mode_matrix_contains_safety_boundaries():
    matrix = load_validation_matrix()
    ids = {item["id"] for item in matrix["critical_failure_modes"]}
    assert {
        "zip_path_traversal",
        "archive_bomb_high_ratio",
        "missing_t1c",
        "invalid_nifti_reused_channel",
        "volumetric_2d_bridge",
        "low_probability_false_reassurance",
        "classifier_segmentation_disagreement",
        "stale_background_worker",
        "rejected_segmentation",
        "unreviewed_segmentation_report",
    }.issubset(ids)


def test_external_requirements_are_never_falsely_marked_as_passed():
    matrix = load_validation_matrix()
    external = {item["id"]: item["status"] for item in matrix["external_or_manual_cases"]}
    assert external["real_jpg_classifier_runtime"] == "requires_local_classifier_checkpoints"
    assert external["real_nifti_multimodal_e2e"] == "requires_deidentified_test_case_and_runtime_assets"
    assert external["real_dicom_multiseries_e2e"] == "requires_deidentified_test_case_and_runtime_assets"
    assert external["authentication_authorization"] == "not_claimed_current_v1"


def test_runtime_validation_reports_are_git_ignored():
    content = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/var/validation/" in content
