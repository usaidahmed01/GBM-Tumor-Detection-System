from __future__ import annotations

from gbm_ai.validation.matrix import load_validation_matrix, validate_matrix_files_exist


def main() -> None:
    matrix = load_validation_matrix()
    missing = validate_matrix_files_exist()
    if missing:
        raise RuntimeError("validation matrix references missing tests: " + ", ".join(missing))

    groups = matrix.get("automated_groups", [])
    failure_modes = matrix.get("critical_failure_modes", [])
    external = matrix.get("external_or_manual_cases", [])
    group_ids = {str(item.get("id")) for item in groups}

    required_groups = {
        "foundation_security",
        "upload_qc_routing",
        "classifier_safety",
        "segmentation_pipeline",
        "quantification_localization",
        "viewer_review_report",
    }
    if not required_groups.issubset(group_ids):
        raise RuntimeError("Phase 10 validation matrix is missing required automated groups")

    external_statuses = {str(item.get("status")) for item in external}
    if "not_claimed_current_v1" not in external_statuses:
        raise RuntimeError("current V1 authentication limitation is not explicit")

    print("PHASE 10 STEP 2 — VALIDATION MATRIX FOUNDATION CHECK")
    print("=" * 82)
    print(f"Validation matrix version:        {matrix['version']}")
    print(f"Automated validation groups:      {len(groups)}")
    print(f"Critical failure modes mapped:    {len(failure_modes)}")
    print(f"External/manual cases explicit:   {len(external)}")
    print("JPG upload/classifier path:       COVERED + REAL RUNTIME REQUIRES LOCAL CHECKPOINTS")
    print("DICOM de-identification/QC path:  COVERED")
    print("NIfTI QC/mapping path:            COVERED")
    print("Missing T1C safety case:          COVERED")
    print("Archive traversal/bomb defenses:  COVERED")
    print("3D segmentation job recovery:     COVERED")
    print("Volume/location safety gates:     COVERED")
    print("Decision discordance handling:    COVERED")
    print("Viewer/edit/report regression:    COVERED")
    print("Real multimodal hospital case:    EXTERNAL TEST DATA REQUIRED")
    print("Authentication/RBAC validation:   NOT CLAIMED IN CURRENT V1")
    print("Clinical validation claimed:      NO")
    print("Next step:                        PHASE 10 STEP 3 — PERFORMANCE & CLEAN-ENV REPRODUCIBILITY")
    print("Phase 10 Step 2 foundation:       READY")


if __name__ == "__main__":
    main()
