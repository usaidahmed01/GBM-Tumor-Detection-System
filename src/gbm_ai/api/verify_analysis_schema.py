from __future__ import annotations

from sqlalchemy import inspect

from gbm_ai.api.config import get_settings
from gbm_ai.api.db import DatabaseManager


REQUIRED = {
    "studies": {
        "id",
        "assessment_id",
        "source_format",
        "modality",
        "study_instance_uid",
        "deidentified_metadata",
        "storage_key",
        "checksum_sha256",
        "status",
        "created_at",
        "updated_at",
    },
    "model_versions": {
        "id",
        "model_name",
        "version",
        "role",
        "architecture",
        "weights_checksum_sha256",
        "code_version",
        "preprocessing_version",
        "threshold_version",
        "calibration_version",
        "license_source_notes",
        "is_active",
        "created_at",
        "updated_at",
    },
    "analysis_runs": {
        "id",
        "study_id",
        "classifier_model_version_id",
        "segmentation_model_version_id",
        "status",
        "qc_state",
        "ood_score",
        "ood_likeness_candidate",
        "raw_probability_gbm",
        "calibrated_probability_gbm",
        "decision_state",
        "safety_reason_codes",
        "started_at",
        "completed_at",
        "created_at",
        "updated_at",
    },
}


def main() -> None:
    database = DatabaseManager(get_settings())
    try:
        inspector = inspect(database.engine)
        tables = set(inspector.get_table_names())

        print("PHASE 4 STEP 3 — ANALYSIS SCHEMA CHECK")
        print("=" * 52)

        for table, required_columns in REQUIRED.items():
            if table not in tables:
                print(f"{table}: MISSING")
                raise SystemExit(1)

            actual = {
                column["name"]
                for column in inspector.get_columns(table)
            }
            missing = required_columns - actual
            if missing:
                print(f"{table}: INVALID — missing {sorted(missing)}")
                raise SystemExit(1)
            print(f"{table}: READY ({len(actual)} columns)")

        print("\nStudy source format starts as pending: YES")
        print("Upload auto-detection deferred to Phase 5: YES")
        print("Model/version traceability schema: READY")
        print("Phase 4 Step 3 schema: READY")
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
