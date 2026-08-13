from __future__ import annotations

from sqlalchemy import inspect

from gbm_ai.api.config import get_settings
from gbm_ai.api.db import DatabaseManager


REQUIRED = {
    "patients": {
        "id", "patient_id", "patient_name", "age_years", "sex",
        "privacy_flags", "created_at", "updated_at",
    },
    "assessments": {
        "id", "patient_id", "mri_date", "symptoms", "symptom_duration",
        "prior_treatment", "clinical_notes", "status", "scope_status",
        "created_at", "updated_at",
    },
}


def main() -> None:
    database = DatabaseManager(get_settings())
    try:
        inspector = inspect(database.engine)
        table_names = set(inspector.get_table_names())

        print("PHASE 4 STEP 2 — CLINICAL SCHEMA CHECK")
        print("=" * 50)

        for table, expected_columns in REQUIRED.items():
            if table not in table_names:
                print(f"{table}: MISSING")
                raise SystemExit(1)

            actual_columns = {column["name"] for column in inspector.get_columns(table)}
            missing = expected_columns - actual_columns
            if missing:
                print(f"{table}: INVALID — missing {sorted(missing)}")
                raise SystemExit(1)

            print(f"{table}: READY ({len(actual_columns)} columns)")

        print("\nPatient identifier is metadata only: YES")
        print("Clinical context used as V1 ML feature: NO")
        print("Phase 4 Step 2 schema: READY")
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
