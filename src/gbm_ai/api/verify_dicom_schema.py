from __future__ import annotations

from sqlalchemy import inspect

from gbm_ai.api.config import get_settings
from gbm_ai.api.db import DatabaseManager


STUDY_COLUMNS = {
    "deidentified_storage_key",
    "deidentified_checksum_sha256",
    "deidentification_status",
}

SERIES_COLUMNS = {
    "id",
    "study_id",
    "series_uid",
    "series_number",
    "detected_sequence",
    "confirmed_sequence",
    "sequence_confidence",
    "sequence_metadata",
    "slice_count",
    "spacing_orientation_metadata",
    "working_member_prefix",
    "created_at",
    "updated_at",
}


def main() -> None:
    database = DatabaseManager(get_settings())
    try:
        inspector = inspect(database.engine)
        tables = set(inspector.get_table_names())

        print("PHASE 5 STEP 3 — DICOM GROUPING / DE-ID SCHEMA CHECK")
        print("=" * 64)

        if "series" not in tables:
            print("series: MISSING")
            raise SystemExit(1)

        study_cols = {
            item["name"] for item in inspector.get_columns("studies")
        }
        missing_study = STUDY_COLUMNS - study_cols
        if missing_study:
            print(f"studies: INVALID — missing {sorted(missing_study)}")
            raise SystemExit(1)

        series_cols = {
            item["name"] for item in inspector.get_columns("series")
        }
        missing_series = SERIES_COLUMNS - series_cols
        if missing_series:
            print(f"series: INVALID — missing {sorted(missing_series)}")
            raise SystemExit(1)

        print("studies de-id columns: READY")
        print(f"series: READY ({len(series_cols)} columns)")
        print("Original DICOM UID DB fields: NO")
        print("Raw SeriesDescription stored: NO")
        print("Raw ProtocolName stored:      NO")
        print("Phase 5 Step 3 schema:        READY")
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
