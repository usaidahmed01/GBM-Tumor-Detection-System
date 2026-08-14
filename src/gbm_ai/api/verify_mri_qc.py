from __future__ import annotations

from sqlalchemy import inspect

from gbm_ai.api.config import get_settings
from gbm_ai.api.db import DatabaseManager
from gbm_ai.api.qc.sequence_detection import detect_series_sequence


REQUIRED_STUDY_COLUMNS = {
    "qc_status",
    "qc_summary",
}


def main() -> None:
    database = DatabaseManager(get_settings())
    try:
        inspector = inspect(database.engine)
        study_columns = {
            item["name"]
            for item in inspector.get_columns("studies")
        }

        missing = REQUIRED_STUDY_COLUMNS - study_columns

        print("PHASE 5 STEP 4 — MRI QC / SEQUENCE DETECTION CHECK")
        print("=" * 64)

        if missing:
            print(f"studies QC columns: MISSING {sorted(missing)}")
            raise SystemExit(1)

        flair = detect_series_sequence(
            {
                "series_description_tokens": ["t2", "flair", "axial"],
                "protocol_name_tokens": ["flair"],
                "repetition_time_ms": 9000,
                "echo_time_ms": 120,
                "inversion_time_ms": 2500,
                "contrast_metadata_present": False,
            }
        )

        t1c = detect_series_sequence(
            {
                "series_description_tokens": ["t1", "post", "contrast"],
                "protocol_name_tokens": ["mprage"],
                "repetition_time_ms": 600,
                "echo_time_ms": 10,
                "contrast_metadata_present": True,
            }
        )

        print("studies QC columns:          READY")
        print(f"FLAIR heuristic check:       {flair.state}")
        print(f"T1c heuristic check:         {t1c.state}")
        print("Low-confidence confirmation: REQUIRED")
        print("QC thresholds clinical:      NO — engineering preflight only")
        print("Inference started by QC:     NO")
        print("Phase 5 Step 4 foundation:   READY")

        if flair.state != "FLAIR" or t1c.state != "T1C":
            raise SystemExit(1)
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
