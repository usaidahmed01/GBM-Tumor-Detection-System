from __future__ import annotations

from sqlalchemy import inspect

from gbm_ai.api.config import get_settings
from gbm_ai.api.db import DatabaseManager
from gbm_ai.api.segmentation.contract import (
    SEGMENTATION_MODEL_CONTRACT,
)


REQUIRED_STUDY_COLUMNS = {
    "source_format",
    "storage_key",
    "deidentified_storage_key",
    "qc_status",
    "qc_summary",
    "nifti_sequence_mapping",
    "capability_routing_status",
    "capability_summary",
}

REQUIRED_SERIES_COLUMNS = {
    "detected_sequence",
    "confirmed_sequence",
    "working_member_prefix",
    "spacing_orientation_metadata",
}


def main() -> None:
    database = DatabaseManager(get_settings())

    try:
        inspector = inspect(database.engine)

        study_columns = {
            item["name"]
            for item in inspector.get_columns("studies")
        }

        series_columns = {
            item["name"]
            for item in inspector.get_columns("series")
        }

        missing_study = (
            REQUIRED_STUDY_COLUMNS - study_columns
        )

        missing_series = (
            REQUIRED_SERIES_COLUMNS - series_columns
        )

        print(
            "PHASE 6 STEP 1 — "
            "3D SEGMENTATION PREFLIGHT CHECK"
        )
        print("=" * 64)

        if missing_study or missing_series:
            if missing_study:
                print(
                    "studies columns: MISSING "
                    f"{sorted(missing_study)}"
                )

            if missing_series:
                print(
                    "series columns:  MISSING "
                    f"{sorted(missing_series)}"
                )

            raise SystemExit(1)

        contract = SEGMENTATION_MODEL_CONTRACT

        print("Phase 5 routing schema:       READY")
        print(
            "Required input order:         "
            + " -> ".join(
                contract.required_input_channel_order
            )
        )
        print(
            "Expected output channels:     "
            + " / ".join(
                contract.output_channel_order
            )
        )
        print(
            "Reference spacing contract:   "
            "1.0 x 1.0 x 1.0 mm"
        )
        print("DICOM/NIfTI only:             YES")
        print("Orientation/alignment gate:   REQUIRED")
        print(
            "MONAI runtime loading:        "
            "NOT IMPLEMENTED IN STEP 1"
        )
        print("Segmentation inference:       NOT STARTED")
        print("Physical volume:              NOT GENERATED")
        print("Anatomical localization:      NOT GENERATED")
        print("Clinical validation claimed:  NO")
        print("Phase 6 Step 1 foundation:    READY")

    finally:
        database.dispose()


if __name__ == "__main__":
    main()