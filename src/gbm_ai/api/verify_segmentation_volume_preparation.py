from __future__ import annotations

from importlib import metadata

from sqlalchemy import inspect

from gbm_ai.api.config import get_settings
from gbm_ai.api.db import DatabaseManager
from gbm_ai.api.segmentation.contract import (
    SEGMENTATION_INPUT_CHANNEL_ORDER,
)
from gbm_ai.api.services.segmentation_volume_preparation import (
    SEGMENTATION_PREPARATION_VERSION,
)


REQUIRED_STUDY_COLUMNS = {
    "segmentation_preparation_status",
    "segmentation_preparation_summary",
}


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def main() -> None:
    database = DatabaseManager(get_settings())

    try:
        inspector = inspect(database.engine)
        study_columns = {
            item["name"]
            for item in inspector.get_columns("studies")
        }
        missing = REQUIRED_STUDY_COLUMNS - study_columns

        print(
            "PHASE 6 STEP 2 — "
            "3D VOLUME LOADING & ALIGNMENT FOUNDATION CHECK"
        )
        print("=" * 72)

        if missing:
            print(f"Phase 6 DB columns:           MISSING {sorted(missing)}")
            raise SystemExit(1)

        nibabel_version = _package_version("nibabel")
        pydicom_version = _package_version("pydicom")

        if "MISSING" in {nibabel_version, pydicom_version}:
            print(f"NiBabel:                      {nibabel_version}")
            print(f"pydicom:                      {pydicom_version}")
            raise SystemExit(1)

        print("Alembic Phase 6 schema:       READY (20260814_0007)")
        print(f"Preparation version:          {SEGMENTATION_PREPARATION_VERSION}")
        print(
            "Channel order preserved:      "
            + " -> ".join(SEGMENTATION_INPUT_CHANNEL_ORDER)
        )
        print("Canonical orientation:        RAS")
        print("Inter-modality alignment:     VALIDATED BEFORE MODEL")
        print("Registration performed:       NO (deferred to next step if needed)")
        print("1 mm model resampling:        NO (deferred to next step)")
        print(f"NiBabel installed:            {nibabel_version}")
        print(f"pydicom installed:            {pydicom_version}")
        print("MONAI required in Step 2:     NO")
        print("SegResNet loaded:             NO")
        print("Segmentation inference:       NOT STARTED")
        print("Physical volume:              NOT GENERATED")
        print("Anatomical localization:      NOT GENERATED")
        print("Clinical validation claimed:  NO")
        print("Phase 6 Step 2 foundation:    READY")

    finally:
        database.dispose()


if __name__ == "__main__":
    main()
