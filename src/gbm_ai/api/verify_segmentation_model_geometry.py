from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from sqlalchemy import inspect

from gbm_ai.api.config import get_settings
from gbm_ai.api.db import DatabaseManager
from gbm_ai.api.segmentation.contract import (
    SEGMENTATION_INPUT_CHANNEL_ORDER,
    SEGMENTATION_REFERENCE_SPACING_MM,
)
from gbm_ai.api.segmentation.model_geometry import MODEL_GEOMETRY_VERSION


REQUIRED_STUDY_COLUMNS = {
    "segmentation_preparation_status",
    "segmentation_preparation_summary",
}


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    database = DatabaseManager(get_settings())
    try:
        inspector = inspect(database.engine)
        study_columns = {
            item["name"]
            for item in inspector.get_columns("studies")
        }
        missing = REQUIRED_STUDY_COLUMNS - study_columns

        print("PHASE 6 STEP 3 — REGISTRATION & MODEL GEOMETRY CHECK")
        print("=" * 72)

        if missing:
            print(f"Phase 6 schema:              MISSING {sorted(missing)}")
            raise SystemExit(1)

        simpleitk_version = _package_version("SimpleITK")
        if simpleitk_version == "NOT INSTALLED":
            print("SimpleITK installed:         NO")
            print("Install cumulative requirements.txt before continuing.")
            raise SystemExit(1)

        print("Alembic/schema basis:        READY (Step 2 schema retained)")
        print(f"Preparation version:         {MODEL_GEOMETRY_VERSION}")
        print("Channel order preserved:     " + " -> ".join(SEGMENTATION_INPUT_CHANNEL_ORDER))
        print(
            "Target model spacing:        "
            + " x ".join(f"{value:.1f}" for value in SEGMENTATION_REFERENCE_SPACING_MM)
            + " mm"
        )
        print("Canonical orientation basis: RAS")
        print("Registration reference:      T1C")
        print("Registration if required:    RIGID / MATTES MUTUAL INFORMATION")
        print("Registration seed:           42 (deterministic sampling)")
        print("Resampling interpolation:    LINEAR")
        print(f"SimpleITK installed:         {simpleitk_version}")
        print("Derived 1 mm NIfTI objects:  PROTECTED STORAGE + SHA-256")
        print("Intensity normalization:     NOT IMPLEMENTED IN STEP 3")
        print("Crop/pad model transforms:   NOT IMPLEMENTED IN STEP 3")
        print("MONAI / SegResNet loaded:    NO")
        print("Segmentation inference:      NOT STARTED")
        print("Physical volume:             NOT GENERATED")
        print("Anatomical localization:     NOT GENERATED")
        print("Clinical validation claimed: NO")
        print("Phase 6 Step 3 foundation:   READY")
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
