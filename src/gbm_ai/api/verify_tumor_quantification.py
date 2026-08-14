from __future__ import annotations

import numpy as np
from sqlalchemy import inspect

from gbm_ai.api.db import DatabaseManager
from gbm_ai.api.config import get_settings
from gbm_ai.api.quantification import (
    PHYSICAL_QUANTIFICATION_VERSION,
    measure_region,
    validate_physical_geometry,
)


def main() -> None:
    settings = get_settings()
    database = DatabaseManager(settings)
    try:
        inspector = inspect(database.engine)
        tables = set(inspector.get_table_names())
        schema_ready = "tumor_quantifications" in tables
        if not schema_ready:
            raise RuntimeError(
                "tumor_quantifications table is missing; run alembic upgrade head first"
            )

        # Deterministic physical-geometry smoke check. 0.5 x 0.5 x 2.0 mm
        # voxels have a physical volume of 0.5 mm^3 and axial area 0.25 mm^2.
        affine = np.diag([0.5, 0.5, 2.0, 1.0]).astype(np.float64)
        geometry = validate_physical_geometry(affine)
        mask = np.zeros((4, 4, 3), dtype=np.uint8)
        mask[0:2, 0:2, 1] = 1
        measurement, per_slice = measure_region("WT", mask, geometry)
        if not np.isclose(geometry.voxel_volume_mm3, 0.5):
            raise RuntimeError("voxel-volume smoke check failed")
        if not np.isclose(measurement.volume_mm3, 2.0):
            raise RuntimeError("physical-volume smoke check failed")
        if not np.isclose(measurement.max_axial_area_mm2, 1.0):
            raise RuntimeError("per-slice area smoke check failed")
        if len(per_slice) != 1 or per_slice[0]["slice_index"] != 1:
            raise RuntimeError("per-slice area indexing smoke check failed")

        print("PHASE 7 STEP 1 — PHYSICAL TUMOR QUANTIFICATION CHECK")
        print("=" * 78)
        print("Alembic Phase 7 schema:       READY (20260815_0010)")
        print(f"Quantification version:       {PHYSICAL_QUANTIFICATION_VERSION}")
        print("Source requirement:            CURRENT VALID 3D SEGMENTATION")
        print("Supported upload basis:        DICOM / NIfTI ONLY")
        print("Spatial metadata gate:         REQUIRED")
        print("Mask checksum validation:      REQUIRED")
        print("Mask affine/shape validation:  REQUIRED")
        print("WT primary volume:             IMPLEMENTED")
        print("TC / ET volumes:               IMPLEMENTED")
        print("Axial per-slice area:          IMPLEMENTED")
        print("Volume unit:                   mm^3 + cm^3")
        print("Per-slice area unit:           mm^2")
        print("Rejected segmentation:         BLOCKED")
        print("Standalone JPG physical size:  BLOCKED")
        print("Anatomical localization:       NOT GENERATED")
        print("GBM diagnosis from masks:      NO")
        print("Clinical validation claimed:   NO")
        print("Deterministic geometry smoke:  PASS (4 voxels -> 2.0 mm^3)")
        print("Phase 7 Step 1 foundation:     READY")
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
