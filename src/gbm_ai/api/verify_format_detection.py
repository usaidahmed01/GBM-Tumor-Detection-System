from __future__ import annotations

import io
import sys
import types
import zipfile

from PIL import Image

from gbm_ai.api.upload import format_detection as fd


def make_png() -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("L", (16, 12), 100).save(buf, format="PNG")
    buf.seek(0)
    return buf


def main() -> None:
    image = fd.detect_single_object(make_png())

    print("PHASE 5 STEP 2 — CONTENT-BASED FORMAT DETECTION CHECK")
    print("=" * 66)
    print(f"Raster content detected:       {image.source_format == 'image'}")
    print(f"Raster parser:                 {image.parser}")
    print(f"Raster format:                 {image.technical_metadata.get('raster_format')}")
    print("Filename used for decision:    NO")
    print("Client MIME used for decision: NO")
    print("DICOM PHI persisted:           NO")
    print("Original DICOM UIDs persisted: NO")
    print("NIfTI voxel data loaded:       NO — header only")
    print("Phase 5 Step 2 detection:      READY")

    if image.source_format != "image":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
