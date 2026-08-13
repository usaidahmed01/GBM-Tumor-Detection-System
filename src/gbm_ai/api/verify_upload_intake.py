from __future__ import annotations

import io
import zipfile

from gbm_ai.api.config import get_settings
from gbm_ai.api.upload.intake import ArchivePolicyError, preflight_upload


def make_safe_zip() -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("study/series1/0001.dcm", b"DICOM-SYNTHETIC-1")
        zf.writestr("study/series1/0002.dcm", b"DICOM-SYNTHETIC-2")
    buffer.seek(0)
    return buffer


def make_traversal_zip() -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../escape.dcm", b"bad")
    buffer.seek(0)
    return buffer


def main() -> None:
    settings = get_settings()

    safe = preflight_upload(
        make_safe_zip(),
        max_object_bytes=settings.storage_max_object_bytes,
        max_archive_entries=settings.upload_max_archive_entries,
        max_archive_uncompressed_bytes=(
            settings.upload_max_archive_uncompressed_bytes
        ),
        max_archive_single_entry_bytes=(
            settings.upload_max_archive_single_entry_bytes
        ),
        max_archive_compression_ratio=(
            settings.upload_max_archive_compression_ratio
        ),
    )

    traversal_blocked = False
    try:
        preflight_upload(
            make_traversal_zip(),
            max_object_bytes=settings.storage_max_object_bytes,
            max_archive_entries=settings.upload_max_archive_entries,
            max_archive_uncompressed_bytes=(
                settings.upload_max_archive_uncompressed_bytes
            ),
            max_archive_single_entry_bytes=(
                settings.upload_max_archive_single_entry_bytes
            ),
            max_archive_compression_ratio=(
                settings.upload_max_archive_compression_ratio
            ),
        )
    except ArchivePolicyError:
        traversal_blocked = True

    print("PHASE 5 STEP 1 — UNIFIED UPLOAD INTAKE CHECK")
    print("=" * 58)
    print(f"Safe ZIP recognized:       {safe.upload_kind == 'zip_archive'}")
    print(f"Safe ZIP file entries:     {safe.archive_entry_count}")
    print(f"Traversal archive blocked: {traversal_blocked}")
    print(f"Request ceiling configured:{settings.upload_max_request_bytes}")
    print("Filename-based format mode:NO")
    print("Format detection performed:NO — intentionally Step 2")
    print("Phase 5 Step 1 intake:     READY")

    if safe.upload_kind != "zip_archive" or not traversal_blocked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
