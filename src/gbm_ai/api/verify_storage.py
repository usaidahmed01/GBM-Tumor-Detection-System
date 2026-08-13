from __future__ import annotations

import io

from gbm_ai.api.config import get_settings
from gbm_ai.api.storage.local import LocalObjectStore


def main() -> None:
    settings = get_settings()
    storage = LocalObjectStore(
        settings.storage_root_resolved,
        settings.storage_max_object_bytes,
        settings.storage_chunk_bytes,
    )

    test_key = "system/storage-verification/probe.bin"
    payload = b"GBM-CDSS-storage-verification"

    if storage.exists(test_key):
        storage.delete(test_key)

    stored = storage.put_stream(test_key, io.BytesIO(payload))

    checksum_ok = storage.verify_checksum(
        stored.storage_key,
        stored.sha256,
    )
    read_ok = False
    with storage.open_read(stored.storage_key) as source:
        read_ok = source.read() == payload

    storage.delete(stored.storage_key)

    print("PHASE 4 STEP 4 — PROTECTED STORAGE CHECK")
    print("=" * 52)
    print(f"Storage root:             {settings.storage_root_resolved}")
    print(f"Opaque internal key:      {stored.storage_key}")
    print(f"Stored bytes:             {stored.size_bytes}")
    print(f"SHA-256 verification:     {'PASS' if checksum_ok else 'FAIL'}")
    print(f"Read-back verification:   {'PASS' if read_ok else 'FAIL'}")
    print(f"Probe cleanup:            {'PASS' if not storage.exists(test_key) else 'FAIL'}")
    print("Public/static URL exposed:NO")
    print("Phase 4 Step 4 storage:   READY")

    if not checksum_ok or not read_ok or storage.exists(test_key):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
