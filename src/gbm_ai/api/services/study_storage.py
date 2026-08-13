from __future__ import annotations

import uuid
from typing import BinaryIO

from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import Study, StudyStatus
from gbm_ai.api.storage.local import LocalObjectStore, StoredObject


class StudyStorageStateError(Exception):
    pass


def attach_study_source_object(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    source: BinaryIO,
) -> StoredObject:
    """
    Internal service used by the future Phase 5 upload route.

    It intentionally does NOT determine whether content is JPG/PNG/DICOM/NIfTI.
    Format parsing and QC belong to Phase 5.
    """
    if study.storage_key is not None:
        raise StudyStorageStateError(
            "study already has a stored source object"
        )

    key = storage.generate_study_source_key(study.id)
    stored = storage.put_stream(key, source)

    study.storage_key = stored.storage_key
    study.checksum_sha256 = stored.sha256
    study.status = StudyStatus.UPLOADED

    try:
        db.commit()
        db.refresh(study)
    except Exception:
        db.rollback()
        # Prevent orphaned filesystem object if DB update fails.
        if storage.exists(stored.storage_key):
            storage.delete(stored.storage_key)
        raise

    return stored
