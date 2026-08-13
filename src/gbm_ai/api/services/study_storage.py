from __future__ import annotations

from typing import BinaryIO

from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import Study, StudyStatus
from gbm_ai.api.models.audit import (
    AuditAction,
    AuditActorType,
    AuditEntityType,
)
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.storage.local import LocalObjectStore, StoredObject


class StudyStorageStateError(Exception):
    pass


def attach_study_source_object(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    source: BinaryIO,
    *,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> StoredObject:
    """
    Internal service used by the future Phase 5 upload route.

    Storage + Study reference + AuditLog are handled as one logical operation.
    File format detection/QC remains intentionally deferred to Phase 5.
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
        record_audit_event(
            db,
            action=AuditAction.STUDY_SOURCE_STORED,
            entity_type=AuditEntityType.STUDY,
            entity_uuid=study.id,
            actor_type=actor_type,
            actor_id=actor_id,
            request_id=request_id,
            technical_context={
                "status": StudyStatus.UPLOADED.value,
                "storage_backend": "local",
                "size_bytes": stored.size_bytes,
                "sha256": stored.sha256,
            },
            commit=False,
        )
        db.commit()
        db.refresh(study)
    except Exception:
        db.rollback()
        if storage.exists(stored.storage_key):
            storage.delete(stored.storage_key)
        raise

    return stored
