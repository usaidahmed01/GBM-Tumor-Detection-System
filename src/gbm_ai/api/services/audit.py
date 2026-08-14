from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import Session

from gbm_ai.api.models.audit import (
    AuditAction,
    AuditActorType,
    AuditEntityType,
    AuditLog,
)


class UnsafeAuditContextError(ValueError):
    pass


# Whitelist instead of a blacklist: only deliberately technical metadata may be
# placed in audit JSON. Clinical/patient fields are intentionally absent.
ALLOWED_TECHNICAL_CONTEXT_KEYS = {
    "status",
    "source_format",
    "modality",
    "model_name",
    "model_version",
    "model_role",
    "architecture",
    "preprocessing_version",
    "threshold_version",
    "calibration_version",
    "size_bytes",
    "sha256",
    "error_type",
    "storage_backend",
    "result",
    "operation",
    "http_method",
    "path_template",
    "qc_status",
    "manual_review_required",
    "reason_count",
    "sequence_label",
    "sequence_status",
    "routing_status",
    "brain_scope_status",
    "eligible_capability_count",
    "review_capability_count",
}


def sanitize_technical_context(
    context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not context:
        return {}

    unknown = set(context) - ALLOWED_TECHNICAL_CONTEXT_KEYS
    if unknown:
        raise UnsafeAuditContextError(
            "Audit technical_context contains non-whitelisted key(s): "
            + ", ".join(sorted(unknown))
        )

    sanitized: dict[str, Any] = {}
    for key, value in context.items():
        if value is None:
            sanitized[key] = None
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            raise UnsafeAuditContextError(
                f"Audit technical_context value for {key!r} must be a "
                "simple scalar, not nested clinical/object data."
            )

    return sanitized


def record_audit_event(
    db: Session,
    *,
    action: AuditAction,
    entity_type: AuditEntityType,
    entity_uuid: uuid.UUID,
    actor_type: AuditActorType = AuditActorType.SYSTEM,
    actor_id: str | None = None,
    request_id: str | None = None,
    technical_context: Mapping[str, Any] | None = None,
    commit: bool = True,
) -> AuditLog:
    safe_context = sanitize_technical_context(technical_context)

    event = AuditLog(
        actor_type=actor_type,
        actor_id=actor_id,
        action=action,
        entity_type=entity_type,
        entity_uuid=entity_uuid,
        request_id=request_id,
        technical_context=safe_context,
    )
    db.add(event)

    if commit:
        db.commit()
        db.refresh(event)
    else:
        db.flush()

    return event
