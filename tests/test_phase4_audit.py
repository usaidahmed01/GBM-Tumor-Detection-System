from __future__ import annotations

import io
import uuid
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.config import Settings
from gbm_ai.api.main import create_app
from gbm_ai.api.models import Base
from gbm_ai.api.models.audit import (
    AuditAction,
    AuditActorType,
    AuditEntityType,
    AuditLog,
)
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.audit import (
    UnsafeAuditContextError,
    record_audit_event,
    sanitize_technical_context,
)
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.services.study_storage import attach_study_source_object
from gbm_ai.api.storage.local import LocalObjectStore


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
    )
    with SessionLocal() as db:
        yield db
    engine.dispose()


def test_audit_context_whitelist_rejects_patient_or_clinical_fields():
    with pytest.raises(UnsafeAuditContextError):
        sanitize_technical_context(
            {
                "patient_name": "Synthetic Demo",
                "status": "uploaded",
            }
        )

    with pytest.raises(UnsafeAuditContextError):
        sanitize_technical_context(
            {
                "clinical_notes": "headache and weakness",
            }
        )


def test_audit_event_persists_only_whitelisted_technical_context(session):
    entity_id = uuid.uuid4()

    event = record_audit_event(
        session,
        action=AuditAction.ANALYSIS_STARTED,
        entity_type=AuditEntityType.ANALYSIS_RUN,
        entity_uuid=entity_id,
        actor_type=AuditActorType.SYSTEM,
        request_id="abc123",
        technical_context={
            "model_name": "gbm_classifier",
            "model_version": "development",
            "status": "running",
        },
    )

    stored = session.scalar(
        select(AuditLog).where(AuditLog.id == event.id)
    )
    assert stored is not None
    assert stored.entity_uuid == entity_id
    assert stored.technical_context["status"] == "running"
    assert "patient_name" not in stored.technical_context


def test_audit_records_are_append_only_at_orm_layer(session):
    event = record_audit_event(
        session,
        action=AuditAction.STUDY_VIEWED,
        entity_type=AuditEntityType.STUDY,
        entity_uuid=uuid.uuid4(),
        technical_context={
            "operation": "view",
            "result": "success",
        },
    )

    event.actor_id = "mutated"
    with pytest.raises(RuntimeError, match="append-only"):
        session.commit()
    session.rollback()


def test_request_id_is_server_generated_and_returned(monkeypatch):
    settings = Settings(environment="test")
    app = create_app(settings)

    class FakeDatabase:
        def dispose(self):
            pass

    monkeypatch.setattr(
        "gbm_ai.api.main.DatabaseManager",
        lambda settings: FakeDatabase(),
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/live",
            headers={"X-Request-ID": "client-controlled-id"},
        )

    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    assert request_id != "client-controlled-id"
    assert len(request_id) == 32
    int(request_id, 16)


def test_study_storage_creates_audit_event_in_same_operation(
    session,
    tmp_path,
):
    patient = create_patient(
        session,
        PatientCreate(
            patient_id="GBM-2026-AUDIT",
            age_years=50,
            privacy_flags={"synthetic": True},
        ),
    )
    assessment = create_assessment(
        session,
        AssessmentCreate(
            patient_uuid=patient.id,
            mri_date=date(2026, 8, 14),
            prior_treatment=False,
        ),
    )
    study = create_study(
        session,
        StudyCreate(assessment_uuid=assessment.id),
    )

    storage = LocalObjectStore(
        tmp_path / "storage",
        max_object_bytes=1024 * 1024,
    )
    stored = attach_study_source_object(
        session,
        storage,
        study,
        io.BytesIO(b"synthetic-upload"),
        request_id="req-test",
        actor_type=AuditActorType.DEMO_USER,
    )

    audit = session.scalar(
        select(AuditLog).where(
            AuditLog.entity_uuid == study.id,
            AuditLog.action == AuditAction.STUDY_SOURCE_STORED,
        )
    )

    assert audit is not None
    assert audit.request_id == "req-test"
    assert audit.technical_context["storage_backend"] == "local"
    assert audit.technical_context["size_bytes"] == len(b"synthetic-upload")
    assert audit.technical_context["sha256"] == stored.sha256
