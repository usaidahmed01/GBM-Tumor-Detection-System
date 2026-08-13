from __future__ import annotations

import io
import zipfile
from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.config import Settings
from gbm_ai.api.dependencies import get_db_session, get_object_store
from gbm_ai.api.main import create_app
from gbm_ai.api.middleware.request_body_limit import RequestBodyLimitMiddleware
from gbm_ai.api.models import Base
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.storage.local import LocalObjectStore
from gbm_ai.api.upload.intake import ArchivePolicyError, preflight_upload


def make_zip(entries: dict[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    buffer.seek(0)
    return buffer


def test_single_object_preflight_does_not_use_filename_or_extension():
    source = io.BytesIO(b"\x89PNG\r\n\x1a\nsynthetic")
    source.seek(3)

    result = preflight_upload(
        source,
        max_object_bytes=1024,
        max_archive_entries=100,
        max_archive_uncompressed_bytes=1024 * 1024,
        max_archive_single_entry_bytes=1024 * 1024,
        max_archive_compression_ratio=200.0,
    )

    assert result.upload_kind == "single_object"
    assert result.size_bytes == len(b"\x89PNG\r\n\x1a\nsynthetic")
    assert source.tell() == 3


def test_safe_zip_is_preflighted_without_extraction():
    source = make_zip(
        {
            "study/series1/0001.dcm": b"one",
            "study/series1/0002.dcm": b"two",
        }
    )

    result = preflight_upload(
        source,
        max_object_bytes=1024 * 1024,
        max_archive_entries=10,
        max_archive_uncompressed_bytes=1024 * 1024,
        max_archive_single_entry_bytes=1024 * 1024,
        max_archive_compression_ratio=200.0,
    )

    assert result.upload_kind == "zip_archive"
    assert result.archive_entry_count == 2
    assert result.archive_total_uncompressed_bytes == 6


def test_zip_path_traversal_is_rejected():
    source = make_zip({"../escape.dcm": b"bad"})

    with pytest.raises(ArchivePolicyError, match="path traversal"):
        preflight_upload(
            source,
            max_object_bytes=1024 * 1024,
            max_archive_entries=10,
            max_archive_uncompressed_bytes=1024 * 1024,
            max_archive_single_entry_bytes=1024 * 1024,
            max_archive_compression_ratio=200.0,
        )


def test_high_compression_ratio_archive_is_rejected():
    source = make_zip({"study/repeated.bin": b"\x00" * (2 * 1024 * 1024)})

    with pytest.raises(ArchivePolicyError, match="compression ratio"):
        preflight_upload(
            source,
            max_object_bytes=3 * 1024 * 1024,
            max_archive_entries=10,
            max_archive_uncompressed_bytes=3 * 1024 * 1024,
            max_archive_single_entry_bytes=3 * 1024 * 1024,
            max_archive_compression_ratio=50.0,
        )


def test_asgi_request_body_limit_returns_413():
    app = FastAPI()
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=1024,
        api_prefix="/api/v1",
    )

    @app.post("/api/v1/studies/{study_uuid}/upload")
    async def fake_upload(study_uuid: str):
        return {"ok": True}

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/studies/abc/upload",
            content=b"x" * 2048,
            headers={"content-type": "application/octet-stream"},
        )

    assert response.status_code == 413
    assert "size limit" in response.json()["detail"]


def test_real_upload_route_stores_bytes_but_keeps_format_pending(
    tmp_path,
    monkeypatch,
):
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
    db = SessionLocal()

    patient = create_patient(
        db,
        PatientCreate(
            patient_id="GBM-2026-UPLOAD",
            age_years=52,
            privacy_flags={"synthetic": True},
        ),
    )
    assessment = create_assessment(
        db,
        AssessmentCreate(
            patient_uuid=patient.id,
            mri_date=date(2026, 8, 14),
            prior_treatment=False,
        ),
    )
    study = create_study(
        db,
        StudyCreate(assessment_uuid=assessment.id),
    )

    settings = Settings(
        environment="test",
        storage_root=tmp_path / "app-storage",
        storage_max_object_bytes=1024 * 1024,
        upload_max_request_bytes=2 * 1024 * 1024,
        upload_max_archive_uncompressed_bytes=4 * 1024 * 1024,
        upload_max_archive_single_entry_bytes=4 * 1024 * 1024,
    )
    app = create_app(settings)

    class FakeDatabase:
        def dispose(self):
            pass

    monkeypatch.setattr(
        "gbm_ai.api.main.DatabaseManager",
        lambda settings: FakeDatabase(),
    )

    storage = LocalObjectStore(
        tmp_path / "test-storage",
        max_object_bytes=1024 * 1024,
    )

    def override_session():
        yield db

    def override_storage():
        return storage

    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_object_store] = override_storage

    payload = b"\x89PNG\r\n\x1a\nsynthetic-image-content"

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/studies/{study.id}/upload",
            files={
                "file": (
                    "misleading-name.dcm",
                    payload,
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["study_status"] == "uploaded"
    assert body["source_format"] == "pending"
    assert body["upload_kind"] == "single_object"
    assert body["format_detection_status"] == "pending"
    assert body["original_filename_stored"] is False
    assert body["stored_size_bytes"] == len(payload)

    db.refresh(study)
    assert study.storage_key is not None
    assert "misleading-name.dcm" not in study.storage_key
    assert study.checksum_sha256 == body["sha256"]
    assert storage.exists(study.storage_key)

    db.close()
    engine.dispose()
