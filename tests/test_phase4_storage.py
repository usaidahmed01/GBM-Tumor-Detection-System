from __future__ import annotations

import hashlib
import io
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import StudyStatus
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.services.study_storage import attach_study_source_object
from gbm_ai.api.storage.local import (
    InvalidStorageKeyError,
    LocalObjectStore,
    ObjectTooLargeError,
    StorageError,
)


def test_local_store_streams_hashes_and_reads_back(tmp_path):
    storage = LocalObjectStore(
        tmp_path / "storage",
        max_object_bytes=1024 * 1024,
        chunk_bytes=8,
    )
    payload = b"synthetic-mri-object"
    key = "studies/test/source/object.bin"

    stored = storage.put_stream(key, io.BytesIO(payload))

    assert stored.storage_key == key
    assert stored.size_bytes == len(payload)
    assert stored.sha256 == hashlib.sha256(payload).hexdigest()
    assert storage.verify_checksum(key, stored.sha256) is True

    with storage.open_read(key) as source:
        assert source.read() == payload


def test_storage_key_cannot_escape_storage_root(tmp_path):
    storage = LocalObjectStore(
        tmp_path / "storage",
        max_object_bytes=1024,
    )

    with pytest.raises(InvalidStorageKeyError):
        storage.put_stream("../outside.bin", io.BytesIO(b"x"))

    with pytest.raises(InvalidStorageKeyError):
        storage.put_stream("/absolute/path.bin", io.BytesIO(b"x"))


def test_size_limit_aborts_without_committed_object(tmp_path):
    storage = LocalObjectStore(
        tmp_path / "storage",
        max_object_bytes=5,
        chunk_bytes=3,
    )
    key = "studies/test/source/too-large.bin"

    with pytest.raises(ObjectTooLargeError):
        storage.put_stream(key, io.BytesIO(b"123456"))

    assert storage.exists(key) is False
    assert list((tmp_path / "storage" / ".tmp").glob("*.part")) == []


def test_existing_object_is_never_silently_overwritten(tmp_path):
    storage = LocalObjectStore(
        tmp_path / "storage",
        max_object_bytes=1024,
    )
    key = "studies/test/source/object.bin"
    storage.put_stream(key, io.BytesIO(b"first"))

    with pytest.raises(StorageError):
        storage.put_stream(key, io.BytesIO(b"second"))

    with storage.open_read(key) as source:
        assert source.read() == b"first"


def test_study_storage_service_updates_database_reference(tmp_path):
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
    storage = LocalObjectStore(
        tmp_path / "storage",
        max_object_bytes=1024 * 1024,
    )

    with SessionLocal() as db:
        patient = create_patient(
            db,
            PatientCreate(
                patient_id="GBM-2026-STORAGE",
                age_years=45,
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

        payload = b"synthetic-upload"
        stored = attach_study_source_object(
            db,
            storage,
            study,
            io.BytesIO(payload),
        )

        assert study.status == StudyStatus.UPLOADED
        assert study.storage_key == stored.storage_key
        assert study.checksum_sha256 == hashlib.sha256(payload).hexdigest()
        assert storage.exists(study.storage_key)

    engine.dispose()
