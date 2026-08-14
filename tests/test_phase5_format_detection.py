from __future__ import annotations

import gzip
import io
import zipfile
from datetime import date

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.config import Settings
from gbm_ai.api.dependencies import get_db_session, get_object_store
from gbm_ai.api.main import create_app
from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import SourceFormat, StudyStatus
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.storage.local import LocalObjectStore
from gbm_ai.api.upload.format_detection import (
    NonMRIInputError,
    detect_single_object,
    detect_zip_contents,
)


def make_png(width=20, height=12) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), 120).save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg(width=18, height=10) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(
        buf, format="JPEG"
    )
    return buf.getvalue()


def make_zip(entries: dict[str, bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    buf.seek(0)
    return buf


def test_png_is_detected_from_content_not_filename():
    detected = detect_single_object(io.BytesIO(make_png()))

    assert detected.source_format == "image"
    assert detected.parser == "pillow"
    assert detected.modality == "UNKNOWN"
    assert detected.technical_metadata["raster_format"] == "PNG"
    assert detected.technical_metadata["width"] == 20


def test_zip_of_supported_raster_images_is_classified_by_contents():
    source = make_zip(
        {
            "renamed-file.dcm": make_png(),
            "another-file.bin": make_jpeg(),
            "README.txt": b"synthetic package notes",
        }
    )

    detected = detect_zip_contents(source)

    assert detected.source_format == "image"
    assert detected.modality == "UNKNOWN"
    assert detected.technical_metadata["recognized_entry_count"] == 2
    assert detected.technical_metadata["unknown_entry_count"] == 1


def test_real_pydicom_mr_detection_when_dependency_is_installed():
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        None,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "MR"
    ds.Rows = 16
    ds.Columns = 16

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, enforce_file_format=True)
    buf.seek(0)

    detected = detect_single_object(buf)

    assert detected.source_format == "dicom"
    assert detected.modality == "MR"
    assert detected.parser == "pydicom"
    assert "PatientName" not in detected.technical_metadata
    assert "StudyInstanceUID" not in detected.technical_metadata


def test_real_pydicom_ct_is_rejected_as_non_mri_when_installed():
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        None,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, enforce_file_format=True)
    buf.seek(0)

    with pytest.raises(NonMRIInputError):
        detect_single_object(buf)


def test_real_nifti1_and_gzip_detection_when_dependency_is_installed():
    nib = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")

    img = nib.Nifti1Image(
        np.zeros((8, 9, 10), dtype=np.float32),
        np.eye(4),
    )
    raw = img.to_bytes()

    detected = detect_single_object(io.BytesIO(raw))
    assert detected.source_format == "nifti"
    assert detected.technical_metadata["nifti_version"] == 1
    assert detected.technical_metadata["shape"] == [8, 9, 10]
    assert detected.technical_metadata["gzip_wrapped"] is False

    gz = gzip.compress(raw)
    detected_gz = detect_single_object(io.BytesIO(gz))
    assert detected_gz.source_format == "nifti"
    assert detected_gz.technical_metadata["gzip_wrapped"] is True


def test_upload_route_detects_png_even_with_misleading_dicom_name(
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
            patient_id="GBM-2026-DETECT",
            age_years=47,
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

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/studies/{study.id}/upload",
            files={
                "file": (
                    "looks-like-dicom.dcm",
                    make_png(),
                    "application/dicom",
                )
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["source_format"] == "image"
    assert body["modality"] == "UNKNOWN"
    assert body["parser"] == "pillow"
    assert body["technical_metadata"]["raster_format"] == "PNG"
    assert body["original_filename_stored"] is False
    assert body["phi_persisted_by_detection"] is False
    assert body["next_step"] == "mri_qc_and_capability_routing"

    db.refresh(study)
    assert study.source_format == SourceFormat.IMAGE
    assert study.status == StudyStatus.UPLOADED
    assert study.modality == "UNKNOWN"
    assert (
        study.deidentified_metadata["format_detection"]["phi_persisted"]
        is False
    )

    db.close()
    engine.dispose()
