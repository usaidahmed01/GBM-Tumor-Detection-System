from __future__ import annotations

import io
from datetime import date

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import (
    DeidentificationStatus,
    Series,
    SourceFormat,
    StudyQCStatus,
    StudyStatus,
)
from gbm_ai.api.models.audit import AuditAction, AuditLog
from gbm_ai.api.qc.sequence_detection import detect_series_sequence
from gbm_ai.api.qc.validators import qc_nifti_object, qc_raster_image
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.services.study_qc import (
    confirm_series_sequence,
    run_study_qc,
)
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


def _make_study(session, patient_id="GBM-2026-QC"):
    patient = create_patient(
        session,
        PatientCreate(
            patient_id=patient_id,
            age_years=49,
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
    return create_study(
        session,
        StudyCreate(assessment_uuid=assessment.id),
    )


def test_sequence_detector_high_confidence_flair():
    result = detect_series_sequence(
        {
            "series_description_tokens": ["t2", "flair", "axial"],
            "protocol_name_tokens": ["flair"],
            "repetition_time_ms": 9000,
            "echo_time_ms": 120,
            "inversion_time_ms": 2500,
            "contrast_metadata_present": False,
        }
    )

    assert result.state == "FLAIR"
    assert result.suggested_sequence == "FLAIR"
    assert result.confidence >= 0.8


def test_sequence_detector_high_confidence_t1_post_contrast():
    result = detect_series_sequence(
        {
            "series_description_tokens": ["t1", "post", "contrast"],
            "protocol_name_tokens": ["mprage"],
            "repetition_time_ms": 600,
            "echo_time_ms": 10,
            "contrast_metadata_present": True,
        }
    )

    assert result.state == "T1C"
    assert result.confidence >= 0.8


def test_sequence_detector_keeps_ambiguous_case_for_confirmation():
    result = detect_series_sequence(
        {
            "series_description_tokens": ["mprage"],
            "protocol_name_tokens": [],
            "contrast_metadata_present": False,
        }
    )

    assert result.state == "NEEDS_CONFIRMATION"
    assert result.suggested_sequence == "T1"
    assert 0.45 <= result.confidence < 0.8


def test_raster_qc_rejects_blank_and_marks_valid_image_partial():
    blank_buf = io.BytesIO()
    Image.new("L", (128, 128), 0).save(blank_buf, format="PNG")
    blank_buf.seek(0)

    blank = qc_raster_image(blank_buf)
    assert "RASTER_BLANK_OR_NEAR_BLANK" in blank.fail_reasons

    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 255, size=(128, 128), dtype=np.uint8)
    valid_buf = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(valid_buf, format="PNG")
    valid_buf.seek(0)

    valid = qc_raster_image(valid_buf)
    assert valid.fail_reasons == []
    assert "BRAIN_SCOPE_UNVERIFIED_FOR_RASTER" in valid.partial_reasons


def test_nifti_header_qc_when_nibabel_available():
    nib = pytest.importorskip("nibabel")

    image = nib.Nifti1Image(
        np.zeros((32, 32, 24), dtype=np.float32),
        np.diag([1.0, 1.0, 1.5, 1.0]),
    )
    raw = io.BytesIO(image.to_bytes())

    result = qc_nifti_object(raw)

    assert result.fail_reasons == []
    assert "NIFTI_SEQUENCE_MAPPING_REQUIRES_CONFIRMATION" in result.partial_reasons
    assert result.checks["volumes"][0]["affine_valid"] is True
    assert result.checks["volumes"][0]["spacing_valid"] is True


def _series_metadata(label: str) -> dict:
    base = {
        "body_part_hints": ["BRAIN_OR_HEAD"],
        "matrix_sizes": [[256, 256]],
        "scanning_sequence": ["SE"],
        "sequence_variant": [],
        "scan_options": [],
        "image_type": [],
        "mr_acquisition_type": ["2D"],
        "contrast_metadata_present": False,
    }

    if label == "T1":
        base.update(
            {
                "series_description_tokens": ["t1"],
                "protocol_name_tokens": ["t1"],
                "repetition_time_ms": 500,
                "echo_time_ms": 10,
            }
        )
    elif label == "T1C":
        base.update(
            {
                "series_description_tokens": ["t1", "post", "contrast"],
                "protocol_name_tokens": ["mprage"],
                "repetition_time_ms": 600,
                "echo_time_ms": 10,
                "contrast_metadata_present": True,
            }
        )
    elif label == "T2":
        base.update(
            {
                "series_description_tokens": ["t2"],
                "protocol_name_tokens": ["t2"],
                "repetition_time_ms": 4000,
                "echo_time_ms": 100,
            }
        )
    elif label == "FLAIR":
        base.update(
            {
                "series_description_tokens": ["t2", "flair"],
                "protocol_name_tokens": ["flair"],
                "repetition_time_ms": 9000,
                "echo_time_ms": 120,
                "inversion_time_ms": 2500,
            }
        )
    return base


def _geometry() -> dict:
    return {
        "pixel_spacing": [1.0, 1.0],
        "pixel_spacing_consistent": True,
        "image_orientation_patient": [1, 0, 0, 0, 1, 0],
        "orientation_consistent": True,
        "slice_thickness": 1.0,
        "spacing_between_slices": 1.0,
        "image_position_available_count": 20,
    }


def test_dicom_qc_can_pass_engineering_preflight_with_four_sequences(
    session,
    tmp_path,
    monkeypatch,
):
    study = _make_study(session, "GBM-2026-QC-DICOM")
    storage = LocalObjectStore(
        tmp_path / "storage",
        max_object_bytes=1024 * 1024,
    )

    source = storage.put_stream(
        storage.generate_study_source_key(study.id),
        io.BytesIO(b"protected-original"),
    )
    working = storage.put_stream(
        storage.generate_study_derived_key(
            study.id,
            "dicom-deidentified",
            suffix=".zip",
        ),
        io.BytesIO(b"synthetic-working-object"),
    )

    study.storage_key = source.storage_key
    study.checksum_sha256 = source.sha256
    study.deidentified_storage_key = working.storage_key
    study.deidentified_checksum_sha256 = working.sha256
    study.source_format = SourceFormat.DICOM
    study.modality = "MR"
    study.deidentification_status = DeidentificationStatus.METADATA_DEIDENTIFIED
    study.status = StudyStatus.UPLOADED

    for number, label in enumerate(("T1", "T1C", "T2", "FLAIR"), start=1):
        session.add(
            Series(
                study_id=study.id,
                series_uid=f"2.25.{number}",
                series_number=number,
                sequence_metadata=_series_metadata(label),
                slice_count=20,
                spacing_orientation_metadata=_geometry(),
                working_member_prefix=f"series_{number:03d}/",
            )
        )
    session.commit()
    session.refresh(study)

    monkeypatch.setattr(
        "gbm_ai.api.services.study_qc.sample_dicom_pixel_quality",
        lambda source: {
            "decoded_sample_count": 4,
            "decode_error_count": 0,
            "blank_sample_count": 0,
            "low_resolution_sample_count": 0,
            "samples": [],
        },
    )

    summary = run_study_qc(
        session,
        storage,
        study,
        request_id="qc-test",
    )

    assert summary["qc_status"] == "pass"
    assert summary["fail_reasons"] == []
    assert summary["partial_reasons"] == []
    assert summary["checks"]["missing_segmentation_sequences"] == []
    assert summary["inference_started"] is False

    detected = {
        item.detected_sequence
        for item in session.scalars(
            select(Series).where(Series.study_id == study.id)
        )
    }
    assert detected == {"T1", "T1C", "T2", "FLAIR"}

    audit = session.scalar(
        select(AuditLog).where(
            AuditLog.action == AuditAction.STUDY_QC_COMPLETED
        )
    )
    assert audit is not None
    assert audit.technical_context["qc_status"] == "pass"


def test_missing_dicom_sequence_is_partial_not_false_failure(
    session,
    tmp_path,
    monkeypatch,
):
    study = _make_study(session, "GBM-2026-QC-MISSING")
    storage = LocalObjectStore(
        tmp_path / "storage",
        max_object_bytes=1024 * 1024,
    )

    source = storage.put_stream(
        storage.generate_study_source_key(study.id),
        io.BytesIO(b"protected-original"),
    )
    working = storage.put_stream(
        storage.generate_study_derived_key(
            study.id,
            "dicom-deidentified",
            suffix=".zip",
        ),
        io.BytesIO(b"working"),
    )

    study.storage_key = source.storage_key
    study.deidentified_storage_key = working.storage_key
    study.source_format = SourceFormat.DICOM
    study.modality = "MR"
    study.deidentification_status = DeidentificationStatus.METADATA_DEIDENTIFIED
    study.status = StudyStatus.UPLOADED

    for number, label in enumerate(("T1", "T2", "FLAIR"), start=1):
        session.add(
            Series(
                study_id=study.id,
                series_uid=f"2.26.{number}",
                series_number=number,
                sequence_metadata=_series_metadata(label),
                slice_count=20,
                spacing_orientation_metadata=_geometry(),
                working_member_prefix=f"series_{number:03d}/",
            )
        )
    session.commit()
    session.refresh(study)

    monkeypatch.setattr(
        "gbm_ai.api.services.study_qc.sample_dicom_pixel_quality",
        lambda source: {
            "decoded_sample_count": 3,
            "decode_error_count": 0,
            "blank_sample_count": 0,
            "low_resolution_sample_count": 0,
            "samples": [],
        },
    )

    summary = run_study_qc(session, storage, study)

    assert summary["qc_status"] == "partial"
    assert "SEGMENTATION_SEQUENCE_MISSING_T1C" in summary["partial_reasons"]
    assert summary["fail_reasons"] == []
    assert study.status == StudyStatus.UPLOADED


def test_sequence_confirmation_invalidates_previous_qc(
    session,
):
    study = _make_study(session, "GBM-2026-QC-CONFIRM")
    series = Series(
        study_id=study.id,
        series_uid="2.25.999",
        series_number=1,
        detected_sequence="NEEDS_CONFIRMATION",
        sequence_confidence=0.68,
        sequence_metadata={
            "sequence_detection": {
                "state": "NEEDS_CONFIRMATION",
                "suggested_sequence": "T1",
            }
        },
        slice_count=20,
        spacing_orientation_metadata=_geometry(),
        working_member_prefix="series_001/",
    )
    session.add(series)

    study.qc_status = StudyQCStatus.PARTIAL
    study.qc_summary = {
        "qc_status": "partial",
        "stale": False,
        "capability_routing_completed": False,
    }
    session.commit()

    updated = confirm_series_sequence(
        session,
        series.id,
        "T1",
        request_id="confirm-test",
    )

    session.refresh(study)
    assert updated.confirmed_sequence == "T1"
    assert study.qc_status == StudyQCStatus.PENDING
    assert study.qc_summary["stale"] is True
    assert (
        study.qc_summary["stale_reason"]
        == "series_sequence_confirmation_changed"
    )

    audit = session.scalar(
        select(AuditLog).where(
            AuditLog.action == AuditAction.SERIES_SEQ_CONFIRMED
        )
    )
    assert audit is not None
    assert audit.technical_context["sequence_label"] == "T1"
