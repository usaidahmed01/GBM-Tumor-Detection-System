from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import (
    AnalysisRun,
    BrainScopeStatus,
    SegmentationPreparationStatus,
    SourceFormat,
    StudyQCStatus,
    StudyStatus,
)
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.segmentation.volume_loading import (
    LoadedSegmentationVolume,
    SegmentationVolumeLoadError,
    summarize_loaded_volume,
    validate_channel_alignment,
)
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.capability_routing import route_study_capabilities
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.services.segmentation_state import invalidate_segmentation_preparation
from gbm_ai.api.services.segmentation_volume_preparation import (
    prepare_segmentation_volumes,
)


class IntegrityOnlyStore:
    def verify_checksum(self, storage_key: str, expected_sha256: str) -> bool:
        return True


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


def make_ready_nifti_study(session: Session, patient_id: str):
    patient = create_patient(
        session,
        PatientCreate(
            patient_id=patient_id,
            age_years=45,
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

    study.source_format = SourceFormat.NIFTI
    study.modality = "UNKNOWN"
    study.storage_key = f"studies/{study.id}/source/test.bin"
    study.checksum_sha256 = "b" * 64
    study.qc_status = StudyQCStatus.PARTIAL
    study.qc_summary = {
        "qc_status": "partial",
        "partial_reasons": [
            "NIFTI_SEQUENCE_MAPPING_REQUIRES_CONFIRMATION",
            "BRAIN_SCOPE_UNVERIFIED_FOR_NIFTI",
        ],
        "fail_reasons": [],
        "warnings": ["NIFTI_VOXEL_QUALITY_NOT_FULLY_SAMPLED"],
        "manual_review_required": True,
        "checks": {
            "brain_scope_status": "UNVERIFIED",
            "sequence_mapping_status": "REQUIRES_CONFIRMATION",
            "volume_count": 4,
            "volumes": [
                {
                    "volume_index": index,
                    "shape": [16, 16, 16],
                    "zooms": [1.0, 1.0, 1.0],
                    "is_3d": True,
                    "is_4d": False,
                    "shape_valid": True,
                    "spatial_size_sufficient": True,
                    "spacing_valid": True,
                    "affine_valid": True,
                }
                for index in range(4)
            ],
        },
        "inference_started": False,
        "capability_routing_completed": False,
    }
    study.brain_scope_status = BrainScopeStatus.CLINICIAN_CONFIRMED
    study.nifti_sequence_mapping = {
        "T1": 0,
        "T1C": 1,
        "T2": 2,
        "FLAIR": 3,
    }
    study.status = StudyStatus.UPLOADED
    session.commit()
    route_study_capabilities(session, study)
    return study


def fake_volume(sequence: str, *, translate_x: float = 0.0):
    affine = np.eye(4, dtype=np.float64)
    affine[0, 3] = translate_x
    data = np.arange(16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16)
    return LoadedSegmentationVolume(
        sequence=sequence,
        source_kind="nifti_volume",
        source_reference="synthetic",
        data=data,
        affine_ras=affine,
        orientation_codes=("R", "A", "S"),
    )


def test_alignment_accepts_equal_canonical_geometry():
    summaries = [
        summarize_loaded_volume(fake_volume(sequence))
        for sequence in ("T1C", "T1", "T2", "FLAIR")
    ]
    result = validate_channel_alignment(summaries)
    assert result["aligned"] is True
    assert result["reasons"] == []


def test_alignment_requires_registration_when_affine_differs():
    summaries = [
        summarize_loaded_volume(fake_volume("T1C")),
        summarize_loaded_volume(fake_volume("T1")),
        summarize_loaded_volume(fake_volume("T2", translate_x=2.0)),
        summarize_loaded_volume(fake_volume("FLAIR")),
    ]
    result = validate_channel_alignment(summaries)
    assert result["aligned"] is False
    assert "T2_AFFINE_DIFFERS_FROM_T1C" in result["reasons"]


def test_prepare_ready_persists_geometry_and_never_creates_analysis_run(
    session,
    monkeypatch,
):
    study = make_ready_nifti_study(session, "GBM-P6-S2-READY")

    def loader(db, storage, loaded_study, channel_plan):
        return fake_volume(channel_plan["sequence"])

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_volume_preparation.load_channel_volume",
        loader,
    )

    before = session.scalar(select(func.count()).select_from(AnalysisRun))
    result = prepare_segmentation_volumes(
        session,
        IntegrityOnlyStore(),
        study,
    )
    after = session.scalar(select(func.count()).select_from(AnalysisRun))

    assert result["status"] == "ready"
    assert result["alignment"]["aligned"] is True
    assert result["channel_order"] == ["T1C", "T1", "T2", "FLAIR"]
    assert result["model_execution_started"] is False
    assert result["registration_performed"] is False
    assert result["reference_geometry_resampling_performed"] is False
    assert study.segmentation_preparation_status == SegmentationPreparationStatus.READY
    assert before == 0
    assert after == 0


def test_prepare_returns_registration_required_without_registering(
    session,
    monkeypatch,
):
    study = make_ready_nifti_study(session, "GBM-P6-S2-REG")

    def loader(db, storage, loaded_study, channel_plan):
        shift = 3.0 if channel_plan["sequence"] == "FLAIR" else 0.0
        return fake_volume(channel_plan["sequence"], translate_x=shift)

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_volume_preparation.load_channel_volume",
        loader,
    )

    result = prepare_segmentation_volumes(
        session,
        IntegrityOnlyStore(),
        study,
    )

    assert result["status"] == "registration_required"
    assert result["alignment"]["aligned"] is False
    assert result["registration_performed"] is False
    assert result["next_step"] == (
        "phase6_step3_registration_and_model_geometry_preprocessing"
    )
    assert study.segmentation_preparation_status == (
        SegmentationPreparationStatus.REGISTRATION_REQUIRED
    )


def test_loader_failure_is_persisted_as_safe_preparation_failure(
    session,
    monkeypatch,
):
    study = make_ready_nifti_study(session, "GBM-P6-S2-FAIL")

    def loader(db, storage, loaded_study, channel_plan):
        raise SegmentationVolumeLoadError(
            "SYNTHETIC_LOAD_FAILURE",
            "synthetic non-PHI failure",
        )

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_volume_preparation.load_channel_volume",
        loader,
    )

    with pytest.raises(ValueError, match="synthetic non-PHI failure"):
        prepare_segmentation_volumes(
            session,
            IntegrityOnlyStore(),
            study,
        )

    assert study.segmentation_preparation_status == SegmentationPreparationStatus.FAILED
    assert study.segmentation_preparation_summary["failure_reason_code"] == (
        "SYNTHETIC_LOAD_FAILURE"
    )
    assert study.segmentation_preparation_summary["model_execution_started"] is False


def test_upstream_change_invalidates_preparation_state(session):
    study = make_ready_nifti_study(session, "GBM-P6-S2-STALE")
    study.segmentation_preparation_status = SegmentationPreparationStatus.READY
    study.segmentation_preparation_summary = {"status": "ready"}

    invalidate_segmentation_preparation(study)

    assert study.segmentation_preparation_status == SegmentationPreparationStatus.PENDING
    assert study.segmentation_preparation_summary == {}


def test_real_nifti_bytes_load_and_canonicalize_when_dependency_available():
    nib = pytest.importorskip("nibabel")
    import io

    from gbm_ai.api.segmentation.volume_loading import (
        _canonicalize_image,
        _load_nifti_image_from_raw,
    )

    data = np.arange(12 * 13 * 14, dtype=np.float32).reshape(12, 13, 14)
    image = nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.0, 1.0]))

    buffer = io.BytesIO()
    file_map = image.make_file_map()
    file_map["image"].fileobj = buffer
    image.to_file_map(file_map)
    raw = buffer.getvalue()

    reloaded = _load_nifti_image_from_raw(raw)
    volume = _canonicalize_image(
        "T1C",
        reloaded,
        source_kind="nifti_volume",
        source_reference="synthetic",
    )

    assert volume.shape == (12, 13, 14)
    assert volume.orientation_codes == ("R", "A", "S")
    assert np.allclose(volume.spacing_mm, (1.0, 1.0, 1.0))


def test_real_dicom_stack_reconstructs_when_dependencies_available():
    pydicom = pytest.importorskip("pydicom")
    pytest.importorskip("nibabel")

    import io
    import uuid
    import zipfile

    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    from gbm_ai.api.models.analysis import Series
    from gbm_ai.api.segmentation.volume_loading import _load_dicom_series_image

    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(8):
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
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelSpacing = [1.0, 1.0]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [0.0, 0.0, float(index)]
            pixels = np.full((16, 16), index + 1, dtype=np.uint16)
            ds.PixelData = pixels.tobytes()

            dicom_bytes = io.BytesIO()
            pydicom.dcmwrite(dicom_bytes, ds, enforce_file_format=True)
            archive.writestr(
                f"series_001/instance_{index + 1:06d}.dcm",
                dicom_bytes.getvalue(),
            )

    archive_bytes.seek(0)
    series = Series(
        id=uuid.uuid4(),
        study_id=uuid.uuid4(),
        series_uid="2.25.1",
        slice_count=8,
        sequence_metadata={},
        spacing_orientation_metadata={},
        working_member_prefix="series_001/",
    )

    image = _load_dicom_series_image(archive_bytes, series, "T1C")

    assert image.shape == (16, 16, 8)
    spacing = np.linalg.norm(np.asarray(image.affine)[:3, :3], axis=0)
    assert np.allclose(spacing, (1.0, 1.0, 1.0))
