from __future__ import annotations

import io
import json
import zipfile
from datetime import date

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.dicom.deidentify import (
    DicomPixelPrivacyRiskError,
    build_deidentified_dicom_package,
    deidentify_dataset,
    safe_sequence_tokens,
    UIDMapper,
)
from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import (
    DeidentificationStatus,
    Series,
    SourceFormat,
    StudyStatus,
)
from gbm_ai.api.schemas.analysis import StudyCreate, StudyRead
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.services.dicom_processing import process_dicom_study
from gbm_ai.api.storage.local import LocalObjectStore


def test_sequence_tokenizer_keeps_only_safe_technical_vocabulary():
    tokens = safe_sequence_tokens(
        "Patient John Doe AX T2 FLAIR Post-Contrast 3D private-note-991"
    )

    assert "t2" in tokens
    assert "flair" in tokens
    assert "post" in tokens
    assert "contrast" in tokens
    assert "3d" in tokens
    assert "john" not in tokens
    assert "doe" not in tokens
    assert "991" not in tokens


def test_public_study_schema_does_not_expose_private_storage_keys():
    assert "storage_key" not in StudyRead.model_fields
    assert "deidentified_storage_key" not in StudyRead.model_fields
    assert "checksum_sha256" in StudyRead.model_fields
    assert "deidentified_checksum_sha256" in StudyRead.model_fields


def test_derived_storage_key_is_opaque_and_study_scoped(tmp_path):
    import uuid

    storage = LocalObjectStore(
        tmp_path / "storage",
        max_object_bytes=1024 * 1024,
    )
    study_id = uuid.uuid4()

    key = storage.generate_study_derived_key(
        study_id,
        "dicom-deidentified",
        suffix=".zip",
    )

    assert str(study_id) in key
    assert "/derived/dicom-deidentified/" in key
    assert key.endswith(".zip")
    assert "patient" not in key.lower()


def _pydicom():
    return pytest.importorskip("pydicom")


def _make_mr_dicom_bytes(
    *,
    study_uid: str,
    series_uid: str,
    instance_uid: str,
    series_number: int,
    instance_number: int,
    description: str,
    protocol: str,
    burned_in: str = "NO",
) -> bytes:
    pydicom = _pydicom()
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        None,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = instance_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.SeriesNumber = series_number
    ds.InstanceNumber = instance_number

    ds.PatientName = "John^Doe"
    ds.PatientID = "PATIENT-SECRET-123"
    ds.PatientBirthDate = "19700101"
    ds.StudyDate = "20260814"
    ds.InstitutionName = "Secret Hospital"
    ds.SeriesDescription = description
    ds.ProtocolName = protocol
    ds.BurnedInAnnotation = burned_in

    ds.Rows = 2
    ds.Columns = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, float(instance_number)]
    ds.ScanningSequence = "SE"
    ds.SequenceVariant = "SK"
    ds.MRAcquisitionType = "2D"
    ds.RepetitionTime = 9000.0 if "FLAIR" in description.upper() else 500.0
    ds.EchoTime = 120.0 if "FLAIR" in description.upper() else 10.0
    ds.InversionTime = 2500.0 if "FLAIR" in description.upper() else 0.0
    ds.FlipAngle = 90.0
    ds.PixelData = b"\x00\x00\x01\x00\x02\x00\x03\x00"

    # Synthetic private element that must disappear.
    ds.add_new((0x0011, 0x0010), "LO", "PRIVATE CREATOR")
    ds.add_new((0x0011, 0x1001), "LO", "PRIVATE SECRET")

    buffer = io.BytesIO()
    pydicom.dcmwrite(
        buffer,
        ds,
        enforce_file_format=True,
    )
    return buffer.getvalue()


def test_deidentify_dataset_removes_direct_phi_private_text_and_remaps_uids():
    pydicom = _pydicom()
    from pydicom.uid import generate_uid

    original_study = generate_uid()
    original_series = generate_uid()
    original_sop = generate_uid()

    raw = _make_mr_dicom_bytes(
        study_uid=original_study,
        series_uid=original_series,
        instance_uid=original_sop,
        series_number=1,
        instance_number=1,
        description="AX T2 FLAIR Patient John Doe",
        protocol="Brain FLAIR secret protocol",
    )
    ds = pydicom.dcmread(io.BytesIO(raw))

    original_pixels = bytes(ds.PixelData)
    original_sop_class = str(ds.SOPClassUID)

    working = deidentify_dataset(ds, UIDMapper())

    assert not any(tag.group == 0x0010 for tag in working.keys())
    assert not any(element.tag.is_private for element in working.iterall())

    assert "PatientName" not in working
    assert "PatientID" not in working
    assert "StudyDate" not in working
    assert "InstitutionName" not in working
    assert "SeriesDescription" not in working
    assert "ProtocolName" not in working

    assert str(working.StudyInstanceUID) != original_study
    assert str(working.SeriesInstanceUID) != original_series
    assert str(working.SOPInstanceUID) != original_sop
    assert str(working.SOPClassUID) == original_sop_class
    assert bytes(working.PixelData) == original_pixels
    assert (
        str(working.file_meta.MediaStorageSOPInstanceUID)
        == str(working.SOPInstanceUID)
    )


def test_burned_in_annotation_yes_blocks_ai_working_copy():
    pydicom = _pydicom()
    from pydicom.uid import generate_uid

    raw = _make_mr_dicom_bytes(
        study_uid=generate_uid(),
        series_uid=generate_uid(),
        instance_uid=generate_uid(),
        series_number=1,
        instance_number=1,
        description="T1",
        protocol="T1",
        burned_in="YES",
    )
    ds = pydicom.dcmread(io.BytesIO(raw))

    with pytest.raises(DicomPixelPrivacyRiskError):
        deidentify_dataset(ds, UIDMapper())


def test_dicom_zip_groups_series_and_does_not_persist_original_uids():
    pydicom = _pydicom()
    from pydicom.uid import generate_uid

    study_uid = generate_uid()
    flair_uid = generate_uid()
    t1_uid = generate_uid()

    original_sops = [generate_uid() for _ in range(4)]

    source = io.BytesIO()
    with zipfile.ZipFile(
        source,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.writestr(
            "slice_weird_name_1.bin",
            _make_mr_dicom_bytes(
                study_uid=study_uid,
                series_uid=flair_uid,
                instance_uid=original_sops[0],
                series_number=10,
                instance_number=1,
                description="AX T2 FLAIR John Doe",
                protocol="Brain FLAIR Patient Secret",
            ),
        )
        zf.writestr(
            "slice_weird_name_2.bin",
            _make_mr_dicom_bytes(
                study_uid=study_uid,
                series_uid=flair_uid,
                instance_uid=original_sops[1],
                series_number=10,
                instance_number=2,
                description="AX T2 FLAIR John Doe",
                protocol="Brain FLAIR Patient Secret",
            ),
        )
        zf.writestr(
            "another_1.dcm",
            _make_mr_dicom_bytes(
                study_uid=study_uid,
                series_uid=t1_uid,
                instance_uid=original_sops[2],
                series_number=20,
                instance_number=1,
                description="3D T1 PRE",
                protocol="MPRAGE T1",
            ),
        )
        zf.writestr(
            "another_2.dcm",
            _make_mr_dicom_bytes(
                study_uid=study_uid,
                series_uid=t1_uid,
                instance_uid=original_sops[3],
                series_number=20,
                instance_number=2,
                description="3D T1 PRE",
                protocol="MPRAGE T1",
            ),
        )
        zf.writestr("README.txt", b"ancillary non-DICOM file")

    source.seek(0)
    package = build_deidentified_dicom_package(source)

    try:
        assert package.input_instance_count == 4
        assert package.output_instance_count == 4
        assert package.ignored_non_dicom_entries == 1
        assert len(package.series_records) == 2
        assert package.study_uid != study_uid

        serialized_records = json.dumps(package.series_records)
        assert study_uid not in serialized_records
        assert flair_uid not in serialized_records
        assert t1_uid not in serialized_records
        assert "John" not in serialized_records
        assert "Secret" not in serialized_records

        package.output_stream.seek(0)
        with zipfile.ZipFile(package.output_stream, "r") as out_zip:
            names = sorted(out_zip.namelist())
            assert len(names) == 4
            assert all(name.startswith("series_") for name in names)

            with out_zip.open(names[0], "r") as member:
                ds = pydicom.dcmread(member)
                assert str(ds.StudyInstanceUID) != study_uid
                assert str(ds.SeriesInstanceUID) not in {flair_uid, t1_uid}
                assert str(ds.SOPInstanceUID) not in set(original_sops)
                assert "PatientName" not in ds
                assert "PatientID" not in ds
    finally:
        package.output_stream.close()


def test_processing_service_keeps_original_and_creates_separate_working_copy(
    tmp_path,
):
    _pydicom()
    from pydicom.uid import generate_uid

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
        max_object_bytes=20 * 1024 * 1024,
    )

    study_uid = generate_uid()
    series_uid = generate_uid()

    source_zip = io.BytesIO()
    with zipfile.ZipFile(
        source_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for i in range(1, 3):
            zf.writestr(
                f"input_{i}.dcm",
                _make_mr_dicom_bytes(
                    study_uid=study_uid,
                    series_uid=series_uid,
                    instance_uid=generate_uid(),
                    series_number=1,
                    instance_number=i,
                    description="T2 FLAIR",
                    protocol="AX FLAIR",
                ),
            )
    source_zip.seek(0)

    with SessionLocal() as db:
        patient = create_patient(
            db,
            PatientCreate(
                patient_id="GBM-2026-DICOM-DEID",
                age_years=50,
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

        source_key = storage.generate_study_source_key(study.id)
        stored_source = storage.put_stream(source_key, source_zip)

        study.storage_key = stored_source.storage_key
        study.checksum_sha256 = stored_source.sha256
        study.source_format = SourceFormat.DICOM
        study.modality = "MR"
        study.status = StudyStatus.UPLOADED
        db.commit()
        db.refresh(study)

        result = process_dicom_study(
            db,
            storage,
            study,
            request_id="test-request",
        )

        db.refresh(study)
        series = list(
            db.scalars(
                select(Series).where(Series.study_id == study.id)
            )
        )

        assert storage.exists(study.storage_key)
        assert storage.exists(study.deidentified_storage_key)
        assert study.storage_key != study.deidentified_storage_key
        assert study.checksum_sha256 == stored_source.sha256
        assert (
            study.deidentification_status
            == DeidentificationStatus.METADATA_DEIDENTIFIED
        )
        assert study.study_instance_uid != study_uid
        assert len(series) == 1
        assert result["series_count"] == 1
        assert result["instance_count"] == 2
        assert (
            study.deidentified_metadata["dicom_deidentification"][
                "ps3_15_profile_compliance_claimed"
            ]
            is False
        )
        assert (
            study.deidentified_metadata["dicom_deidentification"][
                "original_uids_persisted"
            ]
            is False
        )

    engine.dispose()
