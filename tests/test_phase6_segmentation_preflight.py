from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import (
    AnalysisRun,
    BrainScopeStatus,
    CapabilityRoutingStatus,
    DeidentificationStatus,
    Series,
    SourceFormat,
    StudyQCStatus,
    StudyStatus,
)
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import (
    AssessmentCreate,
    PatientCreate,
)
from gbm_ai.api.segmentation.contract import (
    SEGMENTATION_INPUT_CHANNEL_ORDER,
    SEGMENTATION_MODEL_CONTRACT,
    SEGMENTATION_OUTPUT_CHANNEL_ORDER,
)
from gbm_ai.api.services.analysis_records import (
    create_study,
)
from gbm_ai.api.services.capability_routing import (
    route_study_capabilities,
)
from gbm_ai.api.services.clinical_records import (
    create_assessment,
    create_patient,
)
from gbm_ai.api.services.segmentation_preflight import (
    SegmentationPreflightError,
    build_segmentation_preflight,
)


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
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


def make_study(
    session: Session,
    patient_id: str,
):
    patient = create_patient(
        session,
        PatientCreate(
            patient_id=patient_id,
            age_years=45,
            privacy_flags={
                "synthetic": True,
            },
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
        StudyCreate(
            assessment_uuid=assessment.id,
        ),
    )


def make_dicom_ready(
    session: Session,
    patient_id: str,
):
    study = make_study(
        session,
        patient_id,
    )

    study.source_format = SourceFormat.DICOM
    study.modality = "MR"

    study.deidentification_status = (
        DeidentificationStatus.METADATA_DEIDENTIFIED
    )

    study.deidentified_storage_key = (
        f"studies/{study.id}/derived/dicom/test.zip"
    )

    study.deidentified_checksum_sha256 = "a" * 64

    study.qc_status = StudyQCStatus.PASS

    study.qc_summary = {
        "qc_status": "pass",
        "partial_reasons": [],
        "fail_reasons": [],
        "warnings": [
            "DICOM_PIXEL_PRIVACY_NOT_FORMALLY_VALIDATED"
        ],
        "manual_review_required": False,
        "checks": {
            "brain_scope_status": (
                "SUPPORTED_BY_DICOM_HINT"
            ),
            "missing_segmentation_sequences": [],
        },
        "inference_started": False,
        "capability_routing_completed": False,
    }

    study.status = StudyStatus.UPLOADED
    session.commit()

    for number, label in enumerate(
        ["T1", "T1C", "T2", "FLAIR"],
        start=1,
    ):
        session.add(
            Series(
                study_id=study.id,
                series_uid=(
                    f"2.25."
                    f"{study.id.int % 100000}."
                    f"{number}"
                ),
                series_number=number,
                detected_sequence=label,
                sequence_confidence=0.95,
                sequence_metadata={},
                slice_count=20,
                spacing_orientation_metadata={
                    "pixel_spacing": [1.0, 1.0],
                    "pixel_spacing_consistent": True,
                    "image_orientation_patient": [
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                    ],
                    "orientation_consistent": True,
                },
                working_member_prefix=(
                    f"series_{number:03d}/"
                ),
            )
        )

    session.commit()

    route_study_capabilities(
        session,
        study,
    )

    return study


def make_nifti_ready(
    session: Session,
    patient_id: str,
):
    study = make_study(
        session,
        patient_id,
    )

    study.source_format = SourceFormat.NIFTI
    study.modality = "UNKNOWN"

    study.storage_key = (
        f"studies/{study.id}/source/test.bin"
    )
    study.checksum_sha256 = "b" * 64

    study.qc_status = StudyQCStatus.PARTIAL

    study.qc_summary = {
        "qc_status": "partial",
        "partial_reasons": [
            "NIFTI_SEQUENCE_MAPPING_REQUIRES_CONFIRMATION",
            "BRAIN_SCOPE_UNVERIFIED_FOR_NIFTI",
        ],
        "fail_reasons": [],
        "warnings": [
            "NIFTI_VOXEL_QUALITY_NOT_FULLY_SAMPLED"
        ],
        "manual_review_required": True,
        "checks": {
            "brain_scope_status": "UNVERIFIED",
            "sequence_mapping_status": (
                "REQUIRES_CONFIRMATION"
            ),
            "volume_count": 4,
            "volumes": [
                {
                    "volume_index": index,
                    "shape": [128, 128, 128],
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

    study.brain_scope_status = (
        BrainScopeStatus.CLINICIAN_CONFIRMED
    )

    study.nifti_sequence_mapping = {
        "T1": 0,
        "T1C": 1,
        "T2": 2,
        "FLAIR": 3,
    }

    study.status = StudyStatus.UPLOADED
    session.commit()

    route_study_capabilities(
        session,
        study,
    )

    return study


def test_segmentation_contract_freezes_model_channel_order_and_scope():
    assert SEGMENTATION_INPUT_CHANNEL_ORDER == (
        "T1C",
        "T1",
        "T2",
        "FLAIR",
    )

    assert SEGMENTATION_OUTPUT_CHANNEL_ORDER == (
        "TC",
        "WT",
        "ET",
    )

    assert (
        SEGMENTATION_MODEL_CONTRACT.reference_spacing_mm
        == (1.0, 1.0, 1.0)
    )

    assert (
        SEGMENTATION_MODEL_CONTRACT
        .runtime_model_loading_implemented
        is False
    )

    assert (
        SEGMENTATION_MODEL_CONTRACT
        .inference_implemented
        is False
    )

    assert (
        SEGMENTATION_MODEL_CONTRACT
        .clinical_validation_claimed
        is False
    )


def test_eligible_dicom_builds_model_order_without_starting_inference(
    session,
):
    study = make_dicom_ready(
        session,
        "GBM-P6-DICOM",
    )

    result = build_segmentation_preflight(
        session,
        study,
    )

    assert result["status"] == (
        "ready_for_preprocessing"
    )

    assert [
        item["sequence"]
        for item in result["channels"]
    ] == [
        "T1C",
        "T1",
        "T2",
        "FLAIR",
    ]

    assert [
        item["channel_index"]
        for item in result["channels"]
    ] == [0, 1, 2, 3]

    assert all(
        item["source_kind"] == "dicom_series"
        for item in result["channels"]
    )

    assert result["model_execution_started"] is False
    assert result["segmentation_generated"] is False


def test_eligible_nifti_uses_confirmed_opaque_volume_indices_in_model_order(
    session,
):
    study = make_nifti_ready(
        session,
        "GBM-P6-NIFTI",
    )

    result = build_segmentation_preflight(
        session,
        study,
    )

    assert [
        item["sequence"]
        for item in result["channels"]
    ] == [
        "T1C",
        "T1",
        "T2",
        "FLAIR",
    ]

    assert [
        item["volume_index"]
        for item in result["channels"]
    ] == [
        1,
        0,
        2,
        3,
    ]

    assert all(
        item["source_kind"] == "nifti_volume"
        for item in result["channels"]
    )

    assert result["physical_volume_generated"] is False
    assert (
        result["anatomical_localization_generated"]
        is False
    )


def test_preflight_rejects_stale_phase5_routing(
    session,
):
    study = make_nifti_ready(
        session,
        "GBM-P6-STALE",
    )

    study.capability_routing_status = (
        CapabilityRoutingStatus.PENDING
    )
    study.capability_summary = {
        "stale": True,
    }
    study.status = StudyStatus.UPLOADED

    session.commit()

    with pytest.raises(
        SegmentationPreflightError,
        match="not ready for analysis",
    ):
        build_segmentation_preflight(
            session,
            study,
        )


def test_preflight_rejects_standalone_image_even_if_2d_branch_is_ready(
    session,
):
    study = make_study(
        session,
        "GBM-P6-IMAGE",
    )

    study.source_format = SourceFormat.IMAGE
    study.qc_status = StudyQCStatus.PARTIAL

    study.qc_summary = {
        "qc_status": "partial",
        "partial_reasons": [
            "BRAIN_SCOPE_UNVERIFIED_FOR_RASTER"
        ],
        "fail_reasons": [],
        "warnings": [],
        "checks": {
            "brain_scope_status": "UNVERIFIED",
        },
    }

    study.brain_scope_status = (
        BrainScopeStatus.CLINICIAN_CONFIRMED
    )

    study.status = StudyStatus.UPLOADED

    session.commit()

    route_study_capabilities(
        session,
        study,
    )

    with pytest.raises(
        SegmentationPreflightError,
        match="only DICOM or NIfTI",
    ):
        build_segmentation_preflight(
            session,
            study,
        )


def test_preflight_never_creates_analysis_run(
    session,
):
    study = make_dicom_ready(
        session,
        "GBM-P6-NORUN",
    )

    before = session.scalar(
        select(func.count()).select_from(
            AnalysisRun
        )
    )

    result = build_segmentation_preflight(
        session,
        study,
    )

    after = session.scalar(
        select(func.count()).select_from(
            AnalysisRun
        )
    )

    assert before == 0
    assert after == 0

    assert result["model_execution_started"] is False