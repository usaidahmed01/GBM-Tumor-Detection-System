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
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.capability_routing import (
    NiftiSequenceMappingError,
    confirm_brain_scope,
    confirm_nifti_sequence_mapping,
    route_study_capabilities,
)
from gbm_ai.api.services.clinical_records import (
    create_assessment,
    create_patient,
)


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


def make_study(
    session: Session,
    *,
    patient_id: str,
    age: int | None = 45,
    prior_treatment: bool = False,
):
    patient = create_patient(
        session,
        PatientCreate(
            patient_id=patient_id,
            age_years=age,
            privacy_flags={"synthetic": True},
        ),
    )
    assessment = create_assessment(
        session,
        AssessmentCreate(
            patient_uuid=patient.id,
            mri_date=date(2026, 8, 14),
            prior_treatment=prior_treatment,
        ),
    )
    return create_study(
        session,
        StudyCreate(assessment_uuid=assessment.id),
    )


def set_image_qc(study):
    study.source_format = SourceFormat.IMAGE
    study.modality = "UNKNOWN"
    study.qc_status = StudyQCStatus.PARTIAL
    study.qc_summary = {
        "qc_status": "partial",
        "partial_reasons": ["BRAIN_SCOPE_UNVERIFIED_FOR_RASTER"],
        "fail_reasons": [],
        "warnings": ["NO_PHYSICAL_SPATIAL_METADATA_FOR_RASTER"],
        "manual_review_required": True,
        "checks": {
            "brain_scope_status": "UNVERIFIED",
            "physical_spatial_metadata": False,
        },
        "inference_started": False,
        "capability_routing_completed": False,
    }
    study.status = StudyStatus.UPLOADED


def test_image_requires_scope_confirmation_before_classification(session):
    study = make_study(
        session,
        patient_id="GBM-ROUTE-IMAGE-1",
    )
    set_image_qc(study)
    session.commit()

    result = route_study_capabilities(session, study)

    assert result["routing_status"] == "review_required"
    assert (
        result["capabilities"]["two_d_classification"]["state"]
        == "review_required"
    )
    assert (
        "BRAIN_SCOPE_CONFIRMATION_REQUIRED"
        in result["capabilities"]["two_d_classification"]["reasons"]
    )
    assert study.status == StudyStatus.UPLOADED


def test_confirmed_raster_routes_only_to_current_2d_domain(session):
    study = make_study(
        session,
        patient_id="GBM-ROUTE-IMAGE-2",
    )
    set_image_qc(study)
    session.commit()

    confirm_brain_scope(
        session,
        study,
        is_brain_mri=True,
    )
    result = route_study_capabilities(session, study)

    assert result["routing_status"] == "ready"
    assert study.status == StudyStatus.READY_FOR_ANALYSIS
    assert (
        result["capabilities"]["two_d_classification"]["state"]
        == "eligible"
    )
    assert (
        result["capabilities"]["gradcam_2d"]["state"]
        == "deferred"
    )
    assert (
        result["capabilities"]["three_d_segmentation"]["state"]
        == "blocked"
    )
    assert (
        result["capabilities"]["physical_volume"]["state"]
        == "blocked"
    )
    assert (
        result["capabilities"]["physical_volume"]["user_message"]
        == "Physical tumor volume unavailable for this upload"
    )
    assert (
        result["capabilities"]["anatomical_localization"]["state"]
        == "blocked"
    )
    assert result["classifier_deployment_strategy_frozen"] is False
    assert result["model_execution_started"] is False


def test_prior_treatment_blocks_current_v1_analysis(session):
    study = make_study(
        session,
        patient_id="GBM-ROUTE-PRIOR",
        prior_treatment=True,
    )
    set_image_qc(study)
    study.brain_scope_status = BrainScopeStatus.CLINICIAN_CONFIRMED
    session.commit()

    result = route_study_capabilities(session, study)

    assert result["routing_status"] == "no_supported_analysis"
    assert "OUT_OF_SCOPE_PRIOR_TREATMENT" in result["global_block_reasons"]
    assert (
        result["capabilities"]["two_d_classification"]["state"]
        == "blocked"
    )
    assert study.status == StudyStatus.FAILED


def make_dicom_ready(study):
    study.source_format = SourceFormat.DICOM
    study.modality = "MR"
    study.deidentification_status = (
        DeidentificationStatus.METADATA_DEIDENTIFIED
    )
    study.qc_status = StudyQCStatus.PASS
    study.qc_summary = {
        "qc_status": "pass",
        "partial_reasons": [],
        "fail_reasons": [],
        "warnings": [
            "DICOM_PIXEL_PRIVACY_NOT_FORMALLY_VALIDATED",
            "SEQUENCE_DETECTION_HEURISTIC_NOT_CLINICALLY_VALIDATED",
        ],
        "manual_review_required": False,
        "checks": {
            "brain_scope_status": "SUPPORTED_BY_DICOM_HINT",
            "missing_segmentation_sequences": [],
        },
        "inference_started": False,
        "capability_routing_completed": False,
    }
    study.status = StudyStatus.UPLOADED


def add_series(session, study, labels):
    for number, label in enumerate(labels, start=1):
        session.add(
            Series(
                study_id=study.id,
                series_uid=f"2.25.{study.id.int % 100000}.{number}",
                series_number=number,
                detected_sequence=label,
                sequence_confidence=0.95,
                sequence_metadata={
                    "sequence_detection": {
                        "state": label,
                        "confidence": 0.95,
                        "clinically_validated": False,
                    }
                },
                slice_count=20,
                spacing_orientation_metadata={
                    "pixel_spacing": [1.0, 1.0],
                    "pixel_spacing_consistent": True,
                    "image_orientation_patient": [1, 0, 0, 0, 1, 0],
                    "orientation_consistent": True,
                },
                working_member_prefix=f"series_{number:03d}/",
            )
        )
    session.commit()


def test_complete_dicom_routes_to_3d_preprocessing_not_2d_bridge(session):
    study = make_study(
        session,
        patient_id="GBM-ROUTE-DICOM-OK",
    )
    make_dicom_ready(study)
    session.commit()
    add_series(session, study, ["T1", "T1C", "T2", "FLAIR"])

    result = route_study_capabilities(session, study)

    assert result["routing_status"] == "ready"
    assert (
        result["capabilities"]["two_d_classification"]["state"]
        == "blocked"
    )
    assert (
        "VOLUMETRIC_TO_2D_CLASSIFIER_BRIDGE_NOT_VALIDATED"
        in result["capabilities"]["two_d_classification"]["reasons"]
    )
    assert (
        result["capabilities"]["three_d_segmentation"]["state"]
        == "eligible"
    )
    assert (
        result["capabilities"]["physical_volume"]["state"]
        == "deferred"
    )
    assert (
        result["capabilities"]["anatomical_localization"]["state"]
        == "deferred"
    )
    assert study.brain_scope_status == BrainScopeStatus.SUPPORTED_BY_METADATA
    assert study.status == StudyStatus.READY_FOR_ANALYSIS


def test_truly_missing_dicom_t1c_blocks_3d_branch_without_fabrication(session):
    study = make_study(
        session,
        patient_id="GBM-ROUTE-DICOM-MISS",
    )
    make_dicom_ready(study)
    study.qc_status = StudyQCStatus.PARTIAL
    study.qc_summary["qc_status"] = "partial"
    study.qc_summary["partial_reasons"] = [
        "SEGMENTATION_SEQUENCE_MISSING_T1C"
    ]
    session.commit()
    add_series(session, study, ["T1", "T2", "FLAIR"])

    result = route_study_capabilities(session, study)

    assert result["routing_status"] == "no_supported_analysis"
    assert (
        result["capabilities"]["three_d_segmentation"]["state"]
        == "blocked"
    )
    assert (
        "SEGMENTATION_SEQUENCE_MISSING_T1C"
        in result["capabilities"]["three_d_segmentation"]["reasons"]
    )
    assert study.status == StudyStatus.UPLOADED


def test_ambiguous_dicom_series_keeps_missing_channel_in_review(session):
    study = make_study(
        session,
        patient_id="GBM-ROUTE-DICOM-AMB",
    )
    make_dicom_ready(study)
    study.qc_status = StudyQCStatus.PARTIAL
    study.qc_summary["qc_status"] = "partial"
    study.qc_summary["partial_reasons"] = [
        "DICOM_SEQUENCE_MAPPING_REQUIRES_CONFIRMATION",
        "SEGMENTATION_SEQUENCE_MISSING_T1C",
    ]
    session.commit()
    add_series(
        session,
        study,
        ["T1", "T2", "FLAIR", "NEEDS_CONFIRMATION"],
    )

    result = route_study_capabilities(session, study)

    assert result["routing_status"] == "review_required"
    assert (
        result["capabilities"]["three_d_segmentation"]["state"]
        == "review_required"
    )
    assert any(
        "DICOM_SEQUENCE_CONFIRMATION_MAY_RESOLVE_T1C" == reason
        for reason in result["capabilities"]["three_d_segmentation"]["reasons"]
    )


def make_nifti_qc(study):
    study.source_format = SourceFormat.NIFTI
    study.modality = "UNKNOWN"
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
    study.status = StudyStatus.UPLOADED


def test_nifti_requires_scope_and_sequence_mapping_confirmation(session):
    study = make_study(
        session,
        patient_id="GBM-ROUTE-NIFTI-REVIEW",
    )
    make_nifti_qc(study)
    session.commit()

    result = route_study_capabilities(session, study)

    assert result["routing_status"] == "review_required"
    assert (
        result["capabilities"]["three_d_segmentation"]["state"]
        == "review_required"
    )
    assert (
        "NIFTI_SEQUENCE_MAPPING_CONFIRMATION_REQUIRED"
        in result["capabilities"]["three_d_segmentation"]["reasons"]
    )


def test_invalid_nifti_mapping_reusing_volume_is_rejected(session):
    study = make_study(
        session,
        patient_id="GBM-ROUTE-NIFTI-BADMAP",
    )
    make_nifti_qc(study)
    session.commit()

    with pytest.raises(NiftiSequenceMappingError, match="distinct volume"):
        confirm_nifti_sequence_mapping(
            session,
            study,
            {
                "T1": 0,
                "T1C": 0,
                "T2": 2,
                "FLAIR": 3,
            },
        )


def test_confirmed_four_volume_nifti_becomes_3d_input_eligible(session):
    study = make_study(
        session,
        patient_id="GBM-ROUTE-NIFTI-OK",
    )
    make_nifti_qc(study)
    session.commit()

    confirm_brain_scope(
        session,
        study,
        is_brain_mri=True,
    )
    confirm_nifti_sequence_mapping(
        session,
        study,
        {
            "T1": 0,
            "T1C": 1,
            "T2": 2,
            "FLAIR": 3,
        },
    )

    result = route_study_capabilities(session, study)

    assert result["routing_status"] == "ready"
    assert (
        result["capabilities"]["three_d_segmentation"]["state"]
        == "eligible"
    )
    assert (
        result["capabilities"]["two_d_classification"]["state"]
        == "blocked"
    )
    assert (
        result["capabilities"]["physical_volume"]["state"]
        == "deferred"
    )
    assert study.status == StudyStatus.READY_FOR_ANALYSIS


def test_qc_failure_blocks_capabilities(session):
    study = make_study(
        session,
        patient_id="GBM-ROUTE-QC-FAIL",
    )
    study.source_format = SourceFormat.IMAGE
    study.qc_status = StudyQCStatus.FAIL
    study.qc_summary = {
        "qc_status": "fail",
        "partial_reasons": [],
        "fail_reasons": ["RASTER_BLANK_OR_NEAR_BLANK"],
        "warnings": [],
        "manual_review_required": False,
        "checks": {"brain_scope_status": "UNVERIFIED"},
    }
    study.brain_scope_status = BrainScopeStatus.CLINICIAN_CONFIRMED
    study.status = StudyStatus.FAILED
    session.commit()

    result = route_study_capabilities(session, study)

    assert result["routing_status"] == "no_supported_analysis"
    assert "MRI_QC_FAILED" in result["global_block_reasons"]
    assert (
        result["capabilities"]["two_d_classification"]["state"]
        == "blocked"
    )


def test_capability_routing_never_creates_analysis_run_or_starts_inference(
    session,
):
    study = make_study(
        session,
        patient_id="GBM-ROUTE-NO-INFERENCE",
    )
    set_image_qc(study)
    study.brain_scope_status = BrainScopeStatus.CLINICIAN_CONFIRMED
    session.commit()

    before = session.scalar(
        select(func.count()).select_from(AnalysisRun)
    )
    result = route_study_capabilities(session, study)
    after = session.scalar(
        select(func.count()).select_from(AnalysisRun)
    )

    assert before == 0
    assert after == 0
    assert result["model_execution_started"] is False
    assert result["classifier_deployment_strategy_frozen"] is False
    assert (
        result["volumetric_to_2d_classifier_bridge_validated"]
        is False
    )
