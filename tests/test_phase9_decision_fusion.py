from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import (
    AnalysisRun,
    AnalysisStatus,
    CapabilityRoutingStatus,
    DecisionState,
    ModelRole,
    ModelVersion,
    QCState,
    SourceFormat,
    StudyQCStatus,
    StudyStatus,
)
from gbm_ai.api.models.audit import AuditAction, AuditLog
from gbm_ai.api.models.segmentation import (
    Segmentation,
    SegmentationReviewStatus,
    SegmentationStatus,
)
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.services.decision_fusion import (
    DECISION_FUSION_VERSION,
    fuse_study_decision,
    get_current_fused_decision,
)


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    with SessionLocal() as db:
        yield db
    engine.dispose()


def make_study(session: Session, source_format: SourceFormat):
    patient = create_patient(
        session,
        PatientCreate(
            patient_id=f"P9-{source_format.value}",
            age_years=49,
            privacy_flags={"synthetic": True},
        ),
    )
    assessment = create_assessment(
        session,
        AssessmentCreate(
            patient_uuid=patient.id,
            mri_date=date(2026, 8, 16),
            prior_treatment=False,
        ),
    )
    study = create_study(session, StudyCreate(assessment_uuid=assessment.id))
    study.source_format = source_format
    study.status = StudyStatus.READY_FOR_ANALYSIS
    study.qc_status = StudyQCStatus.PASS
    study.capability_routing_status = CapabilityRoutingStatus.READY
    study.capability_summary = {
        "global_hard_block_reasons": [],
        "capabilities": {
            "two_d_classification": {
                "state": "eligible" if source_format == SourceFormat.IMAGE else "blocked"
            },
            "three_d_segmentation": {
                "state": "eligible" if source_format != SourceFormat.IMAGE else "blocked"
            },
        },
        "classifier_deployment_strategy_frozen": False,
        "volumetric_to_2d_classifier_bridge_validated": False,
        "clinical_validation_claimed": False,
    }
    session.commit()
    return study


def freeze_classifier_deployment_for_test(session: Session, study) -> None:
    study.capability_summary = {
        **dict(study.capability_summary or {}),
        "classifier_deployment_strategy_frozen": True,
    }
    session.commit()


def add_classifier_run(
    session: Session,
    study,
    *,
    probability: float,
    state: DecisionState,
    qc_state: QCState = QCState.PASS,
    ood: bool = False,
    model_version: str = "phase9_step2_classifier_deployment_v1",
):
    model = ModelVersion(
        model_name="efficientnet_v2_s",
        version=model_version,
        role=ModelRole.CLASSIFIER,
        architecture="efficientnet_v2_s",
        weights_checksum_sha256="a" * 64,
        code_version="test",
        preprocessing_version="classification_v1",
        threshold_version="classifier_safety_policy_v1",
        calibration_version="cross_fitted_temperature_v1",
        license_source_notes="synthetic test fixture",
        is_active=True,
    )
    session.add(model)
    session.flush()
    run = AnalysisRun(
        study_id=study.id,
        classifier_model_version_id=model.id,
        status=AnalysisStatus.COMPLETE,
        qc_state=qc_state,
        ood_likeness_candidate=ood,
        calibrated_probability_gbm=probability,
        decision_state=state,
        safety_reason_codes=[],
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    session.add(run)
    session.commit()
    return run


def add_reviewed_segmentation(
    session: Session,
    study,
    *,
    wt_voxels: int = 8,
    review_status: SegmentationReviewStatus = SegmentationReviewStatus.ACCEPTED,
):
    model = ModelVersion(
        model_name="brats_mri_segmentation",
        version="0.5.4-test",
        role=ModelRole.SEGMENTATION,
        architecture="SegResNet",
        weights_checksum_sha256="b" * 64,
        code_version="test",
        preprocessing_version="phase6_step4_monai_model_input_v1",
        threshold_version=None,
        calibration_version=None,
        license_source_notes="synthetic test fixture",
        is_active=True,
    )
    session.add(model)
    session.flush()
    analysis = AnalysisRun(
        study_id=study.id,
        segmentation_model_version_id=model.id,
        status=AnalysisStatus.COMPLETE,
        qc_state=QCState.PASS,
        decision_state=DecisionState.PENDING,
        safety_reason_codes=[],
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    session.add(analysis)
    session.flush()
    segmentation = Segmentation(
        analysis_run_id=analysis.id,
        status=SegmentationStatus.GENERATED,
        model_input_checksum_sha256="c" * 64,
        inference_version="phase6_step5_guarded_segmentation_inference_v1",
        preprocessing_version="phase6_step4_monai_model_input_v1",
        bundle_name="brats_mri_segmentation",
        bundle_version="0.5.4",
        weights_checksum_sha256="b" * 64,
        device="cpu",
        amp_enabled=False,
        roi_size=[240, 240, 160],
        overlap=0.5,
        threshold=0.5,
        spatial_shape=[4, 4, 4],
        affine_ras=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        tc_storage_key="derived/tc.nii.gz",
        tc_checksum_sha256="d" * 64,
        tc_size_bytes=10,
        wt_storage_key="derived/wt.nii.gz",
        wt_checksum_sha256="e" * 64,
        wt_size_bytes=10,
        et_storage_key="derived/et.nii.gz",
        et_checksum_sha256="f" * 64,
        et_size_bytes=10,
        labelmap_storage_key="derived/labelmap.nii.gz",
        labelmap_checksum_sha256="1" * 64,
        labelmap_size_bytes=10,
        voxel_counts={"WT": wt_voxels, "TC": max(wt_voxels // 2, 0), "ET": 0},
        runtime_seconds=0.1,
        review_status=review_status,
        clinician_modified=False,
        physical_volume_generated=False,
        anatomical_localization_generated=False,
        clinical_validation_claimed=False,
    )
    session.add(segmentation)
    session.flush()
    study.segmentation_preparation_summary = {
        "inference": {
            "status": "complete",
            "segmentation_uuid": str(segmentation.id),
            "segmentation_generated": True,
        },
        "segmentation_generated": True,
    }
    session.commit()
    return segmentation


def test_volumetric_study_never_gets_gbm_class_from_segmentation_without_validated_bridge(session):
    study = make_study(session, SourceFormat.NIFTI)
    add_reviewed_segmentation(session, study, wt_voxels=12)

    result = fuse_study_decision(session, study)

    assert result["version"] == DECISION_FUSION_VERSION
    assert result["decision_state"] == DecisionState.INDETERMINATE
    assert result["classifier"]["available"] is False
    assert result["classifier"]["blocked_reason"] == "VOLUMETRIC_TO_2D_CLASSIFIER_BRIDGE_NOT_VALIDATED"
    assert result["segmentation"]["lesion_evidence_present"] is True
    assert "SEGMENTATION_IS_NOT_GBM_CLASSIFIER" in result["safety_reason_codes"]
    assert result["segmentation_is_gbm_diagnosis"] is False
    assert result["clinical_validation_claimed"] is False
    assert result["report_ready"] is True


def test_2d_classifier_evidence_must_match_the_frozen_deployment_model(session):
    study = make_study(session, SourceFormat.IMAGE)
    add_classifier_run(
        session,
        study,
        probability=0.86,
        state=DecisionState.GBM_SUSPECTED,
        model_version="legacy-unfrozen-classifier",
    )

    result = fuse_study_decision(session, study)

    assert result["decision_state"] == DecisionState.INDETERMINATE
    assert result["classifier"]["available"] is False
    assert result["classifier"]["blocked_reason"] == "CLASSIFIER_MODEL_NOT_FROZEN_DEPLOYMENT"


def test_image_classifier_low_state_preserves_other_abnormality_warning(session):
    study = make_study(session, SourceFormat.IMAGE)
    freeze_classifier_deployment_for_test(session, study)
    add_classifier_run(
        session,
        study,
        probability=0.08,
        state=DecisionState.GBM_NOT_SUSPECTED,
    )

    result = fuse_study_decision(session, study)

    assert result["decision_state"] == DecisionState.GBM_NOT_SUSPECTED
    assert result["calibrated_probability_gbm"] == pytest.approx(0.08)
    assert result["other_intracranial_abnormality_not_excluded"] is True
    assert "normal brain" not in result["user_facing_summary"].lower()
    assert result["report_ready"] is True


def test_safety_signals_can_downgrade_suspected_but_never_flip_to_not_suspected(session):
    study = make_study(session, SourceFormat.IMAGE)
    freeze_classifier_deployment_for_test(session, study)
    add_classifier_run(
        session,
        study,
        probability=0.91,
        state=DecisionState.GBM_SUSPECTED,
        ood=True,
    )

    result = fuse_study_decision(session, study)

    assert result["decision_state"] == DecisionState.INDETERMINATE
    assert result["decision_state"] != DecisionState.GBM_NOT_SUSPECTED
    assert "CLASSIFIER_OOD_LIKENESS" in result["safety_reason_codes"]


def test_high_classifier_with_reviewed_zero_wt_is_discordant_indeterminate_when_bridge_future_validated(session):
    study = make_study(session, SourceFormat.NIFTI)
    study.capability_summary = {
        **study.capability_summary,
        "volumetric_to_2d_classifier_bridge_validated": True,
        "classifier_deployment_strategy_frozen": True,
        "capabilities": {
            **study.capability_summary["capabilities"],
            "two_d_classification": {"state": "eligible"},
        },
    }
    session.commit()
    add_classifier_run(
        session,
        study,
        probability=0.88,
        state=DecisionState.GBM_SUSPECTED,
    )
    add_reviewed_segmentation(session, study, wt_voxels=0)

    result = fuse_study_decision(session, study)

    assert result["decision_state"] == DecisionState.INDETERMINATE
    assert "CLASSIFIER_SEGMENTATION_DISCORDANCE_HIGH_PROBABILITY_NO_LESION" in result["safety_reason_codes"]


def test_unreviewed_volumetric_segmentation_blocks_final_report_even_when_fusion_is_indeterminate(session):
    study = make_study(session, SourceFormat.DICOM)
    add_reviewed_segmentation(
        session,
        study,
        wt_voxels=6,
        review_status=SegmentationReviewStatus.UNREVIEWED,
    )

    result = fuse_study_decision(session, study)

    assert result["decision_state"] == DecisionState.INDETERMINATE
    assert result["report_ready"] is False
    assert "SEGMENTATION_REVIEW_NOT_EXPLICIT" in result["report_blockers"]


def test_decision_fusion_is_persisted_and_audited(session):
    study = make_study(session, SourceFormat.IMAGE)
    freeze_classifier_deployment_for_test(session, study)
    add_classifier_run(
        session,
        study,
        probability=0.82,
        state=DecisionState.GBM_SUSPECTED,
    )

    result = fuse_study_decision(session, study, request_id="test-request")
    persisted = session.get(AnalysisRun, result["analysis_run_uuid"])
    assert persisted is not None
    assert persisted.decision_fusion_version == DECISION_FUSION_VERSION
    assert persisted.decision_evidence_summary["clinical_validation_claimed"] is False
    assert persisted.decision_fused_at is not None
    assert session.scalar(
        select(func.count()).select_from(AuditLog).where(AuditLog.action == AuditAction.DECISION_FUSED)
    ) == 1

    current = get_current_fused_decision(session, study)
    assert current["analysis_run_uuid"] == result["analysis_run_uuid"]
    assert current["decision_state"] == DecisionState.GBM_SUSPECTED


def test_homepage_uses_single_document_scroll_owner_contract():
    css = Path("frontend/app/globals.css").read_text(encoding="utf-8")
    compact = "".join(css.split())
    assert "html{" in compact
    assert "overflow-x:clip" in compact
    assert "overflow-y:auto" in compact
    assert "body{margin:0" in compact
    assert "overflow:visible" in compact
    assert "body{overflow-x:hidden" not in compact
    assert ".ng-home-shell,.intake-shell{min-height:100vh;position:relative;overflow-x:clip" in compact
    assert ".ng-home-shell,.intake-shell{min-height:100vh;position:relative;overflow-x:hidden" not in compact


def test_decision_routes_are_registered_in_public_openapi():
    from gbm_ai.api.main import create_app

    paths = set(create_app().openapi().get("paths", {}))
    assert "/api/v1/studies/{study_uuid}/decision/fuse" in paths
    assert "/api/v1/studies/{study_uuid}/decision/current" in paths
