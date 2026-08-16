from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import (
    AnalysisRun,
    AnalysisStatus,
    BrainScopeStatus,
    CapabilityRoutingStatus,
    DecisionState,
    QCState,
    SourceFormat,
    StudyQCStatus,
    StudyStatus,
)
from gbm_ai.api.models.clinical import Assessment, AssessmentStatus, Patient, ScopeStatus, Sex
from gbm_ai.api.models.report import ClinicalReport
from gbm_ai.api.services.clinical_report import (
    CLINICAL_REPORT_VERSION,
    ClinicalReportError,
    finalize_report,
    preview_report,
)
from gbm_ai.api.services.decision_fusion import DECISION_FUSION_VERSION


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


def make_ready_study(session: Session, *, report_ready: bool = True):
    patient = Patient(
        patient_id="NGAI-REPORT-001",
        patient_name="Demo Patient",
        age_years=48,
        sex=Sex.UNKNOWN,
        privacy_flags={"synthetic": True},
    )
    session.add(patient)
    session.flush()
    assessment = Assessment(
        patient_id=patient.id,
        mri_date=date(2026, 8, 16),
        symptoms=["headache"],
        symptom_duration="2 weeks",
        prior_treatment=False,
        clinical_notes="synthetic report fixture",
        status=AssessmentStatus.COMPLETE,
        scope_status=ScopeStatus.IN_SCOPE,
    )
    session.add(assessment)
    session.flush()
    from gbm_ai.api.models.analysis import Study
    study = Study(
        assessment_id=assessment.id,
        source_format=SourceFormat.IMAGE,
        modality="MRI",
        qc_status=StudyQCStatus.PASS,
        brain_scope_status=BrainScopeStatus.CLINICIAN_CONFIRMED,
        capability_routing_status=CapabilityRoutingStatus.READY,
        capability_summary={"capabilities": {}},
        status=StudyStatus.READY_FOR_ANALYSIS,
    )
    session.add(study)
    session.flush()
    now = datetime.now(timezone.utc)
    decision = AnalysisRun(
        study_id=study.id,
        status=AnalysisStatus.COMPLETE,
        qc_state=QCState.PASS,
        calibrated_probability_gbm=0.42,
        decision_state=DecisionState.INDETERMINATE,
        safety_reason_codes=["CLASSIFIER_EVIDENCE_NOT_AVAILABLE"],
        decision_fusion_version=DECISION_FUSION_VERSION,
        decision_evidence_summary={
            "source_format": "image",
            "classifier": {"available": False},
            "segmentation": {"available": False},
            "report_ready": report_ready,
            "report_blockers": [] if report_ready else ["SEGMENTATION_REVIEW_NOT_EXPLICIT"],
            "other_intracranial_abnormality_not_excluded": False,
            "user_facing_summary": "GBM assessment is indeterminate.",
            "volumetric_classifier_bridge_validated": False,
        },
        decision_fused_at=now,
        started_at=now,
        completed_at=now,
    )
    session.add(decision)
    session.commit()
    return study, decision


def test_report_preview_contains_required_structured_sections(session):
    study, decision = make_ready_study(session)
    result = preview_report(session, study)

    assert result["report_version"] == CLINICAL_REPORT_VERSION
    assert result["finalization_ready"] is True
    assert result["decision_analysis_run_uuid"] == decision.id
    payload = result["report"]
    assert payload["patient_study"]["case_reference"] == "NGAI-REPORT-001"
    assert payload["input_validation"]["qc_status"] == "pass"
    assert payload["gbm_assessment"]["state"] == "indeterminate"
    assert payload["limitations"]["segmentation_is_gbm_diagnosis"] is False
    assert payload["limitations"]["clinical_validation_claimed"] is False


def test_finalize_report_creates_immutable_checksum_bound_snapshot(session):
    study, decision = make_ready_study(session)
    row = finalize_report(
        session,
        study,
        clinician_name="Dr Demo",
        clinician_comment="Reviewed for synthetic test.",
    )

    assert row.study_id == study.id
    assert row.decision_analysis_run_id == decision.id
    assert row.report_version == CLINICAL_REPORT_VERSION
    assert len(row.report_checksum_sha256) == 64
    assert row.clinician_name == "Dr Demo"
    assert row.signoff_identity_verified is False

    row.clinician_comment = "attempted mutation"
    with pytest.raises(RuntimeError, match="immutable"):
        session.commit()
    session.rollback()


def test_finalize_report_is_blocked_when_report_readiness_gate_fails(session):
    study, _ = make_ready_study(session, report_ready=False)
    with pytest.raises(ClinicalReportError) as exc:
        finalize_report(session, study, clinician_name="Dr Demo", clinician_comment=None)
    assert exc.value.code == "REPORT_NOT_READY"


def test_report_table_is_registered_in_metadata():
    assert ClinicalReport.__tablename__ in Base.metadata.tables
