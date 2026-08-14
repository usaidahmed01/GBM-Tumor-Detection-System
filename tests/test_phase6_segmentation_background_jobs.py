from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import (
    AnalysisRun,
    AnalysisStatus,
    CapabilityRoutingStatus,
    DecisionState,
    QCState,
    SegmentationPreparationStatus,
    StudyQCStatus,
    StudyStatus,
)
from gbm_ai.api.models.segmentation import SegmentationJob, SegmentationJobStatus
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.segmentation.bundle_runtime import BUNDLE_MODEL_SHA256, BUNDLE_NAME, BUNDLE_VERSION
from gbm_ai.api.segmentation.model_input import MODEL_INPUT_PREPROCESSING_VERSION, MODEL_INPUT_VERSION
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.services.segmentation_jobs import (
    LEASE_EXPIRED_CODE,
    claim_next_segmentation_job,
    enqueue_segmentation_job,
    fail_or_requeue_segmentation_job,
    heartbeat_segmentation_job,
    recover_stale_segmentation_jobs,
    utcnow,
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


def make_ready_study(session: Session, patient_id: str):
    patient = create_patient(
        session,
        PatientCreate(patient_id=patient_id, age_years=45, privacy_flags={"synthetic": True}),
    )
    assessment = create_assessment(
        session,
        AssessmentCreate(
            patient_uuid=patient.id,
            mri_date=date(2026, 8, 15),
            prior_treatment=False,
        ),
    )
    study = create_study(session, StudyCreate(assessment_uuid=assessment.id))
    study.status = StudyStatus.READY_FOR_ANALYSIS
    study.qc_status = StudyQCStatus.PASS
    study.capability_routing_status = CapabilityRoutingStatus.READY
    study.capability_summary = {
        "capabilities": {
            "three_d_segmentation": {
                "state": "eligible",
                "input_eligible": True,
                "execution_started": False,
            }
        },
        "model_execution_started": False,
    }
    study.segmentation_preparation_status = SegmentationPreparationStatus.READY
    study.segmentation_preparation_summary = {
        "model_input": {
            "version": MODEL_INPUT_VERSION,
            "status": "ready",
            "preprocessing_version": MODEL_INPUT_PREPROCESSING_VERSION,
            "bundle_name": BUNDLE_NAME,
            "bundle_version": BUNDLE_VERSION,
            "bundle_model_sha256": BUNDLE_MODEL_SHA256,
            "spatial_shape": [8, 8, 8],
            "checksum_sha256": "a" * 64,
        },
        "model_execution_started": False,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
    }
    session.commit()
    return study


def test_enqueue_is_idempotent_for_same_immutable_model_input(session):
    study = make_ready_study(session, "GBM-P6-S6-IDEMPOTENT")

    first = enqueue_segmentation_job(session, study, max_attempts=2)
    second = enqueue_segmentation_job(session, study, max_attempts=2)

    assert first["job_uuid"] == second["job_uuid"]
    assert first["status"] == "queued"
    assert first["model_input_checksum_sha256"] == "a" * 64
    assert len(list(session.scalars(select(SegmentationJob)))) == 1


def test_claim_and_heartbeat_use_worker_lease(session):
    study = make_ready_study(session, "GBM-P6-S6-LEASE")
    created = enqueue_segmentation_job(session, study, max_attempts=2)

    job = claim_next_segmentation_job(
        session,
        worker_id="worker-test",
        lease_seconds=900,
    )

    assert job is not None
    assert job.id == created["job_uuid"]
    assert job.status == SegmentationJobStatus.RUNNING
    assert job.attempts == 1
    original_expiry = job.lease_expires_at

    assert heartbeat_segmentation_job(
        session,
        job.id,
        worker_id="worker-test",
        lease_seconds=1200,
    ) is True
    session.refresh(job)
    assert job.heartbeat_at is not None
    assert job.lease_expires_at is not None
    assert original_expiry is not None


def test_failed_job_requeues_then_stops_at_attempt_limit(session):
    study = make_ready_study(session, "GBM-P6-S6-RETRY")
    enqueue_segmentation_job(session, study, max_attempts=2)

    first = claim_next_segmentation_job(session, worker_id="worker-a", lease_seconds=900)
    assert first is not None and first.attempts == 1
    requeued = fail_or_requeue_segmentation_job(
        session,
        first.id,
        worker_id="worker-a",
        error_code="SYNTHETIC_FAILURE",
        retry_delay_seconds=0,
    )
    assert requeued.status == SegmentationJobStatus.QUEUED
    assert requeued.last_error_code == "SYNTHETIC_FAILURE"

    second = claim_next_segmentation_job(session, worker_id="worker-b", lease_seconds=900)
    assert second is not None and second.attempts == 2
    failed = fail_or_requeue_segmentation_job(
        session,
        second.id,
        worker_id="worker-b",
        error_code="SYNTHETIC_FAILURE_2",
        retry_delay_seconds=0,
    )
    assert failed.status == SegmentationJobStatus.FAILED
    assert failed.completed_at is not None


def test_stale_running_job_is_recovered_and_linked_analysis_is_failed(session):
    study = make_ready_study(session, "GBM-P6-S6-RECOVER")
    enqueue_segmentation_job(session, study, max_attempts=2)
    job = claim_next_segmentation_job(session, worker_id="dead-worker", lease_seconds=900)
    assert job is not None

    analysis = AnalysisRun(
        study_id=study.id,
        status=AnalysisStatus.RUNNING,
        qc_state=QCState.PASS,
        decision_state=DecisionState.PENDING,
        safety_reason_codes=[],
        started_at=utcnow(),
    )
    session.add(analysis)
    session.flush()
    job.analysis_run_id = analysis.id
    job.lease_expires_at = utcnow() - timedelta(seconds=1)
    session.commit()

    recovered = recover_stale_segmentation_jobs(session, retry_delay_seconds=0)
    assert recovered == 1

    session.refresh(job)
    session.refresh(analysis)
    assert job.status == SegmentationJobStatus.QUEUED
    assert job.worker_id is None
    assert job.last_error_code == LEASE_EXPIRED_CODE
    assert analysis.status == AnalysisStatus.FAILED
    assert LEASE_EXPIRED_CODE in analysis.safety_reason_codes


def test_stale_job_at_attempt_limit_fails_instead_of_looping_forever(session):
    study = make_ready_study(session, "GBM-P6-S6-RECOVER-LIMIT")
    enqueue_segmentation_job(session, study, max_attempts=1)
    job = claim_next_segmentation_job(session, worker_id="dead-worker", lease_seconds=900)
    assert job is not None and job.attempts == 1
    job.lease_expires_at = utcnow() - timedelta(seconds=1)
    session.commit()

    assert recover_stale_segmentation_jobs(session, retry_delay_seconds=0) == 1
    session.refresh(job)
    assert job.status == SegmentationJobStatus.FAILED
    assert job.completed_at is not None
    assert job.last_error_code == LEASE_EXPIRED_CODE


def test_gate_failure_does_not_enqueue_job(session):
    study = make_ready_study(session, "GBM-P6-S6-GATE")
    study.capability_routing_status = CapabilityRoutingStatus.REVIEW_REQUIRED
    session.commit()

    with pytest.raises(Exception, match="routing"):
        enqueue_segmentation_job(session, study, max_attempts=2)

    assert list(session.scalars(select(SegmentationJob))) == []
