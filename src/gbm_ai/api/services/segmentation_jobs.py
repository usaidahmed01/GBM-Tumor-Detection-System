from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import AnalysisRun, AnalysisStatus, Study
from gbm_ai.api.models.audit import AuditAction, AuditActorType, AuditEntityType
from gbm_ai.api.models.segmentation import (
    Segmentation,
    SegmentationJob,
    SegmentationJobStatus,
)
from gbm_ai.api.segmentation.inference import SEGMENTATION_INFERENCE_VERSION
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.services.segmentation_inference import (
    SegmentationInferenceServiceError,
    require_segmentation_execution_gate,
)


SEGMENTATION_JOB_VERSION = "phase6_step6_background_segmentation_job_v1"
LEASE_EXPIRED_CODE = "SEGMENTATION_JOB_LEASE_EXPIRED"


class SegmentationJobServiceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _deduplication_key(study_id: uuid.UUID, checksum: str) -> str:
    return f"{study_id}:{checksum.lower()}"


def _current_model_input_checksum(study: Study) -> str:
    model_input = require_segmentation_execution_gate(study)
    checksum = str(model_input.get("checksum_sha256") or "").lower()
    if len(checksum) != 64:
        raise SegmentationJobServiceError(
            "SEGMENTATION_JOB_MODEL_INPUT_CHECKSUM_INVALID",
            "current Step 4 model input is missing its immutable SHA-256 binding",
        )
    return checksum


def _completed_segmentation_for_input(
    db: Session,
    study_id: uuid.UUID,
    checksum: str,
) -> Segmentation | None:
    return db.scalar(
        select(Segmentation)
        .join(AnalysisRun, AnalysisRun.id == Segmentation.analysis_run_id)
        .where(
            AnalysisRun.study_id == study_id,
            AnalysisRun.status == AnalysisStatus.COMPLETE,
            Segmentation.inference_version == SEGMENTATION_INFERENCE_VERSION,
            Segmentation.model_input_checksum_sha256 == checksum,
        )
        .order_by(Segmentation.created_at.desc())
        .limit(1)
    )


def segmentation_job_to_response(job: SegmentationJob) -> dict:
    return {
        "version": SEGMENTATION_JOB_VERSION,
        "job_uuid": job.id,
        "study_uuid": job.study_id,
        "status": job.status.value,
        "model_input_checksum_sha256": job.model_input_checksum_sha256,
        "attempts": int(job.attempts),
        "max_attempts": int(job.max_attempts),
        "available_at": job.available_at,
        "claimed_at": job.claimed_at,
        "heartbeat_at": job.heartbeat_at,
        "lease_expires_at": job.lease_expires_at,
        "completed_at": job.completed_at,
        "analysis_run_uuid": job.analysis_run_id,
        "segmentation_uuid": job.segmentation_id,
        "last_error_code": job.last_error_code,
        "worker_assigned": bool(job.worker_id),
        "result_available": job.status == SegmentationJobStatus.COMPLETE
        and job.segmentation_id is not None,
        "background_execution_implemented": True,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,
        "next_step": "phase6_complete" if job.status == SegmentationJobStatus.COMPLETE else "phase6_step6_background_execution_and_recovery",
    }


def enqueue_segmentation_job(
    db: Session,
    study: Study,
    *,
    max_attempts: int,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> dict:
    checksum = _current_model_input_checksum(study)
    dedupe = _deduplication_key(study.id, checksum)

    existing = db.scalar(
        select(SegmentationJob).where(
            SegmentationJob.deduplication_key == dedupe
        )
    )
    if existing is not None:
        return segmentation_job_to_response(existing)

    segmentation = _completed_segmentation_for_input(db, study.id, checksum)
    now = utcnow()
    if segmentation is not None:
        job = SegmentationJob(
            study_id=study.id,
            deduplication_key=dedupe,
            model_input_checksum_sha256=checksum,
            status=SegmentationJobStatus.COMPLETE,
            attempts=0,
            max_attempts=max_attempts,
            available_at=now,
            completed_at=now,
            analysis_run_id=segmentation.analysis_run_id,
            segmentation_id=segmentation.id,
            request_id=request_id,
        )
    else:
        job = SegmentationJob(
            study_id=study.id,
            deduplication_key=dedupe,
            model_input_checksum_sha256=checksum,
            status=SegmentationJobStatus.QUEUED,
            attempts=0,
            max_attempts=max_attempts,
            available_at=now,
            request_id=request_id,
        )

    db.add(job)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        concurrent = db.scalar(
            select(SegmentationJob).where(
                SegmentationJob.deduplication_key == dedupe
            )
        )
        if concurrent is None:
            raise
        return segmentation_job_to_response(concurrent)

    record_audit_event(
        db,
        action=AuditAction.SEGMENTATION_JOB_ENQUEUED,
        entity_type=AuditEntityType.SEGMENTATION_JOB,
        entity_uuid=job.id,
        actor_type=actor_type,
        actor_id=actor_id,
        request_id=request_id,
        technical_context={
            "status": job.status.value,
            "operation": "segmentation_background_job",
            "result": "existing_result_bound" if segmentation is not None else "queued",
        },
        commit=False,
    )

    current = dict(study.segmentation_preparation_summary or {})
    current["background_job"] = {
        "version": SEGMENTATION_JOB_VERSION,
        "job_uuid": str(job.id),
        "status": job.status.value,
        "model_input_checksum_sha256": checksum,
        "background_execution_implemented": True,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
        "clinical_validation_claimed": False,
    }
    current["next_step"] = "phase6_complete" if job.status == SegmentationJobStatus.COMPLETE else "phase6_step6_background_execution_and_recovery"
    study.segmentation_preparation_summary = current

    db.commit()
    db.refresh(job)
    return segmentation_job_to_response(job)


def get_segmentation_job(db: Session, job_id: uuid.UUID) -> SegmentationJob:
    job = db.get(SegmentationJob, job_id)
    if job is None:
        raise SegmentationJobServiceError(
            "SEGMENTATION_JOB_NOT_FOUND",
            "segmentation background job not found",
        )
    return job


def claim_next_segmentation_job(
    db: Session,
    *,
    worker_id: str,
    lease_seconds: int,
) -> SegmentationJob | None:
    now = utcnow()
    statement = (
        select(SegmentationJob)
        .where(
            SegmentationJob.status == SegmentationJobStatus.QUEUED,
            SegmentationJob.available_at <= now,
            SegmentationJob.attempts < SegmentationJob.max_attempts,
        )
        .order_by(SegmentationJob.available_at.asc(), SegmentationJob.created_at.asc())
        .limit(1)
    )
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        statement = statement.with_for_update(skip_locked=True)
    else:
        statement = statement.with_for_update()

    job = db.scalar(statement)
    if job is None:
        return None

    job.status = SegmentationJobStatus.RUNNING
    job.attempts = int(job.attempts) + 1
    job.worker_id = worker_id
    job.claimed_at = now
    job.heartbeat_at = now
    job.lease_expires_at = now + timedelta(seconds=lease_seconds)
    job.last_error_code = None

    record_audit_event(
        db,
        action=AuditAction.SEGMENTATION_JOB_CLAIMED,
        entity_type=AuditEntityType.SEGMENTATION_JOB,
        entity_uuid=job.id,
        actor_type=AuditActorType.SYSTEM,
        technical_context={
            "status": "running",
            "operation": "segmentation_background_job",
            "result": "claimed",
        },
        commit=False,
    )
    db.commit()
    db.refresh(job)
    return job


def heartbeat_segmentation_job(
    db: Session,
    job_id: uuid.UUID,
    *,
    worker_id: str,
    lease_seconds: int,
) -> bool:
    job = db.get(SegmentationJob, job_id)
    if (
        job is None
        or job.status != SegmentationJobStatus.RUNNING
        or job.worker_id != worker_id
    ):
        return False
    now = utcnow()
    job.heartbeat_at = now
    job.lease_expires_at = now + timedelta(seconds=lease_seconds)
    db.commit()
    return True


def complete_segmentation_job(
    db: Session,
    job_id: uuid.UUID,
    *,
    worker_id: str,
    analysis_run_id: uuid.UUID,
    segmentation_id: uuid.UUID,
) -> SegmentationJob:
    job = get_segmentation_job(db, job_id)
    if job.status == SegmentationJobStatus.COMPLETE:
        return job
    if job.status != SegmentationJobStatus.RUNNING or job.worker_id != worker_id:
        raise SegmentationJobServiceError(
            "SEGMENTATION_JOB_OWNERSHIP_LOST",
            "worker no longer owns this segmentation job lease",
        )

    now = utcnow()
    job.status = SegmentationJobStatus.COMPLETE
    job.analysis_run_id = analysis_run_id
    job.segmentation_id = segmentation_id
    job.completed_at = now
    job.heartbeat_at = now
    job.lease_expires_at = None
    job.worker_id = None
    job.last_error_code = None

    study = db.get(Study, job.study_id)
    if study is not None:
        current = dict(study.segmentation_preparation_summary or {})
        current["background_job"] = {
            "version": SEGMENTATION_JOB_VERSION,
            "job_uuid": str(job.id),
            "status": "complete",
            "model_input_checksum_sha256": job.model_input_checksum_sha256,
            "analysis_run_uuid": str(analysis_run_id),
            "segmentation_uuid": str(segmentation_id),
            "background_execution_implemented": True,
            "physical_volume_generated": False,
            "anatomical_localization_generated": False,
            "clinical_validation_claimed": False,
        }
        current["next_step"] = "phase6_complete"
        study.segmentation_preparation_summary = current

    record_audit_event(
        db,
        action=AuditAction.SEGMENTATION_JOB_COMPLETED,
        entity_type=AuditEntityType.SEGMENTATION_JOB,
        entity_uuid=job.id,
        actor_type=AuditActorType.SYSTEM,
        technical_context={
            "status": "complete",
            "operation": "segmentation_background_job",
            "result": "wt_tc_et_generated",
        },
        commit=False,
    )
    db.commit()
    db.refresh(job)
    return job


def fail_or_requeue_segmentation_job(
    db: Session,
    job_id: uuid.UUID,
    *,
    worker_id: str | None,
    error_code: str,
    retry_delay_seconds: int,
) -> SegmentationJob:
    job = get_segmentation_job(db, job_id)
    if job.status in {SegmentationJobStatus.COMPLETE, SegmentationJobStatus.FAILED}:
        return job
    if worker_id is not None and job.worker_id not in {None, worker_id}:
        raise SegmentationJobServiceError(
            "SEGMENTATION_JOB_OWNERSHIP_LOST",
            "worker no longer owns this segmentation job lease",
        )

    now = utcnow()
    job.last_error_code = error_code
    job.heartbeat_at = now
    job.lease_expires_at = None
    job.worker_id = None

    if int(job.attempts) < int(job.max_attempts):
        job.status = SegmentationJobStatus.QUEUED
        job.available_at = now + timedelta(seconds=retry_delay_seconds)
        action = AuditAction.SEGMENTATION_JOB_REQUEUED
        result = "requeued"
    else:
        job.status = SegmentationJobStatus.FAILED
        job.completed_at = now
        action = AuditAction.SEGMENTATION_JOB_FAILED
        result = "failed_attempt_limit"

    study = db.get(Study, job.study_id)
    if study is not None:
        current = dict(study.segmentation_preparation_summary or {})
        current["background_job"] = {
            "version": SEGMENTATION_JOB_VERSION,
            "job_uuid": str(job.id),
            "status": job.status.value,
            "model_input_checksum_sha256": job.model_input_checksum_sha256,
            "attempts": int(job.attempts),
            "max_attempts": int(job.max_attempts),
            "last_error_code": error_code,
            "background_execution_implemented": True,
            "physical_volume_generated": False,
            "anatomical_localization_generated": False,
            "clinical_validation_claimed": False,
        }
        current["next_step"] = "phase6_step6_background_execution_and_recovery"
        study.segmentation_preparation_summary = current

    record_audit_event(
        db,
        action=action,
        entity_type=AuditEntityType.SEGMENTATION_JOB,
        entity_uuid=job.id,
        actor_type=AuditActorType.SYSTEM,
        technical_context={
            "status": job.status.value,
            "operation": "segmentation_background_job",
            "result": result,
            "error_type": error_code,
        },
        commit=False,
    )
    db.commit()
    db.refresh(job)
    return job


def recover_stale_segmentation_jobs(
    db: Session,
    *,
    retry_delay_seconds: int,
) -> int:
    now = utcnow()
    statement = select(SegmentationJob).where(
        SegmentationJob.status == SegmentationJobStatus.RUNNING,
        SegmentationJob.lease_expires_at.is_not(None),
        SegmentationJob.lease_expires_at < now,
    )
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        statement = statement.with_for_update(skip_locked=True)
    else:
        statement = statement.with_for_update()

    jobs = list(db.scalars(statement))
    recovered = 0
    for job in jobs:
        if job.analysis_run_id is not None:
            analysis = db.get(AnalysisRun, job.analysis_run_id)
            if analysis is not None and analysis.status == AnalysisStatus.RUNNING:
                analysis.status = AnalysisStatus.FAILED
                analysis.completed_at = now
                codes = list(analysis.safety_reason_codes or [])
                if LEASE_EXPIRED_CODE not in codes:
                    codes.append(LEASE_EXPIRED_CODE)
                analysis.safety_reason_codes = codes

        job.last_error_code = LEASE_EXPIRED_CODE
        job.worker_id = None
        job.heartbeat_at = now
        job.lease_expires_at = None

        if int(job.attempts) < int(job.max_attempts):
            job.status = SegmentationJobStatus.QUEUED
            job.available_at = now + timedelta(seconds=retry_delay_seconds)
            action = AuditAction.SEGMENTATION_JOB_REQUEUED
            result = "recovered_stale_lease"
        else:
            job.status = SegmentationJobStatus.FAILED
            job.completed_at = now
            action = AuditAction.SEGMENTATION_JOB_FAILED
            result = "stale_lease_attempt_limit"

        record_audit_event(
            db,
            action=action,
            entity_type=AuditEntityType.SEGMENTATION_JOB,
            entity_uuid=job.id,
            actor_type=AuditActorType.SYSTEM,
            technical_context={
                "status": job.status.value,
                "operation": "segmentation_background_job",
                "result": result,
                "error_type": LEASE_EXPIRED_CODE,
            },
            commit=False,
        )
        recovered += 1

    if recovered:
        db.commit()
    return recovered
