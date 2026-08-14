from __future__ import annotations

import argparse
import socket
import threading
import time
import uuid
from contextlib import contextmanager

from gbm_ai.api.config import Settings, get_settings
from gbm_ai.api.db import DatabaseManager
from gbm_ai.api.models.analysis import Study
from gbm_ai.api.services.segmentation_inference import (
    SegmentationInferenceServiceError,
    run_segmentation_inference,
)
from gbm_ai.api.services.segmentation_jobs import (
    SegmentationJobServiceError,
    claim_next_segmentation_job,
    complete_segmentation_job,
    fail_or_requeue_segmentation_job,
    get_segmentation_job,
    heartbeat_segmentation_job,
    recover_stale_segmentation_jobs,
)
from gbm_ai.api.storage.local import LocalObjectStore


WORKER_VERSION = "phase6_step6_segmentation_worker_v1"


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{uuid.uuid4().hex[:12]}"


def build_storage(settings: Settings) -> LocalObjectStore:
    return LocalObjectStore(
        settings.storage_root_resolved,
        max_object_bytes=settings.storage_max_object_bytes,
        chunk_bytes=settings.storage_chunk_bytes,
    )


@contextmanager
def heartbeat_lease(
    database: DatabaseManager,
    *,
    job_id: uuid.UUID,
    worker_id: str,
    lease_seconds: int,
    heartbeat_seconds: int,
):
    stop_event = threading.Event()

    def loop() -> None:
        while not stop_event.wait(heartbeat_seconds):
            db = database.session_factory()
            try:
                alive = heartbeat_segmentation_job(
                    db,
                    job_id,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                )
                if not alive:
                    return
            except Exception:
                db.rollback()
            finally:
                db.close()

    thread = threading.Thread(
        target=loop,
        name=f"segmentation-heartbeat-{job_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=max(1.0, min(float(heartbeat_seconds), 5.0)))


def run_one_job(
    database: DatabaseManager,
    storage: LocalObjectStore,
    settings: Settings,
    *,
    worker_id: str,
) -> bool:
    with database.session_factory() as db:
        recover_stale_segmentation_jobs(
            db,
            retry_delay_seconds=settings.segmentation_job_retry_delay_seconds,
        )
        job = claim_next_segmentation_job(
            db,
            worker_id=worker_id,
            lease_seconds=settings.segmentation_job_lease_seconds,
        )
        if job is None:
            return False
        job_id = job.id
        study_id = job.study_id

    try:
        with heartbeat_lease(
            database,
            job_id=job_id,
            worker_id=worker_id,
            lease_seconds=settings.segmentation_job_lease_seconds,
            heartbeat_seconds=settings.segmentation_job_heartbeat_seconds,
        ):
            with database.session_factory() as db:
                job = get_segmentation_job(db, job_id)
                study = db.get(Study, study_id)
                if study is None:
                    raise SegmentationJobServiceError(
                        "SEGMENTATION_JOB_STUDY_NOT_FOUND",
                        "background job study no longer exists",
                    )

                result = run_segmentation_inference(
                    db,
                    storage,
                    study,
                    bundle_dir=(
                        settings.segmentation_bundle_root_resolved
                        / "brats_mri_segmentation"
                    ),
                    device_preference=settings.segmentation_inference_device,
                    max_spatial_voxels=settings.segmentation_inference_max_spatial_voxels,
                    request_id=job.request_id,
                    background_job=job,
                )
                analysis_run_id = result["analysis_run_uuid"]
                segmentation_id = result["segmentation_uuid"]

            with database.session_factory() as db:
                complete_segmentation_job(
                    db,
                    job_id,
                    worker_id=worker_id,
                    analysis_run_id=analysis_run_id,
                    segmentation_id=segmentation_id,
                )
        return True

    except SegmentationInferenceServiceError as exc:
        error_code = exc.code
    except SegmentationJobServiceError as exc:
        error_code = exc.code
    except Exception:
        error_code = "SEGMENTATION_BACKGROUND_WORKER_UNEXPECTED_FAILURE"

    with database.session_factory() as db:
        try:
            fail_or_requeue_segmentation_job(
                db,
                job_id,
                worker_id=worker_id,
                error_code=error_code,
                retry_delay_seconds=settings.segmentation_job_retry_delay_seconds,
            )
        except SegmentationJobServiceError:
            db.rollback()
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Durable Phase 6 SegResNet worker. It claims PostgreSQL-backed "
            "jobs, refreshes a lease during inference, and requeues stale "
            "jobs after unexpected process loss."
        )
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="process at most one available job and exit",
    )
    parser.add_argument(
        "--recover-only",
        action="store_true",
        help="recover expired leases and exit without claiming a new job",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="optional technical worker identifier; defaults to host + random suffix",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    database = DatabaseManager(settings)
    storage = build_storage(settings)
    worker_id = args.worker_id or default_worker_id()

    print("PHASE 6 STEP 6 — SEGMENTATION BACKGROUND WORKER")
    print("=" * 72)
    print(f"Worker version:               {WORKER_VERSION}")
    print(f"Worker id:                    {worker_id}")
    print(f"Inference device preference:  {settings.segmentation_inference_device}")
    print(f"Lease seconds:                {settings.segmentation_job_lease_seconds}")
    print(f"Heartbeat seconds:            {settings.segmentation_job_heartbeat_seconds}")
    print(f"Max attempts:                 {settings.segmentation_job_max_attempts}")

    try:
        if args.recover_only:
            with database.session_factory() as db:
                recovered = recover_stale_segmentation_jobs(
                    db,
                    retry_delay_seconds=settings.segmentation_job_retry_delay_seconds,
                )
            print(f"Recovered stale jobs:         {recovered}")
            return

        while True:
            processed = run_one_job(
                database,
                storage,
                settings,
                worker_id=worker_id,
            )
            if args.once:
                print("Job processed:                 YES" if processed else "Job processed:                 NO (queue empty)")
                return
            if not processed:
                time.sleep(settings.segmentation_job_poll_seconds)
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
