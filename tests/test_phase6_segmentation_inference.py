from __future__ import annotations

import uuid
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import (
    AnalysisRun,
    AnalysisStatus,
    CapabilityRoutingStatus,
    SegmentationPreparationStatus,
    StudyQCStatus,
    StudyStatus,
)
from gbm_ai.api.models.segmentation import Segmentation, SegmentationReviewStatus
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.segmentation.bundle_runtime import BUNDLE_MODEL_SHA256, BUNDLE_NAME, BUNDLE_VERSION
from gbm_ai.api.segmentation.inference import (
    SEGMENTATION_INFERENCE_VERSION,
    SegmentationExecutionResult,
    SegmentationInferenceError,
    SegmentationMaskArtifact,
    binary_masks_to_brats_labelmap,
    resolve_inference_device,
)
from gbm_ai.api.segmentation.model_input import MODEL_INPUT_PREPROCESSING_VERSION, MODEL_INPUT_VERSION
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.services.segmentation_inference import (
    SegmentationInferenceServiceError,
    get_latest_segmentation_result,
    run_segmentation_inference,
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
            "inference_contract": {
                "roi_size": [240, 240, 160],
                "sw_batch_size": 1,
                "overlap": 0.5,
                "activation": "sigmoid",
                "threshold": 0.5,
            },
        },
        "model_execution_started": False,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
    }
    session.commit()
    return study


def make_fake_execution(storage: LocalObjectStore, study_id) -> SegmentationExecutionResult:
    artifacts = {}
    for label in ("tc", "wt", "et", "labelmap"):
        key = storage.generate_study_derived_key(study_id, f"test_{label}", suffix=".bin")
        stored = storage.put_stream(key, BytesIO(f"synthetic-{label}".encode()))
        artifacts[label] = SegmentationMaskArtifact.from_stored(
            stored,
            foreground_voxels={"tc": 3, "wt": 7, "et": 2, "labelmap": 8}[label],
        )
    return SegmentationExecutionResult(
        device="cpu",
        amp_enabled=False,
        runtime_seconds=0.125,
        spatial_shape=(8, 8, 8),
        affine_ras=np.eye(4, dtype=np.float64),
        tc=artifacts["tc"],
        wt=artifacts["wt"],
        et=artifacts["et"],
        labelmap=artifacts["labelmap"],
    )


def test_brats_priority_labelmap_matches_frozen_bundle_semantics():
    tc = np.zeros((2, 2, 2), dtype=np.uint8)
    wt = np.zeros_like(tc)
    et = np.zeros_like(tc)
    wt[0, 0, 0] = 1
    tc[0, 0, 1] = 1
    et[0, 1, 0] = 1
    # Overlap proves ET priority over TC/WT and TC priority over WT.
    wt[1, 0, 0] = 1
    tc[1, 0, 0] = 1
    et[1, 1, 0] = 1
    tc[1, 1, 0] = 1
    wt[1, 1, 0] = 1

    labelmap = binary_masks_to_brats_labelmap(tc, wt, et)
    assert labelmap[0, 0, 0] == 2
    assert labelmap[0, 0, 1] == 1
    assert labelmap[0, 1, 0] == 4
    assert labelmap[1, 0, 0] == 1
    assert labelmap[1, 1, 0] == 4


def test_cpu_device_resolution_is_explicit_and_safe():
    assert resolve_inference_device("cpu") == "cpu"
    with pytest.raises(SegmentationInferenceError, match="auto, cpu, or cuda"):
        resolve_inference_device("quantum")


def test_successful_inference_creates_analysis_model_registry_and_segmentation(monkeypatch, session, tmp_path):
    study = make_ready_study(session, "GBM-P6-S5-SUCCESS")
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_inference.execute_and_persist_segmentation",
        lambda storage, study_uuid, **kwargs: make_fake_execution(storage, study_uuid),
    )

    result = run_segmentation_inference(
        session,
        storage,
        study,
        bundle_dir=Path(tmp_path / "unused-bundle"),
        device_preference="cpu",
        max_spatial_voxels=20_000_000,
    )

    assert result["status"] == "complete"
    assert result["output_channel_order"] == ["TC", "WT", "ET"]
    assert result["decision_state"] == "pending"
    assert result["segmentation_is_gbm_diagnosis"] is False
    assert result["physical_volume_generated"] is False
    assert result["anatomical_localization_generated"] is False
    assert result["review_status"] == "unreviewed"

    run = session.scalar(select(AnalysisRun))
    segmentation = session.scalar(select(Segmentation))
    assert run is not None and run.status == AnalysisStatus.COMPLETE
    assert run.decision_state.value == "pending"
    assert segmentation is not None
    assert segmentation.review_status == SegmentationReviewStatus.UNREVIEWED
    assert segmentation.weights_checksum_sha256 == BUNDLE_MODEL_SHA256
    inference_summary = dict(study.segmentation_preparation_summary or {}).get("inference") or {}
    assert uuid.UUID(str(inference_summary["segmentation_uuid"])) == segmentation.id
    assert uuid.UUID(str(inference_summary["analysis_run_uuid"])) == run.id
    assert segmentation.voxel_counts["TC"] == 3
    assert segmentation.voxel_counts["WT"] == 7
    assert segmentation.voxel_counts["ET"] == 2


def test_repeated_inference_is_idempotent_when_artifacts_are_intact(monkeypatch, session, tmp_path):
    study = make_ready_study(session, "GBM-P6-S5-IDEMPOTENT")
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)
    calls = {"count": 0}

    def fake(storage, study_uuid, **kwargs):
        calls["count"] += 1
        return make_fake_execution(storage, study_uuid)

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_inference.execute_and_persist_segmentation",
        fake,
    )

    first = run_segmentation_inference(
        session, storage, study,
        bundle_dir=tmp_path,
        device_preference="cpu",
        max_spatial_voxels=20_000_000,
    )
    second = run_segmentation_inference(
        session, storage, study,
        bundle_dir=tmp_path,
        device_preference="cpu",
        max_spatial_voxels=20_000_000,
    )

    assert first["segmentation_uuid"] == second["segmentation_uuid"]
    assert calls["count"] == 1
    assert session.scalar(select(func.count()).select_from(AnalysisRun)) == 1
    assert session.scalar(select(func.count()).select_from(Segmentation)) == 1


def test_inference_failure_marks_analysis_failed_and_creates_no_segmentation(monkeypatch, session, tmp_path):
    study = make_ready_study(session, "GBM-P6-S5-FAIL")
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)

    def fail(*args, **kwargs):
        raise SegmentationInferenceError("SEGMENTATION_INFERENCE_OUT_OF_MEMORY", "synthetic OOM")

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_inference.execute_and_persist_segmentation",
        fail,
    )

    with pytest.raises(SegmentationInferenceServiceError, match="synthetic OOM"):
        run_segmentation_inference(
            session, storage, study,
            bundle_dir=tmp_path,
            device_preference="cpu",
            max_spatial_voxels=20_000_000,
        )

    run = session.scalar(select(AnalysisRun))
    assert run is not None and run.status == AnalysisStatus.FAILED
    assert run.safety_reason_codes == ["SEGMENTATION_INFERENCE_OUT_OF_MEMORY"]
    assert session.scalar(select(func.count()).select_from(Segmentation)) == 0


def test_gate_failure_creates_no_analysis_run(session, tmp_path):
    study = make_ready_study(session, "GBM-P6-S5-GATE")
    study.capability_routing_status = CapabilityRoutingStatus.REVIEW_REQUIRED
    session.commit()
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)

    with pytest.raises(SegmentationInferenceServiceError, match="routing"):
        run_segmentation_inference(
            session, storage, study,
            bundle_dir=tmp_path,
            device_preference="cpu",
            max_spatial_voxels=20_000_000,
        )

    assert session.scalar(select(func.count()).select_from(AnalysisRun)) == 0


def test_get_latest_result_preserves_non_diagnostic_boundary(monkeypatch, session, tmp_path):
    study = make_ready_study(session, "GBM-P6-S5-GET")
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_inference.execute_and_persist_segmentation",
        lambda storage, study_uuid, **kwargs: make_fake_execution(storage, study_uuid),
    )
    run_segmentation_inference(
        session, storage, study,
        bundle_dir=tmp_path,
        device_preference="cpu",
        max_spatial_voxels=20_000_000,
    )
    result = get_latest_segmentation_result(session, study)
    assert result["segmentation_is_gbm_diagnosis"] is False
    assert result["decision_state"] == "pending"
    assert result["clinical_validation_claimed"] is False


def test_upstream_invalidation_prevents_old_result_from_being_served(monkeypatch, session, tmp_path):
    study = make_ready_study(session, "GBM-P6-S5-STALE")
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_inference.execute_and_persist_segmentation",
        lambda storage, study_uuid, **kwargs: make_fake_execution(storage, study_uuid),
    )
    run_segmentation_inference(
        session, storage, study,
        bundle_dir=tmp_path,
        device_preference="cpu",
        max_spatial_voxels=20_000_000,
    )

    study.segmentation_preparation_status = SegmentationPreparationStatus.PENDING
    study.segmentation_preparation_summary = {}
    session.commit()

    with pytest.raises(
        SegmentationInferenceServiceError,
        match="must not be served as current",
    ):
        get_latest_segmentation_result(session, study)


def test_new_model_input_checksum_does_not_reuse_old_segmentation(monkeypatch, session, tmp_path):
    study = make_ready_study(session, "GBM-P6-S5-NEWINPUT")
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)
    calls = {"count": 0}

    def fake(storage, study_uuid, **kwargs):
        calls["count"] += 1
        return make_fake_execution(storage, study_uuid)

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_inference.execute_and_persist_segmentation",
        fake,
    )
    first = run_segmentation_inference(
        session, storage, study,
        bundle_dir=tmp_path,
        device_preference="cpu",
        max_spatial_voxels=20_000_000,
    )

    current = dict(study.segmentation_preparation_summary)
    model_input = dict(current["model_input"])
    model_input["checksum_sha256"] = "b" * 64
    current["model_input"] = model_input
    current["model_execution_started"] = False
    current["segmentation_generated"] = False
    current["inference"] = {}
    study.segmentation_preparation_summary = current
    study.segmentation_preparation_status = SegmentationPreparationStatus.READY
    session.commit()

    second = run_segmentation_inference(
        session, storage, study,
        bundle_dir=tmp_path,
        device_preference="cpu",
        max_spatial_voxels=20_000_000,
    )

    assert first["segmentation_uuid"] != second["segmentation_uuid"]
    assert calls["count"] == 2
    assert session.scalar(select(func.count()).select_from(AnalysisRun)) == 2
    assert session.scalar(select(func.count()).select_from(Segmentation)) == 2
