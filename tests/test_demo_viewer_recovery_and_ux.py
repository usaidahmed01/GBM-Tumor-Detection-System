from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import (
    CapabilityRoutingStatus,
    SegmentationPreparationStatus,
    StudyQCStatus,
    StudyStatus,
)
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.segmentation.bundle_runtime import BUNDLE_MODEL_SHA256, BUNDLE_NAME, BUNDLE_VERSION
from gbm_ai.api.segmentation.inference import SegmentationExecutionResult, SegmentationMaskArtifact
from gbm_ai.api.segmentation.model_input import MODEL_INPUT_PREPROCESSING_VERSION, MODEL_INPUT_VERSION
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.services.clinical_viewer import _require_current_segmentation
from gbm_ai.api.services.segmentation_inference import run_segmentation_inference
from gbm_ai.api.storage.local import LocalObjectStore


ROOT = Path(__file__).resolve().parents[1]


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


def _ready_study(session: Session):
    patient = create_patient(
        session,
        PatientCreate(patient_id="DEMO-RECOVERY", age_years=45, privacy_flags={"synthetic": True}),
    )
    assessment = create_assessment(
        session,
        AssessmentCreate(patient_uuid=patient.id, mri_date=date(2026, 8, 16), prior_treatment=False),
    )
    study = create_study(session, StudyCreate(assessment_uuid=assessment.id))
    study.status = StudyStatus.READY_FOR_ANALYSIS
    study.qc_status = StudyQCStatus.PASS
    study.capability_routing_status = CapabilityRoutingStatus.READY
    study.capability_summary = {"capabilities": {"three_d_segmentation": {"state": "eligible", "input_eligible": True}}}
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
            "checksum_sha256": "b" * 64,
            "inference_contract": {"roi_size": [240, 240, 160], "sw_batch_size": 1, "overlap": 0.5, "activation": "sigmoid", "threshold": 0.5},
        },
        "model_execution_started": False,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
    }
    session.commit()
    return study


def _fake_execution(storage: LocalObjectStore, study_id) -> SegmentationExecutionResult:
    artifacts = {}
    for label in ("tc", "wt", "et", "labelmap"):
        stored = storage.put_stream(
            storage.generate_study_derived_key(study_id, f"demo_{label}", suffix=".bin"),
            __import__("io").BytesIO(label.encode()),
        )
        artifacts[label] = SegmentationMaskArtifact.from_stored(stored, foreground_voxels=3)
    return SegmentationExecutionResult(
        device="cpu",
        amp_enabled=False,
        runtime_seconds=0.1,
        spatial_shape=(8, 8, 8),
        affine_ras=np.eye(4, dtype=np.float64),
        tc=artifacts["tc"], wt=artifacts["wt"], et=artifacts["et"], labelmap=artifacts["labelmap"],
    )


def test_legacy_none_segmentation_reference_is_recovered_from_current_model_input(monkeypatch, session, tmp_path):
    study = _ready_study(session)
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=8 * 1024 * 1024)
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_inference.execute_and_persist_segmentation",
        lambda storage, study_uuid, **kwargs: _fake_execution(storage, study_uuid),
    )
    result = run_segmentation_inference(
        session, storage, study, bundle_dir=tmp_path, device_preference="cpu", max_spatial_voxels=1_000_000
    )
    expected = uuid.UUID(str(result["segmentation_uuid"]))

    broken = dict(study.segmentation_preparation_summary or {})
    inference = dict(broken.get("inference") or {})
    inference["segmentation_uuid"] = "None"
    broken["inference"] = inference
    study.segmentation_preparation_summary = broken
    session.commit()

    _, segmentation = _require_current_segmentation(session, study)
    assert segmentation.id == expected
    repaired = dict(study.segmentation_preparation_summary or {}).get("inference") or {}
    assert uuid.UUID(str(repaired["segmentation_uuid"])) == expected


def test_home_recent_analysis_card_is_fully_clickable():
    source = (ROOT / "frontend/app/page.jsx").read_text(encoding="utf-8")
    assert 'className="ng-recent-card"' in source
    assert 'aria-label={`Continue ${item.caseReference}`}' in source
    assert '<article key={item.studyUuid} className="ng-recent-card">' not in source


def test_intake_has_visible_async_operation_feedback_and_no_duplicate_banner():
    source = (ROOT / "frontend/components/intake/AnalysisIntakeWorkspace.jsx").read_text(encoding="utf-8")
    assert 'className="operation-progress"' in source
    assert 'button-working-mark' in source
    assert 'aria-busy={busy}' in source
    assert source.count('intake-banner intake-banner--info') == 1
    assert 'Technical study IDs are managed internally' not in source
    assert 'Internal study routing was restored automatically' not in source
    assert 'value.reasons?.[0]' not in source
    assert 'Confirm detected sequences' in source


def test_viewer_avoids_extra_next_dynamic_chunk_and_supports_retry():
    source = (ROOT / "frontend/components/viewer/ViewerWorkspace.jsx").read_text(encoding="utf-8")
    assert "import dynamic from 'next/dynamic'" not in source
    assert "import CornerstoneMprViewer from './CornerstoneMprViewer'" in source
    assert 'Preparing measurements…' in source
    assert '>Retry</button>' in source
    assert 'Protected session' not in source
    assert '<span>UI</span>' not in source


def test_global_interaction_feedback_styles_are_present():
    css = (ROOT / "frontend/app/globals.css").read_text(encoding="utf-8")
    for token in (".button-working-mark", ".operation-progress", ".ng-recent-card:hover", "@keyframes ng-working-spin"):
        assert token in css


def test_next_config_suppresses_only_the_known_cornerstone_worker_runtime_warning():
    source = (ROOT / "frontend/next.config.mjs").read_text(encoding="utf-8")
    assert "config.ignoreWarnings" in source
    assert "compute, webpack" in source
    assert "Circular dependency between chunks with runtime" in source
