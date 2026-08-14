from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import (
    AnalysisRun,
    SegmentationPreparationStatus,
    StudyStatus,
)
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.segmentation.bundle_runtime import (
    BUNDLE_INPUT_CHANNEL_ORDER,
    BUNDLE_OUTPUT_CHANNEL_ORDER,
    BUNDLE_VERSION,
    SegmentationBundleRuntimeError,
    validate_bundle_layout,
)
from gbm_ai.api.segmentation.model_input import (
    MODEL_INPUT_VERSION,
    build_normalized_model_input,
    load_prepared_model_input,
    persist_normalized_model_input,
)
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.services.segmentation_model_geometry import (
    SegmentationModelGeometryPreparationError,
)
from gbm_ai.api.services.segmentation_model_input import (
    SegmentationModelInputPreparationError,
    prepare_segmentation_model_input,
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
    SessionLocal = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
    )
    with SessionLocal() as db:
        yield db
    engine.dispose()


def make_study(session: Session, patient_id: str):
    patient = create_patient(
        session,
        PatientCreate(
            patient_id=patient_id,
            age_years=45,
            privacy_flags={"synthetic": True},
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
    study = create_study(
        session,
        StudyCreate(assessment_uuid=assessment.id),
    )
    study.status = StudyStatus.READY_FOR_ANALYSIS
    study.segmentation_preparation_status = SegmentationPreparationStatus.READY
    study.segmentation_preparation_summary = {
        "version": "phase6_step2_volume_preparation_v1",
        "status": "ready",
        "model_geometry": {
            "version": "phase6_step3_model_geometry_v1",
            "status": "ready",
            "channel_order": ["T1C", "T1", "T2", "FLAIR"],
            "channels": [],
            "model_execution_started": False,
        },
        "model_execution_started": False,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
    }
    session.commit()
    return study


def write_synthetic_bundle(bundle_dir: Path, *, overlap: float = 0.5) -> str:
    (bundle_dir / "models").mkdir(parents=True)
    (bundle_dir / "configs").mkdir(parents=True)
    model_bytes = b"synthetic frozen bundle checkpoint bytes"
    (bundle_dir / "models" / "model.pt").write_bytes(model_bytes)
    model_sha = hashlib.sha256(model_bytes).hexdigest()

    metadata = {
        "version": "0.5.4",
        "monai_version": "1.4.0",
        "pytorch_version": "2.4.0",
        "network_data_format": {
            "inputs": {
                "image": {
                    "channel_def": {
                        "0": "T1c",
                        "1": "T1",
                        "2": "T2",
                        "3": "FLAIR",
                    }
                }
            },
            "outputs": {
                "pred": {
                    "channel_def": {
                        "0": "Tumor core",
                        "1": "Whole tumor",
                        "2": "Enhancing tumor",
                    }
                }
            },
        },
    }
    inference = {
        "network_def": {
            "_target_": "SegResNet",
            "blocks_down": [1, 2, 2, 4],
            "blocks_up": [1, 1, 1],
            "init_filters": 16,
            "in_channels": 4,
            "out_channels": 3,
            "dropout_prob": 0.2,
        },
        "preprocessing": {
            "transforms": [
                {"_target_": "LoadImaged"},
                {
                    "_target_": "NormalizeIntensityd",
                    "nonzero": True,
                    "channel_wise": True,
                },
            ]
        },
        "inferer": {
            "_target_": "SlidingWindowInferer",
            "roi_size": [240, 240, 160],
            "sw_batch_size": 1,
            "overlap": overlap,
        },
        "postprocessing": {
            "transforms": [
                {"_target_": "Activationsd", "sigmoid": True},
                {"_target_": "AsDiscreted", "threshold": 0.5},
            ]
        },
    }
    (bundle_dir / "configs" / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (bundle_dir / "configs" / "inference.json").write_text(
        json.dumps(inference), encoding="utf-8"
    )
    return model_sha


def test_frozen_bundle_contract_accepts_exact_054_layout(tmp_path):
    bundle_dir = tmp_path / "brats_mri_segmentation"
    model_sha = write_synthetic_bundle(bundle_dir)

    result = validate_bundle_layout(
        bundle_dir,
        expected_model_sha256=model_sha,
    )

    assert result.version == BUNDLE_VERSION
    assert result.model_sha256 == model_sha
    assert result.metadata_monai_version == "1.4.0"
    assert BUNDLE_INPUT_CHANNEL_ORDER == ("T1C", "T1", "T2", "FLAIR")
    assert BUNDLE_OUTPUT_CHANNEL_ORDER == ("TC", "WT", "ET")


def test_frozen_bundle_contract_rejects_inferer_drift(tmp_path):
    bundle_dir = tmp_path / "brats_mri_segmentation"
    model_sha = write_synthetic_bundle(bundle_dir, overlap=0.25)

    with pytest.raises(
        SegmentationBundleRuntimeError,
        match="sliding-window inference settings",
    ):
        validate_bundle_layout(
            bundle_dir,
            expected_model_sha256=model_sha,
        )


def test_normalization_preserves_channel_order_and_nonzero_standardization(monkeypatch):
    arrays = {}
    for index, sequence in enumerate(BUNDLE_INPUT_CHANNEL_ORDER, start=1):
        data = np.zeros((8, 8, 8), dtype=np.float32)
        data[1:7, 1:7, 1:7] = (
            np.arange(216, dtype=np.float32).reshape(6, 6, 6) + index * 100.0
        )
        arrays[sequence] = data

    monkeypatch.setattr(
        "gbm_ai.api.segmentation.model_input._load_geometry_channel",
        lambda storage, item: (
            arrays[item["sequence"]].copy(),
            np.eye(4, dtype=np.float64),
        ),
    )

    class NumpyNormalizeIntensity:
        def __init__(self, *, nonzero, channel_wise):
            assert nonzero is True
            assert channel_wise is True

        def __call__(self, image):
            output = np.asarray(image, dtype=np.float32).copy()
            for channel in range(output.shape[0]):
                mask = output[channel] != 0
                values = output[channel][mask]
                output[channel][mask] = (
                    values - float(values.mean())
                ) / float(values.std())
            return output

    monkeypatch.setattr(
        "gbm_ai.api.segmentation.model_input._require_monai_normalizer",
        lambda: NumpyNormalizeIntensity,
    )

    geometry = {
        "status": "ready",
        "channel_order": list(BUNDLE_INPUT_CHANNEL_ORDER),
        "channels": [
            {"sequence": sequence}
            for sequence in BUNDLE_INPUT_CHANNEL_ORDER
        ],
    }
    normalized, affine, stats = build_normalized_model_input(object(), geometry)

    assert normalized.shape == (4, 8, 8, 8)
    assert np.array_equal(affine, np.eye(4))
    assert [item["sequence"] for item in stats] == list(BUNDLE_INPUT_CHANNEL_ORDER)
    for index in range(4):
        assert np.all(normalized[index][arrays[BUNDLE_INPUT_CHANNEL_ORDER[index]] == 0] == 0)
        mask = normalized[index] != 0
        assert abs(float(normalized[index][mask].mean())) < 1e-5
        assert abs(float(normalized[index][mask].std()) - 1.0) < 1e-5


def test_actual_monai_normalize_intensity_contract_when_dependency_available():
    monai = pytest.importorskip("monai")
    from monai.transforms import NormalizeIntensity

    data = np.zeros((4, 8, 8, 8), dtype=np.float32)
    for index in range(4):
        values = np.arange(216, dtype=np.float32).reshape(6, 6, 6) + (index + 1) * 10.0
        data[index, 1:7, 1:7, 1:7] = values

    output = NormalizeIntensity(nonzero=True, channel_wise=True)(data)
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    output = np.asarray(output, dtype=np.float32)

    assert output.shape == data.shape
    assert monai.__version__
    for index in range(4):
        assert np.all(output[index][data[index] == 0] == 0)
        mask = data[index] != 0
        assert abs(float(output[index][mask].mean())) < 1e-5
        assert abs(float(output[index][mask].std()) - 1.0) < 1e-5


def test_model_input_artifact_round_trip_preserves_order_and_checksum(tmp_path):
    storage = LocalObjectStore(
        tmp_path / "storage",
        max_object_bytes=64 * 1024 * 1024,
    )
    study_id = __import__("uuid").uuid4()
    image = np.arange(4 * 8 * 8 * 8, dtype=np.float32).reshape(4, 8, 8, 8)
    affine = np.eye(4, dtype=np.float64)

    stored = persist_normalized_model_input(
        storage,
        study_id,
        image=image,
        affine_ras=affine,
    )
    assert storage.verify_checksum(stored.storage_key, stored.sha256)

    reloaded, reloaded_affine = load_prepared_model_input(
        storage,
        {
            "storage_key": stored.storage_key,
            "checksum_sha256": stored.sha256,
        },
    )
    assert np.array_equal(reloaded, image)
    assert np.array_equal(reloaded_affine, affine)


def test_step4_service_is_idempotent_and_never_creates_analysis_run(
    session,
    tmp_path,
    monkeypatch,
):
    study = make_study(session, "GBM-P6-S4-READY")
    storage = LocalObjectStore(
        tmp_path / "storage",
        max_object_bytes=64 * 1024 * 1024,
    )
    geometry = dict(study.segmentation_preparation_summary["model_geometry"])
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_input.get_segmentation_model_geometry",
        lambda study: geometry,
    )

    image = np.zeros((4, 8, 8, 8), dtype=np.float32)
    for index in range(4):
        image[index, 1:7, 1:7, 1:7] = (
            np.arange(216, dtype=np.float32).reshape(6, 6, 6) + index
        )
    stats = [
        {
            "sequence": sequence,
            "nonzero_voxels": 216,
            "mean_before": 100.0,
            "std_before": 10.0,
            "mean_after": 0.0,
            "std_after": 1.0,
        }
        for sequence in BUNDLE_INPUT_CHANNEL_ORDER
    ]
    calls = {"count": 0}

    def build(storage_arg, model_geometry_arg):
        calls["count"] += 1
        return image.copy(), np.eye(4, dtype=np.float64), stats

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_input.build_normalized_model_input",
        build,
    )

    before = session.scalar(select(func.count()).select_from(AnalysisRun))
    first = prepare_segmentation_model_input(session, storage, study)
    middle = session.scalar(select(func.count()).select_from(AnalysisRun))
    second = prepare_segmentation_model_input(session, storage, study)
    after = session.scalar(select(func.count()).select_from(AnalysisRun))

    assert first["version"] == MODEL_INPUT_VERSION
    assert first["status"] == "ready"
    assert first["channel_order"] == ["T1C", "T1", "T2", "FLAIR"]
    assert first["intensity_normalization_performed"] is True
    assert first["crop_pad_performed"] is False
    assert first["bundle_runtime_loading_verified_per_environment"] is False
    assert first["model_execution_started"] is False
    assert first["segmentation_generated"] is False
    assert first["physical_volume_generated"] is False
    assert first["anatomical_localization_generated"] is False
    assert first["next_step"] == "phase6_step5_segmentation_inference"
    assert second["storage_key"] == first["storage_key"]
    assert second["checksum_sha256"] == first["checksum_sha256"]
    assert calls["count"] == 1
    assert before == 0
    assert middle == 0
    assert after == 0


def test_step4_blocks_when_step3_geometry_is_not_ready(session, tmp_path, monkeypatch):
    study = make_study(session, "GBM-P6-S4-BLOCKED")
    storage = LocalObjectStore(
        tmp_path / "storage",
        max_object_bytes=64 * 1024 * 1024,
    )

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_input.get_segmentation_model_geometry",
        lambda study: (_ for _ in ()).throw(
            SegmentationModelGeometryPreparationError(
                "SEGMENTATION_MODEL_GEOMETRY_NOT_RUN",
                "synthetic Step 3 not ready",
            )
        ),
    )

    with pytest.raises(
        SegmentationModelInputPreparationError,
        match="synthetic Step 3 not ready",
    ):
        prepare_segmentation_model_input(session, storage, study)

    assert session.scalar(select(func.count()).select_from(AnalysisRun)) == 0
