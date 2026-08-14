from __future__ import annotations

from datetime import date
from types import SimpleNamespace

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
from gbm_ai.api.segmentation.model_geometry import (
    MODEL_GEOMETRY_VERSION,
    create_isotropic_reference,
    loaded_volume_to_sitk,
    resample_to_reference,
    sitk_affine_ras,
    sitk_to_numpy_xyz,
)
from gbm_ai.api.segmentation.volume_loading import LoadedSegmentationVolume
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.services.segmentation_model_geometry import (
    SegmentationModelGeometryPreparationError,
    get_segmentation_model_geometry,
    prepare_segmentation_model_geometry,
)
from gbm_ai.api.storage.local import StoredObject


class FakeStore:
    def __init__(self):
        self.keys: set[str] = set()

    def verify_checksum(self, storage_key: str, expected_sha256: str) -> bool:
        return storage_key in self.keys

    def exists(self, storage_key: str) -> bool:
        return storage_key in self.keys

    def delete(self, storage_key: str) -> None:
        self.keys.discard(storage_key)


class FakeReferenceImage:
    def GetSize(self):
        return (16, 16, 16)

    def GetSpacing(self):
        return (1.0, 1.0, 1.0)


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
        "study_uuid": str(study.id),
        "status": "ready",
        "source_checksum_sha256": "a" * 64,
        "channel_order": ["T1C", "T1", "T2", "FLAIR"],
        "model_execution_started": False,
        "segmentation_generated": False,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
    }
    session.commit()
    return study


def fake_volume(sequence: str, *, shift: float = 0.0):
    data = np.arange(16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16)
    affine = np.eye(4, dtype=np.float64)
    affine[0, 3] = shift
    return LoadedSegmentationVolume(
        sequence=sequence,
        source_kind="nifti_volume",
        source_reference="synthetic",
        data=data,
        affine_ras=affine,
        orientation_codes=("R", "A", "S"),
    )


def install_lightweight_service_fakes(monkeypatch, store: FakeStore, *, mismatch_sequence=None):
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.build_segmentation_preflight",
        lambda db, study: {
            "channels": [
                {
                    "sequence": sequence,
                    "source_kind": "nifti_volume",
                    "volume_index": index,
                }
                for index, sequence in enumerate(("T1C", "T1", "T2", "FLAIR"))
            ]
        },
    )
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.prepare_segmentation_volumes",
        lambda db, storage, study: dict(study.segmentation_preparation_summary),
    )
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.load_channel_volume",
        lambda db, storage, study, plan: fake_volume(
            plan["sequence"],
            shift=3.0 if plan["sequence"] == mismatch_sequence else 0.0,
        ),
    )
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.loaded_volume_to_sitk",
        lambda volume: SimpleNamespace(sequence=volume.sequence),
    )
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.create_isotropic_reference",
        lambda image: FakeReferenceImage(),
    )
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.sitk_affine_ras",
        lambda image: np.eye(4, dtype=np.float64),
    )
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.resample_to_reference",
        lambda moving, reference, transform=None: SimpleNamespace(
            sequence=getattr(moving, "sequence", "T1C")
        ),
    )

    def persist(storage, study_uuid, *, sequence, image):
        key = f"studies/{study_uuid}/derived/segmentation_model_geometry/{sequence}.nii.gz"
        store.keys.add(key)
        stored = StoredObject(
            storage_key=key,
            sha256=(sequence.lower() * 64)[:64],
            size_bytes=1234,
        )
        return stored, {
            "sequence": sequence,
            "storage_key": key,
            "checksum_sha256": stored.sha256,
            "size_bytes": 1234,
            "shape": [16, 16, 16],
            "spacing_mm": [1.0, 1.0, 1.0],
            "affine_ras": np.eye(4).tolist(),
            "dtype": "float32",
            "interpolation": "linear",
        }

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.persist_resampled_nifti",
        persist,
    )

    def geometry_match(reference, moving):
        return moving.sequence != mismatch_sequence

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.geometries_match",
        geometry_match,
    )

    registration = SimpleNamespace(
        as_dict=lambda: {
            "performed": True,
            "metric": "mattes_mutual_information",
            "final_metric_value": -0.5,
            "optimizer_stop_condition": "synthetic",
            "transform": "rigid_euler3d",
            "max_sampled_displacement_mm": 3.0,
            "deterministic_seed": 42,
        }
    )
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.register_rigid_to_reference",
        lambda fixed, moving: ("synthetic-transform", registration),
    )
    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.identity_registration_summary",
        lambda: SimpleNamespace(
            as_dict=lambda: {
                "performed": False,
                "metric": None,
                "final_metric_value": None,
                "optimizer_stop_condition": None,
                "transform": None,
                "max_sampled_displacement_mm": 0.0,
                "deterministic_seed": None,
            }
        ),
    )


def test_step3_aligned_case_persists_four_1mm_channels_without_inference(
    session,
    monkeypatch,
):
    study = make_study(session, "GBM-P6-S3-ALIGNED")
    store = FakeStore()
    install_lightweight_service_fakes(monkeypatch, store)

    before = session.scalar(select(func.count()).select_from(AnalysisRun))
    result = prepare_segmentation_model_geometry(session, store, study)
    after = session.scalar(select(func.count()).select_from(AnalysisRun))

    assert result["version"] == MODEL_GEOMETRY_VERSION
    assert result["status"] == "ready"
    assert result["channel_order"] == ["T1C", "T1", "T2", "FLAIR"]
    assert result["target_spacing_mm"] == [1.0, 1.0, 1.0]
    assert len(result["channels"]) == 4
    assert result["registration_performed"] is False
    assert result["reference_geometry_resampling_performed"] is True
    assert result["intensity_normalization_performed"] is False
    assert result["model_execution_started"] is False
    assert result["segmentation_generated"] is False
    assert result["physical_volume_generated"] is False
    assert result["anatomical_localization_generated"] is False
    assert before == 0
    assert after == 0


def test_step3_registers_only_mismatched_channel_and_keeps_inference_stopped(
    session,
    monkeypatch,
):
    study = make_study(session, "GBM-P6-S3-REG")
    store = FakeStore()
    install_lightweight_service_fakes(monkeypatch, store, mismatch_sequence="FLAIR")

    result = prepare_segmentation_model_geometry(session, store, study)

    assert result["registration_performed"] is True
    assert result["registration_method"] == "rigid_euler3d_mattes_mutual_information"
    flair = next(item for item in result["channels"] if item["sequence"] == "FLAIR")
    t2 = next(item for item in result["channels"] if item["sequence"] == "T2")
    assert flair["registration"]["performed"] is True
    assert t2["registration"]["performed"] is False
    assert study.segmentation_preparation_status == SegmentationPreparationStatus.READY
    assert study.segmentation_preparation_summary["model_execution_started"] is False


def test_step3_existing_verified_artifacts_are_idempotent(session, monkeypatch):
    study = make_study(session, "GBM-P6-S3-IDEMPOTENT")
    store = FakeStore()
    install_lightweight_service_fakes(monkeypatch, store)

    first = prepare_segmentation_model_geometry(session, store, study)

    def should_not_load(*args, **kwargs):
        raise AssertionError("verified Step 3 artifacts should be reused")

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.load_channel_volume",
        should_not_load,
    )
    second = prepare_segmentation_model_geometry(session, store, study)
    assert second == first


def test_step3_failure_persists_safe_non_inference_state(session, monkeypatch):
    study = make_study(session, "GBM-P6-S3-FAIL")
    store = FakeStore()
    install_lightweight_service_fakes(monkeypatch, store, mismatch_sequence="T2")

    def fail_registration(fixed, moving):
        error = SegmentationModelGeometryPreparationError(
            "SYNTHETIC_REGISTRATION_FAILURE",
            "synthetic failure",
        )
        error.code = "SYNTHETIC_REGISTRATION_FAILURE"
        raise error

    monkeypatch.setattr(
        "gbm_ai.api.services.segmentation_model_geometry.register_rigid_to_reference",
        fail_registration,
    )

    # The service intentionally normalizes known registration failures into its
    # own safe error response and persists no inference state.
    with pytest.raises(ValueError):
        prepare_segmentation_model_geometry(session, store, study)

    current = study.segmentation_preparation_summary
    assert current["model_execution_started"] is False
    assert current["segmentation_generated"] is False


def test_get_model_geometry_requires_step3_state(session):
    study = make_study(session, "GBM-P6-S3-NOTRUN")
    with pytest.raises(
        SegmentationModelGeometryPreparationError,
        match="has not been completed",
    ):
        get_segmentation_model_geometry(study)


def test_real_simpleitk_conversion_and_1mm_resampling_when_dependency_available():
    pytest.importorskip("SimpleITK")

    data = np.arange(10 * 11 * 12, dtype=np.float32).reshape(10, 11, 12)
    affine = np.diag([2.0, 1.5, 2.5, 1.0])
    volume = LoadedSegmentationVolume(
        sequence="T1C",
        source_kind="nifti_volume",
        source_reference="synthetic",
        data=data,
        affine_ras=affine,
        orientation_codes=("R", "A", "S"),
    )

    sitk_image = loaded_volume_to_sitk(volume)
    reference = create_isotropic_reference(sitk_image)
    output = resample_to_reference(sitk_image, reference)
    result = sitk_to_numpy_xyz(output)
    result_affine = sitk_affine_ras(output)

    assert result.ndim == 3
    assert np.isfinite(result).all()
    assert np.allclose(np.linalg.norm(result_affine[:3, :3], axis=0), (1.0, 1.0, 1.0))
    assert tuple(output.GetSpacing()) == (1.0, 1.0, 1.0)
