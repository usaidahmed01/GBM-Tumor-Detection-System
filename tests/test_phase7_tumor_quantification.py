from __future__ import annotations

from datetime import date, datetime, timezone
from io import BytesIO

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
    DecisionState,
    QCState,
    SegmentationPreparationStatus,
    SourceFormat,
    StudyQCStatus,
    StudyStatus,
)
from gbm_ai.api.models.quantification import TumorQuantification
from gbm_ai.api.models.segmentation import (
    Segmentation,
    SegmentationReviewStatus,
    SegmentationStatus,
)
from gbm_ai.api.quantification import measure_region, validate_physical_geometry
from gbm_ai.api.schemas.analysis import StudyCreate
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.services.analysis_records import create_study
from gbm_ai.api.services.clinical_records import create_assessment, create_patient
from gbm_ai.api.services.tumor_quantification import (
    TumorQuantificationServiceError,
    run_tumor_quantification,
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


def _stored(storage: LocalObjectStore, study_id, name: str):
    key = storage.generate_study_derived_key(study_id, f"test_{name}", suffix=".nii.gz")
    return storage.put_stream(key, BytesIO(f"synthetic-{name}".encode()))


def make_current_segmentation(session: Session, storage: LocalObjectStore, patient_id: str):
    patient = create_patient(
        session,
        PatientCreate(patient_id=patient_id, age_years=51, privacy_flags={"synthetic": True}),
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
    study.source_format = SourceFormat.NIFTI
    study.status = StudyStatus.READY_FOR_ANALYSIS
    study.qc_status = StudyQCStatus.PASS
    study.capability_routing_status = CapabilityRoutingStatus.READY
    study.segmentation_preparation_status = SegmentationPreparationStatus.READY

    analysis = AnalysisRun(
        study_id=study.id,
        status=AnalysisStatus.COMPLETE,
        qc_state=QCState.PASS,
        decision_state=DecisionState.PENDING,
        safety_reason_codes=[],
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    session.add(analysis)
    session.flush()

    wt = _stored(storage, study.id, "wt")
    tc = _stored(storage, study.id, "tc")
    et = _stored(storage, study.id, "et")
    labelmap = _stored(storage, study.id, "labelmap")
    segmentation = Segmentation(
        analysis_run_id=analysis.id,
        status=SegmentationStatus.GENERATED,
        model_input_checksum_sha256="a" * 64,
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
        spatial_shape=[4, 4, 3],
        affine_ras=np.eye(4).tolist(),
        tc_storage_key=tc.storage_key,
        tc_checksum_sha256=tc.sha256,
        tc_size_bytes=tc.size_bytes,
        wt_storage_key=wt.storage_key,
        wt_checksum_sha256=wt.sha256,
        wt_size_bytes=wt.size_bytes,
        et_storage_key=et.storage_key,
        et_checksum_sha256=et.sha256,
        et_size_bytes=et.size_bytes,
        labelmap_storage_key=labelmap.storage_key,
        labelmap_checksum_sha256=labelmap.sha256,
        labelmap_size_bytes=labelmap.size_bytes,
        voxel_counts={"WT": 8, "TC": 4, "ET": 1, "LABELMAP_NONZERO": 8},
        runtime_seconds=0.1,
        review_status=SegmentationReviewStatus.UNREVIEWED,
        clinician_modified=False,
        physical_volume_generated=False,
        anatomical_localization_generated=False,
        clinical_validation_claimed=False,
    )
    session.add(segmentation)
    session.flush()

    study.segmentation_preparation_summary = {
        "model_input": {
            "status": "ready",
            "checksum_sha256": "a" * 64,
        },
        "inference": {
            "status": "complete",
            "segmentation_uuid": str(segmentation.id),
            "segmentation_generated": True,
            "physical_volume_generated": False,
            "anatomical_localization_generated": False,
        },
        "segmentation_generated": True,
        "physical_volume_generated": False,
        "anatomical_localization_generated": False,
    }
    session.commit()
    return study, analysis, segmentation


def fake_masks():
    wt = np.zeros((4, 4, 3), dtype=np.uint8)
    tc = np.zeros_like(wt)
    et = np.zeros_like(wt)
    wt[0:2, 0:2, 0:2] = 1  # 8 voxels; 4 mm2 max axial area at identity geometry
    tc[0:2, 0:2, 0] = 1    # 4 voxels
    et[0, 0, 0] = 1        # 1 voxel
    return {"WT": wt, "TC": tc, "ET": et}


def install_fake_mask_loader(monkeypatch, segmentation: Segmentation, masks: dict[str, np.ndarray]):
    by_key = {
        segmentation.wt_storage_key: masks["WT"],
        segmentation.tc_storage_key: masks["TC"],
        segmentation.et_storage_key: masks["ET"],
    }

    def fake_loader(storage, *, storage_key, checksum_sha256, expected_shape, expected_affine_ras):
        return np.asarray(by_key[storage_key], dtype=np.uint8)

    monkeypatch.setattr(
        "gbm_ai.api.services.tumor_quantification.load_and_validate_mask",
        fake_loader,
    )


def test_physical_geometry_and_volume_formula_support_anisotropic_voxels():
    affine = np.diag([0.5, 0.5, 2.0, 1.0])
    geometry = validate_physical_geometry(affine)
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[0:2, 0:2, 1] = 1
    measurement, per_slice = measure_region("WT", mask, geometry)

    assert geometry.voxel_spacing_mm == pytest.approx((0.5, 0.5, 2.0))
    assert geometry.voxel_volume_mm3 == pytest.approx(0.5)
    assert geometry.axial_pixel_area_mm2 == pytest.approx(0.25)
    assert measurement.voxel_count == 4
    assert measurement.volume_mm3 == pytest.approx(2.0)
    assert measurement.volume_cm3 == pytest.approx(0.002)
    assert measurement.max_axial_area_mm2 == pytest.approx(1.0)
    assert measurement.max_axial_slice_index == 1
    assert per_slice == [{"slice_index": 1, "foreground_voxels": 4, "area_mm2": 1.0}]


def test_successful_quantification_persists_volume_area_and_safety_boundary(monkeypatch, session, tmp_path):
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)
    study, analysis, segmentation = make_current_segmentation(session, storage, "GBM-P7-S1-SUCCESS")
    install_fake_mask_loader(monkeypatch, segmentation, fake_masks())

    result = run_tumor_quantification(session, storage, study)

    assert result["status"] == "complete"
    assert result["primary_quantitative_region"] == "WT"
    regions = {item["region"]: item for item in result["regions"]}
    assert regions["WT"]["volume_mm3"] == pytest.approx(8.0)
    assert regions["WT"]["volume_cm3"] == pytest.approx(0.008)
    assert regions["WT"]["max_axial_area_mm2"] == pytest.approx(4.0)
    assert regions["TC"]["volume_mm3"] == pytest.approx(4.0)
    assert regions["ET"]["volume_mm3"] == pytest.approx(1.0)
    assert result["physical_volume_generated"] is True
    assert result["anatomical_localization_generated"] is False
    assert result["segmentation_is_gbm_diagnosis"] is False
    assert result["clinical_validation_claimed"] is False

    persisted = session.scalar(select(TumorQuantification))
    assert persisted is not None
    assert persisted.segmentation_id == segmentation.id
    assert persisted.wt_voxel_count == 8
    assert persisted.wt_axial_nonzero_slice_count == 2
    assert storage.verify_checksum(
        persisted.per_slice_area_storage_key,
        persisted.per_slice_area_checksum_sha256,
    )
    session.refresh(segmentation)
    session.refresh(study)
    assert segmentation.physical_volume_generated is True
    assert study.segmentation_preparation_summary["physical_volume_generated"] is True
    assert study.segmentation_preparation_summary["next_step"] == "phase7_step2_anatomical_localization"
    assert analysis.decision_state == DecisionState.PENDING


def test_repeated_quantification_is_idempotent(monkeypatch, session, tmp_path):
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)
    study, _, segmentation = make_current_segmentation(session, storage, "GBM-P7-S1-IDEMPOTENT")
    install_fake_mask_loader(monkeypatch, segmentation, fake_masks())

    first = run_tumor_quantification(session, storage, study)
    second = run_tumor_quantification(session, storage, study)

    assert first["quantification_uuid"] == second["quantification_uuid"]
    assert session.scalar(select(func.count()).select_from(TumorQuantification)) == 1


def test_rejected_segmentation_blocks_physical_volume(monkeypatch, session, tmp_path):
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)
    study, _, segmentation = make_current_segmentation(session, storage, "GBM-P7-S1-REJECT")
    segmentation.review_status = SegmentationReviewStatus.REJECTED
    session.commit()
    install_fake_mask_loader(monkeypatch, segmentation, fake_masks())

    with pytest.raises(TumorQuantificationServiceError, match="rejected segmentation"):
        run_tumor_quantification(session, storage, study)
    assert session.scalar(select(func.count()).select_from(TumorQuantification)) == 0


def test_voxel_count_provenance_mismatch_blocks_measurement(monkeypatch, session, tmp_path):
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)
    study, _, segmentation = make_current_segmentation(session, storage, "GBM-P7-S1-COUNT")
    segmentation.voxel_counts = {"WT": 999, "TC": 4, "ET": 1}
    session.commit()
    install_fake_mask_loader(monkeypatch, segmentation, fake_masks())

    with pytest.raises(TumorQuantificationServiceError, match="foreground count"):
        run_tumor_quantification(session, storage, study)
    assert session.scalar(select(func.count()).select_from(TumorQuantification)) == 0


def test_standalone_image_source_can_never_generate_physical_volume(monkeypatch, session, tmp_path):
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)
    study, _, segmentation = make_current_segmentation(session, storage, "GBM-P7-S1-IMAGE")
    study.source_format = SourceFormat.IMAGE
    session.commit()
    install_fake_mask_loader(monkeypatch, segmentation, fake_masks())

    with pytest.raises(TumorQuantificationServiceError, match="DICOM/NIfTI"):
        run_tumor_quantification(session, storage, study)
    assert session.scalar(select(func.count()).select_from(TumorQuantification)) == 0


def test_labelmap_checksum_guard_fails_before_volume_is_persisted(monkeypatch, session, tmp_path):
    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)
    study, _, segmentation = make_current_segmentation(session, storage, "GBM-P7-S1-CHECKSUM")
    segmentation.labelmap_checksum_sha256 = "0" * 64
    session.commit()
    install_fake_mask_loader(monkeypatch, segmentation, fake_masks())

    with pytest.raises(TumorQuantificationServiceError, match="label map"):
        run_tumor_quantification(session, storage, study)
    assert session.scalar(select(func.count()).select_from(TumorQuantification)) == 0


def test_real_nifti_mask_loader_enforces_affine_and_binary_contract(tmp_path):
    nib = pytest.importorskip("nibabel")
    from gbm_ai.api.quantification import (
        PhysicalQuantificationError,
        load_and_validate_mask,
    )

    storage = LocalObjectStore(tmp_path / "storage", max_object_bytes=64 * 1024 * 1024)
    mask = np.zeros((3, 3, 2), dtype=np.uint8)
    mask[1, 1, 1] = 1
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    image_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), str(image_path))
    key = storage.generate_study_derived_key(__import__("uuid").uuid4(), "test_real_mask", suffix=".nii.gz")
    with image_path.open("rb") as source:
        stored = storage.put_stream(key, source)

    loaded = load_and_validate_mask(
        storage,
        storage_key=stored.storage_key,
        checksum_sha256=stored.sha256,
        expected_shape=(3, 3, 2),
        expected_affine_ras=affine,
    )
    assert int(loaded.sum()) == 1

    with pytest.raises(PhysicalQuantificationError, match="physical geometry"):
        load_and_validate_mask(
            storage,
            storage_key=stored.storage_key,
            checksum_sha256=stored.sha256,
            expected_shape=(3, 3, 2),
            expected_affine_ras=np.diag([2.0, 1.0, 1.0, 1.0]),
        )
