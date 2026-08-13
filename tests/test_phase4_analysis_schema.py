from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import (
    AnalysisStatus,
    DecisionState,
    ModelRole,
    QCState,
    SourceFormat,
    StudyStatus,
)
from gbm_ai.api.schemas.analysis import (
    AnalysisRunCreate,
    ModelVersionCreate,
    StudyCreate,
)
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.services.analysis_records import (
    ModelRoleMismatchError,
    create_analysis_run,
    create_model_version,
    create_study,
)
from gbm_ai.api.services.clinical_records import (
    create_assessment,
    create_patient,
)


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


def make_assessment(session):
    patient = create_patient(
        session,
        PatientCreate(
            patient_id="GBM-2026-0200",
            age_years=51,
            sex="female",
            privacy_flags={"synthetic": True},
        ),
    )
    return create_assessment(
        session,
        AssessmentCreate(
            patient_uuid=patient.id,
            mri_date=date(2026, 8, 14),
            symptoms=["headache"],
            prior_treatment=False,
        ),
    )


def test_study_starts_pending_for_backend_auto_detection(session):
    assessment = make_assessment(session)
    study = create_study(
        session,
        StudyCreate(assessment_uuid=assessment.id),
    )

    assert study.assessment_id == assessment.id
    assert study.source_format == SourceFormat.PENDING
    assert study.status == StudyStatus.AWAITING_UPLOAD
    assert study.modality == "MRI"
    assert study.storage_key is None


def test_model_version_and_analysis_run_traceability(session):
    assessment = make_assessment(session)
    study = create_study(
        session,
        StudyCreate(assessment_uuid=assessment.id),
    )

    classifier = create_model_version(
        session,
        ModelVersionCreate(
            model_name="gbm_classifier",
            version="phase3-development",
            role="classifier",
            architecture="EfficientNetV2-S",
            weights_checksum_sha256="a" * 64,
            code_version="test-commit",
            preprocessing_version="classification_v1.0-384",
            threshold_version="classifier_safety_fusion_v1",
            calibration_version="cross_fitted_temperature_v1",
            license_source_notes="Synthetic test metadata.",
            is_active=False,
        ),
    )

    run = create_analysis_run(
        session,
        AnalysisRunCreate(
            study_uuid=study.id,
            classifier_model_version_uuid=classifier.id,
        ),
    )

    assert classifier.role == ModelRole.CLASSIFIER
    assert run.study_id == study.id
    assert run.classifier_model_version_id == classifier.id
    assert run.status == AnalysisStatus.PENDING
    assert run.qc_state == QCState.PENDING
    assert run.decision_state == DecisionState.PENDING


def test_analysis_run_rejects_segmentation_model_in_classifier_slot(session):
    assessment = make_assessment(session)
    study = create_study(
        session,
        StudyCreate(assessment_uuid=assessment.id),
    )

    segmentation = create_model_version(
        session,
        ModelVersionCreate(
            model_name="glioma_segmentation",
            version="test",
            role="segmentation",
            architecture="SegResNet",
            weights_checksum_sha256="b" * 64,
            code_version="test",
            preprocessing_version="test",
            is_active=False,
        ),
    )

    with pytest.raises(ModelRoleMismatchError):
        create_analysis_run(
            session,
            AnalysisRunCreate(
                study_uuid=study.id,
                classifier_model_version_uuid=segmentation.id,
            ),
        )


def test_model_version_checksum_must_be_sha256_length():
    with pytest.raises(Exception):
        ModelVersionCreate(
            model_name="bad",
            version="v1",
            role="classifier",
            architecture="EfficientNetV2-S",
            weights_checksum_sha256="too-short",
            code_version="x",
            preprocessing_version="x",
        )
