from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.config import Settings
from gbm_ai.api.dependencies import get_db_session
from gbm_ai.api.main import create_app
from gbm_ai.api.models import Base
from gbm_ai.api.models.clinical import ScopeStatus
from gbm_ai.api.schemas.clinical import AssessmentCreate, PatientCreate
from gbm_ai.api.services.clinical_records import create_assessment, create_patient


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


def test_patient_and_assessment_persist(session):
    patient = create_patient(
        session,
        PatientCreate(patient_id="GBM-2026-0001", age_years=55, sex="male"),
    )
    assessment = create_assessment(
        session,
        AssessmentCreate(
            patient_uuid=patient.id,
            mri_date=date(2026, 8, 14),
            symptoms=["headache", "weakness"],
            prior_treatment=False,
        ),
    )

    assert assessment.patient_id == patient.id
    assert assessment.scope_status == ScopeStatus.IN_SCOPE
    assert assessment.symptoms == ["headache", "weakness"]


def test_prior_treatment_marks_case_out_of_v1_scope(session):
    patient = create_patient(
        session,
        PatientCreate(patient_id="GBM-2026-0002", age_years=63),
    )
    assessment = create_assessment(
        session,
        AssessmentCreate(
            patient_uuid=patient.id,
            mri_date=date(2026, 8, 14),
            prior_treatment=True,
        ),
    )
    assert assessment.scope_status == ScopeStatus.OUT_OF_SCOPE_PRIOR_TREATMENT


def test_age_validation_matches_documented_v1_adult_range():
    with pytest.raises(Exception):
        PatientCreate(patient_id="GBM-2026-0003", age_years=17)


def test_api_create_patient_and_assessment(session, monkeypatch):
    settings = Settings(environment="test")
    app = create_app(settings)

    class FakeDatabase:
        def dispose(self):
            pass

    monkeypatch.setattr(
        "gbm_ai.api.main.DatabaseManager",
        lambda settings: FakeDatabase(),
    )

    def override_session():
        yield session

    app.dependency_overrides[get_db_session] = override_session

    with TestClient(app) as client:
        patient_response = client.post(
            "/api/v1/patients",
            json={
                "patient_id": "GBM-2026-0100",
                "patient_name": "Synthetic Demo",
                "age_years": 48,
                "sex": "female",
                "privacy_flags": {"synthetic": True},
            },
        )
        assert patient_response.status_code == 201
        patient = patient_response.json()

        assessment_response = client.post(
            "/api/v1/assessments",
            json={
                "patient_uuid": patient["id"],
                "mri_date": "2026-08-14",
                "symptoms": ["headache", "seizure"],
                "symptom_duration": "3 weeks",
                "prior_treatment": False,
                "clinical_notes": "Synthetic demo context only.",
            },
        )
        assert assessment_response.status_code == 201
        assessment = assessment_response.json()
        assert assessment["scope_status"] == "in_scope"
        assert assessment["status"] == "ready_for_upload"


def test_clinical_context_policy_explicitly_excludes_metadata_from_ml(
    session, monkeypatch
):
    settings = Settings(environment="test")
    app = create_app(settings)

    class FakeDatabase:
        def dispose(self):
            pass

    monkeypatch.setattr(
        "gbm_ai.api.main.DatabaseManager",
        lambda settings: FakeDatabase(),
    )

    def override_session():
        yield session

    app.dependency_overrides[get_db_session] = override_session

    with TestClient(app) as client:
        response = client.get("/api/v1/clinical-context-policy")

    assert response.status_code == 200
    body = response.json()
    assert body["ml_input_policy"] == "MRI_ONLY_V1"
    assert body["patient_id_used_as_ml_feature"] is False
    assert body["patient_name_used_as_ml_feature"] is False
    assert body["age_used_as_ml_feature"] is False
    assert body["sex_used_as_ml_feature"] is False
    assert body["symptoms_used_as_ml_feature"] is False
    assert body["symptom_duration_used_as_ml_feature"] is False
    assert body["prior_treatment_used_as_ml_feature"] is False
    assert body["clinical_notes_used_as_ml_feature"] is False
