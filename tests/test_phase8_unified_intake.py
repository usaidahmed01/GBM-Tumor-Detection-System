from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gbm_ai.api.models import Base
from gbm_ai.api.models.analysis import Study, StudyStatus
from gbm_ai.api.models.clinical import Assessment, Patient
from gbm_ai.api.routers.intake import router
from gbm_ai.api.schemas.intake import UnifiedIntakeCreate
from gbm_ai.api.services.intake_workflow import (
    UNIFIED_INTAKE_VERSION,
    create_unified_intake,
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


def test_unified_intake_generates_human_case_reference_and_internal_records(session):
    result = create_unified_intake(
        session,
        UnifiedIntakeCreate(
            mri_date=date(2026, 8, 16),
            age_years=45,
            prior_treatment=False,
            symptoms=["headache"],
        ),
    )

    assert result["version"] == UNIFIED_INTAKE_VERSION
    assert result["case_reference"].startswith("NGAI-")
    assert len(result["case_reference"].split("-")) == 3
    assert result["internal_identifiers_managed_by_system"] is True
    assert result["patient_context_used_as_ml_features"] is False
    assert result["study_status"] == StudyStatus.AWAITING_UPLOAD.value
    assert session.scalar(select(func.count()).select_from(Patient)) == 1
    assert session.scalar(select(func.count()).select_from(Assessment)) == 1
    assert session.scalar(select(func.count()).select_from(Study)) == 1


def test_explicit_case_reference_can_be_reused_without_asking_for_uuid(session):
    first = create_unified_intake(
        session,
        UnifiedIntakeCreate(
            case_reference="local-case-17",
            mri_date=date(2026, 8, 16),
            prior_treatment=False,
        ),
    )
    second = create_unified_intake(
        session,
        UnifiedIntakeCreate(
            case_reference="LOCAL-CASE-17",
            mri_date=date(2026, 8, 16),
            prior_treatment=False,
        ),
    )

    assert first["patient_reused"] is False
    assert second["patient_reused"] is True
    assert first["patient_uuid"] == second["patient_uuid"]
    assert first["study_uuid"] != second["study_uuid"]
    assert session.scalar(select(func.count()).select_from(Patient)) == 1
    assert session.scalar(select(func.count()).select_from(Assessment)) == 2
    assert session.scalar(select(func.count()).select_from(Study)) == 2


def test_unified_intake_router_contract_is_registered():
    paths = {getattr(route, "path", None) for route in router.routes}
    assert "/intake/studies" in paths


def test_invalid_symptom_is_rejected_before_persistence():
    with pytest.raises(ValueError):
        UnifiedIntakeCreate(
            mri_date=date(2026, 8, 16),
            prior_treatment=False,
            symptoms=["made_up_symptom"],
        )
