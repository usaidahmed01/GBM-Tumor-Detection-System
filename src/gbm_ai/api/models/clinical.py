from __future__ import annotations

import enum
import uuid
from datetime import date

from sqlalchemy import Boolean, Date, Enum, ForeignKey, Index, Integer, JSON, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gbm_ai.api.models.base import Base, TimestampMixin


class Sex(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class AssessmentStatus(str, enum.Enum):
    DRAFT = "draft"
    READY_FOR_UPLOAD = "ready_for_upload"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class ScopeStatus(str, enum.Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE_PRIOR_TREATMENT = "out_of_scope_prior_treatment"


class Patient(TimestampMixin, Base):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    patient_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    age_years: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sex: Mapped[Sex] = mapped_column(
        Enum(Sex, name="patient_sex", native_enum=False, validate_strings=True),
        nullable=False,
        default=Sex.UNKNOWN,
    )
    privacy_flags: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    assessments: Mapped[list["Assessment"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )


class Assessment(TimestampMixin, Base):
    __tablename__ = "assessments"
    __table_args__ = (
        Index("ix_assessments_patient_created", "patient_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    mri_date: Mapped[date] = mapped_column(Date, nullable=False)
    symptoms: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    symptom_duration: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prior_treatment: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    clinical_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[AssessmentStatus] = mapped_column(
        Enum(AssessmentStatus, name="assessment_status", native_enum=False, validate_strings=True),
        nullable=False,
        default=AssessmentStatus.DRAFT,
    )
    scope_status: Mapped[ScopeStatus] = mapped_column(
        Enum(ScopeStatus, name="assessment_scope_status", native_enum=False, validate_strings=True),
        nullable=False,
        default=ScopeStatus.IN_SCOPE,
    )

    patient: Mapped[Patient] = relationship(back_populates="assessments")
