from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import (
    AnalysisRun,
    AnalysisStatus,
    DecisionState,
    ModelVersion,
    QCState,
    SourceFormat,
    Study,
    StudyStatus,
)
from gbm_ai.api.models.clinical import Assessment
from gbm_ai.api.schemas.analysis import (
    AnalysisRunCreate,
    ModelVersionCreate,
    StudyCreate,
)


class AssessmentNotFoundForStudyError(Exception):
    pass


class StudyNotFoundError(Exception):
    pass


class ModelVersionNotFoundError(Exception):
    pass


class DuplicateModelVersionError(Exception):
    pass


class ModelRoleMismatchError(Exception):
    pass


def create_study(db: Session, payload: StudyCreate) -> Study:
    assessment = db.get(Assessment, payload.assessment_uuid)
    if assessment is None:
        raise AssessmentNotFoundForStudyError(str(payload.assessment_uuid))

    study = Study(
        assessment_id=assessment.id,
        source_format=SourceFormat.PENDING,
        modality="MRI",
        deidentified_metadata={},
        status=StudyStatus.AWAITING_UPLOAD,
    )
    db.add(study)
    db.commit()
    db.refresh(study)
    return study


def get_study(db: Session, study_uuid: uuid.UUID) -> Study:
    study = db.get(Study, study_uuid)
    if study is None:
        raise StudyNotFoundError(str(study_uuid))
    return study


def create_model_version(
    db: Session,
    payload: ModelVersionCreate,
) -> ModelVersion:
    model_version = ModelVersion(
        model_name=payload.model_name,
        version=payload.version,
        role=payload.role,
        architecture=payload.architecture,
        weights_checksum_sha256=payload.weights_checksum_sha256,
        code_version=payload.code_version,
        preprocessing_version=payload.preprocessing_version,
        threshold_version=payload.threshold_version,
        calibration_version=payload.calibration_version,
        license_source_notes=payload.license_source_notes,
        is_active=payload.is_active,
    )
    db.add(model_version)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise DuplicateModelVersionError(
            f"{payload.model_name}:{payload.version}"
        ) from exc
    db.refresh(model_version)
    return model_version


def get_model_version(
    db: Session,
    model_version_uuid: uuid.UUID,
) -> ModelVersion:
    model_version = db.get(ModelVersion, model_version_uuid)
    if model_version is None:
        raise ModelVersionNotFoundError(str(model_version_uuid))
    return model_version


def create_analysis_run(
    db: Session,
    payload: AnalysisRunCreate,
) -> AnalysisRun:
    study = get_study(db, payload.study_uuid)

    classifier = None
    if payload.classifier_model_version_uuid is not None:
        classifier = get_model_version(
            db, payload.classifier_model_version_uuid
        )
        if classifier.role.value != "classifier":
            raise ModelRoleMismatchError(
                "classifier_model_version_uuid must reference classifier role"
            )

    segmentation = None
    if payload.segmentation_model_version_uuid is not None:
        segmentation = get_model_version(
            db, payload.segmentation_model_version_uuid
        )
        if segmentation.role.value != "segmentation":
            raise ModelRoleMismatchError(
                "segmentation_model_version_uuid must reference segmentation role"
            )

    run = AnalysisRun(
        study_id=study.id,
        classifier_model_version_id=classifier.id if classifier else None,
        segmentation_model_version_id=segmentation.id if segmentation else None,
        status=AnalysisStatus.PENDING,
        qc_state=QCState.PENDING,
        decision_state=DecisionState.PENDING,
        safety_reason_codes=[],
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_analysis_run(
    db: Session,
    analysis_run_uuid: uuid.UUID,
) -> AnalysisRun:
    run = db.get(AnalysisRun, analysis_run_uuid)
    if run is None:
        raise StudyNotFoundError(str(analysis_run_uuid))
    return run


def list_model_versions(db: Session) -> list[ModelVersion]:
    return list(
        db.scalars(
            select(ModelVersion).order_by(
                ModelVersion.model_name,
                ModelVersion.version,
            )
        )
    )
