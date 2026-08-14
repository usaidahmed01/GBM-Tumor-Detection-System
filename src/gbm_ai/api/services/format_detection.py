from __future__ import annotations

from sqlalchemy.orm import Session

from gbm_ai.api.models.analysis import (
    DeidentificationStatus,
    SourceFormat,
    Study,
    StudyStatus,
)
from gbm_ai.api.storage.local import LocalObjectStore
from gbm_ai.api.upload.format_detection import (
    AmbiguousArchiveError,
    DetectedFormat,
    NonMRIInputError,
    UnsupportedInputError,
    detect_stored_object,
)


class StudyHasNoStoredObjectError(ValueError):
    pass


def _safe_detection_metadata(detected: DetectedFormat) -> dict:
    return {
        "format_detection": {
            "status": "complete",
            "source_format": detected.source_format,
            "parser": detected.parser,
            "modality": detected.modality,
            "technical_metadata": detected.technical_metadata,
            "phi_persisted": False,
            "original_dicom_uids_persisted": False,
        }
    }


def detect_and_update_study_format(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
) -> DetectedFormat:
    if not study.storage_key:
        raise StudyHasNoStoredObjectError(
            "study has no stored source object"
        )

    try:
        with storage.open_read(study.storage_key) as source:
            detected = detect_stored_object(source)
    except NonMRIInputError as exc:
        study.source_format = SourceFormat.DICOM
        study.modality = exc.modality
        study.status = StudyStatus.FAILED
        study.deidentification_status = DeidentificationStatus.FAILED
        study.deidentified_metadata = {
            "format_detection": {
                "status": "failed",
                "reason": "non_mri_dicom_modality",
                "modality": exc.modality,
                "phi_persisted": False,
            }
        }
        db.commit()
        db.refresh(study)
        raise
    except (UnsupportedInputError, AmbiguousArchiveError) as exc:
        study.status = StudyStatus.FAILED
        study.deidentification_status = DeidentificationStatus.FAILED
        study.deidentified_metadata = {
            "format_detection": {
                "status": "failed",
                "reason": exc.__class__.__name__,
                "phi_persisted": False,
            }
        }
        db.commit()
        db.refresh(study)
        raise

    study.source_format = SourceFormat(detected.source_format)
    study.modality = detected.modality
    study.deidentified_metadata = _safe_detection_metadata(detected)

    if study.source_format == SourceFormat.DICOM:
        study.deidentification_status = DeidentificationStatus.PENDING
    else:
        study.deidentification_status = DeidentificationStatus.NOT_APPLICABLE

    study.status = StudyStatus.UPLOADED

    db.commit()
    db.refresh(study)
    return detected
