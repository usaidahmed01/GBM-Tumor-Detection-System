from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from gbm_ai.api.dicom.deidentify import (
    DicomDeidentificationError,
    DicomGroupingError,
    DicomModalityError,
    DicomPixelPrivacyRiskError,
    DicomProcessingError,
    build_deidentified_dicom_package,
)
from gbm_ai.api.models.analysis import (
    DeidentificationStatus,
    Series,
    SourceFormat,
    Study,
    StudyStatus,
)
from gbm_ai.api.models.audit import (
    AuditAction,
    AuditActorType,
    AuditEntityType,
)
from gbm_ai.api.services.audit import record_audit_event
from gbm_ai.api.storage.local import LocalObjectStore


class DicomStudyStateError(ValueError):
    pass


def process_dicom_study(
    db: Session,
    storage: LocalObjectStore,
    study: Study,
    *,
    request_id: str | None = None,
    actor_type: AuditActorType = AuditActorType.DEMO_USER,
    actor_id: str | None = None,
) -> dict:
    if study.source_format != SourceFormat.DICOM:
        raise DicomStudyStateError(
            "DICOM processing requires study.source_format == dicom"
        )
    if not study.storage_key:
        raise DicomStudyStateError(
            "study has no protected source object"
        )
    if study.deidentified_storage_key:
        raise DicomStudyStateError(
            "study already has a de-identified working object"
        )

    try:
        with storage.open_read(study.storage_key) as source:
            package = build_deidentified_dicom_package(source)
    except DicomPixelPrivacyRiskError as exc:
        study.deidentification_status = (
            DeidentificationStatus.BLOCKED_PIXEL_PHI_RISK
        )
        study.status = StudyStatus.FAILED

        metadata = dict(study.deidentified_metadata or {})
        metadata["dicom_deidentification"] = {
            "status": "blocked",
            "reason": "pixel_phi_risk_flag",
            "detail": str(exc),
            "ps3_15_profile_compliance_claimed": False,
            "ai_working_copy_created": False,
        }
        study.deidentified_metadata = metadata
        db.commit()
        db.refresh(study)
        raise
    except (DicomGroupingError, DicomModalityError, DicomProcessingError) as exc:
        study.deidentification_status = DeidentificationStatus.FAILED
        study.status = StudyStatus.FAILED

        metadata = dict(study.deidentified_metadata or {})
        metadata["dicom_deidentification"] = {
            "status": "failed",
            "reason": exc.__class__.__name__,
            "ps3_15_profile_compliance_claimed": False,
            "ai_working_copy_created": False,
        }
        study.deidentified_metadata = metadata
        db.commit()
        db.refresh(study)
        raise

    stored = None
    try:
        package.output_stream.seek(0)
        key = storage.generate_study_derived_key(
            study.id,
            "dicom-deidentified",
            suffix=".zip",
        )
        stored = storage.put_stream(
            key,
            package.output_stream,
        )

        # Reprocessing is not allowed after a working object exists, but this
        # cleanup makes retries after a prior DB-only partial attempt safe.
        db.execute(
            delete(Series).where(Series.study_id == study.id)
        )

        for record in package.series_records:
            db.add(
                Series(
                    study_id=study.id,
                    series_uid=record["series_uid"],
                    series_number=record["series_number"],
                    detected_sequence=record["detected_sequence"],
                    confirmed_sequence=record["confirmed_sequence"],
                    sequence_confidence=record["sequence_confidence"],
                    sequence_metadata=record["sequence_metadata"],
                    slice_count=record["slice_count"],
                    spacing_orientation_metadata=(
                        record["spacing_orientation_metadata"]
                    ),
                    working_member_prefix=record["working_member_prefix"],
                )
            )

        study.study_instance_uid = package.study_uid
        study.modality = "MR"
        study.deidentified_storage_key = stored.storage_key
        study.deidentified_checksum_sha256 = stored.sha256
        study.deidentification_status = (
            DeidentificationStatus.METADATA_DEIDENTIFIED
        )

        metadata = dict(study.deidentified_metadata or {})
        metadata["dicom_deidentification"] = {
            "status": "metadata_deidentified",
            "input_instance_count": package.input_instance_count,
            "output_instance_count": package.output_instance_count,
            "series_count": len(package.series_records),
            "ignored_non_dicom_entries": package.ignored_non_dicom_entries,
            "private_tags_removed": package.private_tags_removed,
            "free_text_removed": package.free_text_removed,
            "uids_remapped": package.uid_remapping_applied,
            "original_uids_persisted": False,
            "pixel_data_modified": package.pixel_data_modified,
            "pixel_privacy_status": package.pixel_privacy_status,
            "ps3_15_profile_compliance_claimed": False,
            "ai_working_copy_created": True,
        }
        study.deidentified_metadata = metadata

        # QC/sequence detection are still pending, so do not mark ready.
        study.status = StudyStatus.UPLOADED

        record_audit_event(
            db,
            action=AuditAction.STUDY_SOURCE_STORED,
            entity_type=AuditEntityType.STUDY,
            entity_uuid=study.id,
            actor_type=actor_type,
            actor_id=actor_id,
            request_id=request_id,
            technical_context={
                "operation": "dicom_deidentified_working_copy",
                "status": "metadata_deidentified",
                "storage_backend": "local",
                "size_bytes": stored.size_bytes,
                "sha256": stored.sha256,
                "result": "success",
            },
            commit=False,
        )

        db.commit()
        db.refresh(study)

        return {
            "deidentified_storage_key": stored.storage_key,
            "deidentified_sha256": stored.sha256,
            "deidentified_size_bytes": stored.size_bytes,
            "series_count": len(package.series_records),
            "instance_count": package.output_instance_count,
            "pixel_privacy_status": package.pixel_privacy_status,
        }
    except Exception:
        db.rollback()
        if stored is not None and storage.exists(stored.storage_key):
            storage.delete(stored.storage_key)
        raise
    finally:
        package.output_stream.close()


def list_study_series(
    db: Session,
    study: Study,
) -> list[Series]:
    return list(
        db.scalars(
            select(Series)
            .where(Series.study_id == study.id)
            .order_by(
                Series.series_number.asc().nulls_last(),
                Series.created_at.asc(),
            )
        )
    )
