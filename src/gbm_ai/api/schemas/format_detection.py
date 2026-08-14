from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel

from gbm_ai.api.models.analysis import (
    DeidentificationStatus,
    SourceFormat,
    StudyStatus,
)


class FormatDetectionResponse(BaseModel):
    study_uuid: uuid.UUID
    study_status: StudyStatus
    source_format: SourceFormat
    modality: str
    parser: str
    technical_metadata: dict
    deidentification_status: DeidentificationStatus
    phi_persisted_by_detection: Literal[False] = False
    original_dicom_uids_persisted_by_detection: Literal[False] = False
    next_step: str


class UploadAndDetectionResponse(BaseModel):
    study_uuid: uuid.UUID
    study_status: StudyStatus
    source_format: SourceFormat
    modality: str
    stored_size_bytes: int
    sha256: str
    upload_kind: Literal["single_object", "zip_archive"]
    archive_entry_count: int | None = None
    archive_total_uncompressed_bytes: int | None = None
    archive_max_compression_ratio_observed: float | None = None
    parser: str
    technical_metadata: dict
    original_filename_stored: Literal[False] = False
    phi_persisted_by_detection: Literal[False] = False

    deidentification_status: DeidentificationStatus
    deidentified_working_copy_created: bool
    deidentified_sha256: str | None = None
    dicom_series_count: int | None = None
    pixel_privacy_status: str | None = None

    next_step: str
