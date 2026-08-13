from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel

from gbm_ai.api.models.analysis import SourceFormat, StudyStatus


class UploadIntakeResponse(BaseModel):
    study_uuid: uuid.UUID
    study_status: StudyStatus
    source_format: SourceFormat
    stored_size_bytes: int
    sha256: str
    upload_kind: Literal["single_object", "zip_archive"]
    archive_entry_count: int | None = None
    archive_total_uncompressed_bytes: int | None = None
    archive_max_compression_ratio_observed: float | None = None
    original_filename_stored: Literal[False] = False
    format_detection_status: Literal["pending"] = "pending"
    next_step: Literal["content_based_format_detection"] = (
        "content_based_format_detection"
    )
