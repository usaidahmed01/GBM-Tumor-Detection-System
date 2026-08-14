from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel

from gbm_ai.api.models.analysis import DeidentificationStatus, StudyStatus
from gbm_ai.api.schemas.analysis import SeriesRead


class DicomDeidentificationResponse(BaseModel):
    study_uuid: uuid.UUID
    study_status: StudyStatus
    deidentification_status: DeidentificationStatus
    deidentified_working_copy_created: Literal[True] = True
    deidentified_sha256: str
    deidentified_size_bytes: int
    series_count: int
    instance_count: int
    pixel_privacy_status: str
    original_uids_persisted: Literal[False] = False
    ps3_15_profile_compliance_claimed: Literal[False] = False
    pixel_data_modified: Literal[False] = False
    next_step: Literal["mri_qc_and_sequence_detection"] = (
        "mri_qc_and_sequence_detection"
    )


class DicomSeriesListResponse(BaseModel):
    study_uuid: uuid.UUID
    series_count: int
    series: list[SeriesRead]
