from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ReportFinalizeRequest(BaseModel):
    clinician_name: str = Field(min_length=2, max_length=200)
    clinician_comment: str | None = Field(default=None, max_length=4000)


class ReportPreviewResponse(BaseModel):
    report_version: str
    study_uuid: uuid.UUID
    decision_analysis_run_uuid: uuid.UUID
    finalization_ready: bool
    blockers: list[str]
    report: dict


class ReportFinalizedResponse(BaseModel):
    report_uuid: uuid.UUID
    report_version: str
    study_uuid: uuid.UUID
    decision_analysis_run_uuid: uuid.UUID
    status: str
    report_checksum_sha256: str
    signed_at: datetime
    clinician_name: str
    clinician_comment: str | None = None
    signoff_identity_verified: bool
    report: dict
