from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class LiveHealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class ReadyHealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    database: Literal["available", "unavailable"]
    environment: str
    detail: str | None = None


class VersionResponse(BaseModel):
    service: str
    application_version: str
    api_version: str
    environment: str
    ml_scope: str
    clinical_validation_status: str
