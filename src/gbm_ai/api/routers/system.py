from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from gbm_ai.api.config import Settings
from gbm_ai.api.db import DatabaseManager
from gbm_ai.api.dependencies import get_app_settings, get_database
from gbm_ai.api.schemas.system import (
    LiveHealthResponse,
    ReadyHealthResponse,
    VersionResponse,
)

router = APIRouter(tags=["system"])


@router.get(
    "/health/live",
    response_model=LiveHealthResponse,
    summary="Process liveness",
)
def health_live(
    settings: Settings = Depends(get_app_settings),
) -> LiveHealthResponse:
    return LiveHealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get(
    "/health/ready",
    response_model=ReadyHealthResponse,
    summary="Service readiness including PostgreSQL",
    responses={503: {"model": ReadyHealthResponse}},
)
def health_ready(
    settings: Settings = Depends(get_app_settings),
    database: DatabaseManager = Depends(get_database),
):
    try:
        database.ping()
    except Exception as exc:
        # Do not return credentials, SQL text or raw driver messages.
        payload = ReadyHealthResponse(
            status="not_ready",
            database="unavailable",
            environment=settings.environment,
            detail=f"database_check_failed:{exc.__class__.__name__}",
        )
        return JSONResponse(
            status_code=503,
            content=payload.model_dump(),
        )

    return ReadyHealthResponse(
        status="ready",
        database="available",
        environment=settings.environment,
    )


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Backend/model-scope version information",
)
def version(
    settings: Settings = Depends(get_app_settings),
) -> VersionResponse:
    return VersionResponse(
        service=settings.app_name,
        application_version=settings.app_version,
        api_version="v1",
        environment=settings.environment,
        ml_scope="pre-biopsy 2D MRI GBM-vs-no-GBM decision support",
        clinical_validation_status="academic_prototype_not_clinically_validated",
    )
