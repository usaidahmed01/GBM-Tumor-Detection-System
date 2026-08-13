from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from gbm_ai.api.config import Settings, get_settings
from gbm_ai.api.db import DatabaseManager
from gbm_ai.api.middleware.request_body_limit import RequestBodyLimitMiddleware
from gbm_ai.api.middleware.request_id import RequestIdMiddleware
from gbm_ai.api.routers.analysis import router as analysis_router
from gbm_ai.api.routers.clinical import router as clinical_router
from gbm_ai.api.routers.system import router as system_router
from gbm_ai.api.routers.uploads import router as uploads_router
from gbm_ai.api.storage.local import LocalObjectStore


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.database = DatabaseManager(resolved_settings)
        app.state.object_store = LocalObjectStore(
            resolved_settings.storage_root_resolved,
            resolved_settings.storage_max_object_bytes,
            resolved_settings.storage_chunk_bytes,
        )
        try:
            yield
        finally:
            app.state.database.dispose()

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        debug=resolved_settings.debug,
        docs_url="/docs" if not resolved_settings.is_production else None,
        redoc_url="/redoc" if not resolved_settings.is_production else None,
        lifespan=lifespan,
        description=(
            "Backend API for the academic/production-minded GBM clinical "
            "decision-support prototype. This API does not represent a "
            "clinically validated diagnostic system."
        ),
    )

    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=resolved_settings.upload_max_request_bytes,
        api_prefix=resolved_settings.api_v1_prefix,
    )
    app.add_middleware(RequestIdMiddleware)

    prefix = resolved_settings.api_v1_prefix
    app.include_router(system_router, prefix=prefix)
    app.include_router(clinical_router, prefix=prefix)
    app.include_router(analysis_router, prefix=prefix)
    app.include_router(uploads_router, prefix=prefix)
    return app


app = create_app()
