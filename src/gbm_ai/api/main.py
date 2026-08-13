from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from gbm_ai.api.config import Settings, get_settings
from gbm_ai.api.db import DatabaseManager
from gbm_ai.api.routers.system import router as system_router


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.database = DatabaseManager(resolved_settings)
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

    app.include_router(
        system_router,
        prefix=resolved_settings.api_v1_prefix,
    )
    return app


app = create_app()
