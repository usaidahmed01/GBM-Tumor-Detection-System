from __future__ import annotations

from fastapi import Request

from gbm_ai.api.config import Settings
from gbm_ai.api.db import DatabaseManager


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> DatabaseManager:
    return request.app.state.database
