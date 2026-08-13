from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from gbm_ai.api.config import Settings
from gbm_ai.api.db import DatabaseManager
from gbm_ai.api.storage.local import LocalObjectStore


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> DatabaseManager:
    return request.app.state.database


def get_object_store(request: Request) -> LocalObjectStore:
    return request.app.state.object_store


def get_db_session(
    database: DatabaseManager = Depends(get_database),
) -> Generator[Session, None, None]:
    yield from database.session()
