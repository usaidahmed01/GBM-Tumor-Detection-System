from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from gbm_ai.api.config import Settings


class DatabaseManager:
    """
    SQLAlchemy database foundation.

    Engine creation is lazy with respect to actual network connection:
    constructing DatabaseManager does not require PostgreSQL to be online.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: Engine = create_engine(
            settings.database_url_value,
            pool_pre_ping=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout_seconds,
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )

    def ping(self) -> None:
        """Raise on database connection/query failure."""
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def session(self) -> Generator[Session, None, None]:
        db = self.session_factory()
        try:
            yield db
        finally:
            db.close()

    def dispose(self) -> None:
        self.engine.dispose()
