from __future__ import annotations

from fastapi.testclient import TestClient

from gbm_ai.api.config import Settings
from gbm_ai.api.main import create_app


class FakeDatabase:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    def ping(self):
        if self.should_fail:
            raise RuntimeError("synthetic database failure")

    def dispose(self):
        pass


def test_settings_summary_does_not_expose_database_secret():
    settings = Settings(
        database_url="postgresql+psycopg://secret_user:super_secret@db:5432/gbm"
    )
    summary = settings.safe_summary()

    serialized = str(summary)
    assert "super_secret" not in serialized
    assert "secret_user" not in serialized
    assert summary["database_driver"] == "postgresql+psycopg"


def test_liveness_and_version_endpoints(monkeypatch):
    settings = Settings(environment="test", debug=False)
    app = create_app(settings)

    # Replace the real DB constructor used by lifespan so tests need no PostgreSQL.
    monkeypatch.setattr(
        "gbm_ai.api.main.DatabaseManager",
        lambda settings: FakeDatabase(False),
    )

    with TestClient(app) as client:
        live = client.get("/api/v1/health/live")
        assert live.status_code == 200
        assert live.json()["status"] == "ok"

        version = client.get("/api/v1/version")
        assert version.status_code == 200
        body = version.json()
        assert body["api_version"] == "v1"
        assert body["clinical_validation_status"] == (
            "academic_prototype_not_clinically_validated"
        )


def test_readiness_reports_database_available(monkeypatch):
    settings = Settings(environment="test")
    app = create_app(settings)
    monkeypatch.setattr(
        "gbm_ai.api.main.DatabaseManager",
        lambda settings: FakeDatabase(False),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["database"] == "available"


def test_readiness_fails_safely_without_leaking_raw_error(monkeypatch):
    settings = Settings(environment="test")
    app = create_app(settings)
    monkeypatch.setattr(
        "gbm_ai.api.main.DatabaseManager",
        lambda settings: FakeDatabase(True),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["database"] == "unavailable"
    assert body["detail"] == "database_check_failed:RuntimeError"
    assert "synthetic database failure" not in response.text


def test_production_disables_interactive_docs(monkeypatch):
    settings = Settings(environment="production", debug=False)
    app = create_app(settings)
    monkeypatch.setattr(
        "gbm_ai.api.main.DatabaseManager",
        lambda settings: FakeDatabase(False),
    )

    with TestClient(app) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
