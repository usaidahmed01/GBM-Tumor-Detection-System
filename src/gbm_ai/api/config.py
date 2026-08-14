from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application settings loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GBM_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "GBM Clinical Decision Support System API"
    app_version: str = "0.5.1"
    api_v1_prefix: str = "/api/v1"

    environment: Literal["development", "test", "staging", "production"] = (
        "development"
    )
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    database_url: SecretStr = SecretStr(
        "postgresql+psycopg://gbm_cdss:CHANGE_ME@localhost:5432/gbm_cdss"
    )
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=5, ge=0, le=100)
    db_pool_timeout_seconds: int = Field(default=30, ge=1, le=300)

    storage_root: Path = Path("var/storage")
    storage_max_object_bytes: int = Field(
        default=5 * 1024 * 1024 * 1024,
        ge=1024,
    )
    storage_chunk_bytes: int = Field(
        default=1024 * 1024,
        ge=64 * 1024,
        le=16 * 1024 * 1024,
    )

    # Phase 6 frozen MONAI model bundle cache. This path contains downloaded
    # model weights/configs and must stay outside source control.
    segmentation_bundle_root: Path = Path("var/model_bundles")

    # Phase 6 Step 5 guarded inference controls. "auto" prefers CUDA when
    # available and otherwise uses CPU. The spatial-voxel guard is rechecked
    # immediately before the real SegResNet forward path.
    segmentation_inference_device: Literal["auto", "cpu", "cuda"] = "auto"
    segmentation_inference_max_spatial_voxels: int = Field(
        default=20_000_000,
        ge=32 * 32 * 32,
        le=100_000_000,
    )

    # Phase 6 Step 6 durable PostgreSQL-backed worker controls. The worker
    # refreshes its lease while SegResNet runs so an unexpected process exit
    # can be detected and safely requeued without keeping API requests open.
    segmentation_job_lease_seconds: int = Field(default=900, ge=60, le=86_400)
    segmentation_job_heartbeat_seconds: int = Field(default=60, ge=5, le=3_600)
    segmentation_job_poll_seconds: float = Field(default=2.0, ge=0.1, le=60.0)
    segmentation_job_retry_delay_seconds: int = Field(default=30, ge=0, le=86_400)
    segmentation_job_max_attempts: int = Field(default=2, ge=1, le=10)

    # Phase 5 upload boundary. Multipart overhead means the request ceiling is
    # intentionally a little larger than the maximum stored object.
    upload_max_request_bytes: int = Field(
        default=5 * 1024 * 1024 * 1024 + 64 * 1024 * 1024,
        ge=1024,
    )
    upload_max_archive_entries: int = Field(default=5000, ge=1, le=100000)
    upload_max_archive_uncompressed_bytes: int = Field(
        default=10 * 1024 * 1024 * 1024,
        ge=1024,
    )
    upload_max_archive_single_entry_bytes: int = Field(
        default=5 * 1024 * 1024 * 1024,
        ge=1024,
    )
    upload_max_archive_compression_ratio: float = Field(
        default=200.0,
        ge=1.0,
        le=10000.0,
    )

    @field_validator("api_v1_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith("/"):
            raise ValueError("api_v1_prefix must start with '/'")
        return value.rstrip("/") or "/api/v1"

    @property
    def database_url_value(self) -> str:
        return self.database_url.get_secret_value()

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def storage_root_resolved(self) -> Path:
        return self.storage_root.expanduser().resolve()

    @property
    def segmentation_bundle_root_resolved(self) -> Path:
        return self.segmentation_bundle_root.expanduser().resolve()

    def safe_summary(self) -> dict[str, object]:
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "api_v1_prefix": self.api_v1_prefix,
            "environment": self.environment,
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "database_driver": self.database_url_value.split("://", 1)[0],
            "db_pool_size": self.db_pool_size,
            "db_max_overflow": self.db_max_overflow,
            "storage_root": str(self.storage_root),
            "segmentation_bundle_root": str(self.segmentation_bundle_root),
            "segmentation_inference_device": self.segmentation_inference_device,
            "segmentation_inference_max_spatial_voxels": self.segmentation_inference_max_spatial_voxels,
            "segmentation_job_lease_seconds": self.segmentation_job_lease_seconds,
            "segmentation_job_heartbeat_seconds": self.segmentation_job_heartbeat_seconds,
            "segmentation_job_poll_seconds": self.segmentation_job_poll_seconds,
            "segmentation_job_retry_delay_seconds": self.segmentation_job_retry_delay_seconds,
            "segmentation_job_max_attempts": self.segmentation_job_max_attempts,
            "storage_max_object_bytes": self.storage_max_object_bytes,
            "upload_max_request_bytes": self.upload_max_request_bytes,
            "upload_max_archive_entries": self.upload_max_archive_entries,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
