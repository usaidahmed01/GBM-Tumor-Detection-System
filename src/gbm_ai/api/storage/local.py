from __future__ import annotations

import hashlib
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


class StorageError(Exception):
    pass


class InvalidStorageKeyError(StorageError):
    pass


class ObjectTooLargeError(StorageError):
    pass


class ObjectNotFoundError(StorageError):
    pass


@dataclass(frozen=True)
class StoredObject:
    storage_key: str
    sha256: str
    size_bytes: int


class LocalObjectStore:
    """
    Protected local object storage for development.

    Design rules:
    - objects are addressed by opaque internal storage keys;
    - original patient/file names are not embedded into generated keys;
    - paths are always resolved under the configured storage root;
    - writes are streamed, SHA-256 hashed and atomically committed;
    - storage is not exposed as FastAPI static/public content.
    """

    def __init__(
        self,
        root: Path,
        max_object_bytes: int,
        chunk_bytes: int = 1024 * 1024,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.max_object_bytes = int(max_object_bytes)
        self.chunk_bytes = int(chunk_bytes)

        if self.max_object_bytes <= 0:
            raise ValueError("max_object_bytes must be > 0")
        if self.chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be > 0")

        self.root.mkdir(parents=True, exist_ok=True)
        self._apply_private_permissions(self.root)

        self.tmp_root = self.root / ".tmp"
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self._apply_private_permissions(self.tmp_root)

    @staticmethod
    def _apply_private_permissions(path: Path) -> None:
        # On POSIX, restrict storage directories to the current user.
        # Windows ACLs are not modified here; the app still never exposes
        # this directory through a static/public route.
        if os.name == "posix":
            path.chmod(stat.S_IRWXU)

    def generate_study_source_key(self, study_uuid: uuid.UUID) -> str:
        object_uuid = uuid.uuid4()
        return f"studies/{study_uuid}/source/{object_uuid}.bin"

    def _resolve_key(self, storage_key: str) -> Path:
        if not storage_key or storage_key.startswith(("/", "\\")):
            raise InvalidStorageKeyError("storage key must be relative")

        normalized = storage_key.replace("\\", "/")
        parts = Path(normalized).parts

        if any(part in {"", ".", ".."} for part in parts):
            raise InvalidStorageKeyError("invalid storage key")

        candidate = (self.root / Path(*parts)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise InvalidStorageKeyError(
                "storage key escapes configured storage root"
            ) from exc

        return candidate

    def put_stream(
        self,
        storage_key: str,
        source: BinaryIO,
    ) -> StoredObject:
        final_path = self._resolve_key(storage_key)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        self._apply_private_permissions(final_path.parent)

        temp_path = self.tmp_root / f"{uuid.uuid4()}.part"
        digest = hashlib.sha256()
        total = 0

        try:
            with temp_path.open("xb") as output:
                while True:
                    chunk = source.read(self.chunk_bytes)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray)):
                        raise StorageError(
                            "source stream must return bytes"
                        )

                    total += len(chunk)
                    if total > self.max_object_bytes:
                        raise ObjectTooLargeError(
                            f"object exceeds configured maximum "
                            f"{self.max_object_bytes} bytes"
                        )

                    digest.update(chunk)
                    output.write(chunk)

                output.flush()
                os.fsync(output.fileno())

            if final_path.exists():
                raise StorageError(
                    "refusing to overwrite existing storage object"
                )

            os.replace(temp_path, final_path)

            if os.name == "posix":
                final_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

            return StoredObject(
                storage_key=storage_key,
                sha256=digest.hexdigest(),
                size_bytes=total,
            )
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def open_read(self, storage_key: str):
        path = self._resolve_key(storage_key)
        if not path.is_file():
            raise ObjectNotFoundError(storage_key)
        return path.open("rb")

    def exists(self, storage_key: str) -> bool:
        return self._resolve_key(storage_key).is_file()

    def delete(self, storage_key: str) -> None:
        path = self._resolve_key(storage_key)
        if not path.is_file():
            raise ObjectNotFoundError(storage_key)
        path.unlink()

    def verify_checksum(
        self,
        storage_key: str,
        expected_sha256: str,
    ) -> bool:
        path = self._resolve_key(storage_key)
        if not path.is_file():
            raise ObjectNotFoundError(storage_key)

        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(self.chunk_bytes), b""):
                digest.update(chunk)

        return digest.hexdigest().lower() == expected_sha256.lower()
