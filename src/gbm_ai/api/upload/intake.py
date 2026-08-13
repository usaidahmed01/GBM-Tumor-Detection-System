from __future__ import annotations

import io
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import BinaryIO


ZIP_MAGIC = (
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
)

NESTED_ARCHIVE_SUFFIXES = (
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".tar.gz",
    ".tgz",
)


class UploadIntakeError(ValueError):
    pass


class EmptyUploadError(UploadIntakeError):
    pass


class UploadObjectTooLargeError(UploadIntakeError):
    pass


class ArchivePolicyError(UploadIntakeError):
    pass


@dataclass(frozen=True)
class UploadPreflight:
    upload_kind: str
    size_bytes: int
    archive_entry_count: int | None = None
    archive_total_uncompressed_bytes: int | None = None
    archive_total_compressed_bytes: int | None = None
    archive_max_compression_ratio_observed: float | None = None


def stream_size(source: BinaryIO) -> int:
    current = source.tell()
    source.seek(0, io.SEEK_END)
    size = source.tell()
    source.seek(current, io.SEEK_SET)
    return int(size)


def _starts_with_zip_magic(source: BinaryIO) -> bool:
    current = source.tell()
    source.seek(0)
    prefix = source.read(4)
    source.seek(current)
    return any(prefix.startswith(magic) for magic in ZIP_MAGIC)


def _safe_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")

    if normalized.startswith("/") or normalized.startswith("//"):
        raise ArchivePolicyError("archive contains absolute path")

    if re.match(r"^[A-Za-z]:", normalized):
        raise ArchivePolicyError("archive contains drive-qualified path")

    path = PurePosixPath(normalized)
    if ".." in path.parts:
        raise ArchivePolicyError("archive contains path traversal")

    return normalized


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0o170000
    return unix_mode == stat.S_IFLNK


def inspect_zip_archive(
    source: BinaryIO,
    *,
    max_entries: int,
    max_total_uncompressed_bytes: int,
    max_single_entry_bytes: int,
    max_compression_ratio: float,
) -> UploadPreflight:
    current = source.tell()
    source.seek(0)

    try:
        with zipfile.ZipFile(source, "r") as archive:
            infos = archive.infolist()

            file_infos = [info for info in infos if not info.is_dir()]
            if not file_infos:
                raise ArchivePolicyError(
                    "archive contains no files"
                )

            if len(file_infos) > max_entries:
                raise ArchivePolicyError(
                    f"archive contains {len(file_infos)} files; "
                    f"maximum is {max_entries}"
                )

            total_uncompressed = 0
            total_compressed = 0
            max_ratio_observed = 1.0

            for info in file_infos:
                normalized_name = _safe_member_name(info.filename)

                if _is_symlink(info):
                    raise ArchivePolicyError(
                        "archive contains symbolic link"
                    )

                if info.flag_bits & 0x1:
                    raise ArchivePolicyError(
                        "encrypted archive members are not accepted"
                    )

                lower_name = normalized_name.lower()
                if lower_name.endswith(NESTED_ARCHIVE_SUFFIXES):
                    raise ArchivePolicyError(
                        "nested archive members are not accepted"
                    )

                if info.file_size > max_single_entry_bytes:
                    raise ArchivePolicyError(
                        f"archive member exceeds maximum size "
                        f"{max_single_entry_bytes} bytes"
                    )

                total_uncompressed += int(info.file_size)
                total_compressed += int(info.compress_size)

                if total_uncompressed > max_total_uncompressed_bytes:
                    raise ArchivePolicyError(
                        "archive total uncompressed size exceeds configured limit"
                    )

                compressed = max(int(info.compress_size), 1)
                ratio = float(info.file_size) / compressed
                max_ratio_observed = max(max_ratio_observed, ratio)

                # Ignore tiny files when considering a compression-ratio bomb.
                if info.file_size >= 1024 * 1024 and ratio > max_compression_ratio:
                    raise ArchivePolicyError(
                        "archive member compression ratio exceeds configured limit"
                    )

            overall_ratio = float(total_uncompressed) / max(total_compressed, 1)
            max_ratio_observed = max(max_ratio_observed, overall_ratio)

            if (
                total_uncompressed >= 1024 * 1024
                and overall_ratio > max_compression_ratio
            ):
                raise ArchivePolicyError(
                    "archive overall compression ratio exceeds configured limit"
                )

            return UploadPreflight(
                upload_kind="zip_archive",
                size_bytes=stream_size(source),
                archive_entry_count=len(file_infos),
                archive_total_uncompressed_bytes=total_uncompressed,
                archive_total_compressed_bytes=total_compressed,
                archive_max_compression_ratio_observed=max_ratio_observed,
            )
    except zipfile.BadZipFile as exc:
        raise ArchivePolicyError(
            "ZIP signature detected but archive structure is invalid"
        ) from exc
    finally:
        source.seek(current)


def preflight_upload(
    source: BinaryIO,
    *,
    max_object_bytes: int,
    max_archive_entries: int,
    max_archive_uncompressed_bytes: int,
    max_archive_single_entry_bytes: int,
    max_archive_compression_ratio: float,
) -> UploadPreflight:
    current = source.tell()
    try:
        size = stream_size(source)
        if size <= 0:
            raise EmptyUploadError("uploaded file is empty")

        if size > max_object_bytes:
            raise UploadObjectTooLargeError(
                f"uploaded object exceeds configured maximum "
                f"{max_object_bytes} bytes"
            )

        source.seek(0)
        if _starts_with_zip_magic(source):
            return inspect_zip_archive(
                source,
                max_entries=max_archive_entries,
                max_total_uncompressed_bytes=max_archive_uncompressed_bytes,
                max_single_entry_bytes=max_archive_single_entry_bytes,
                max_compression_ratio=max_archive_compression_ratio,
            )

        return UploadPreflight(
            upload_kind="single_object",
            size_bytes=size,
        )
    finally:
        source.seek(current)
