from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger("gbm.dataset.prepare")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SOURCE_PREFIX = "brain_tumor_dataset/"
CLASS_TO_LABEL = {"no": 0, "yes": 1}


@dataclass(frozen=True)
class ImageRecord:
    sample_id: str
    class_name: str
    label: int
    original_name: str
    source_relative_path: str
    clean_relative_path: str
    extension: str
    sha256: str
    width: int
    height: int
    color_mode: str
    file_size_bytes: int


@dataclass(frozen=True)
class DuplicateRecord:
    class_name: str
    label: int
    duplicate_source_path: str
    canonical_source_path: str
    sha256: str


@dataclass(frozen=True)
class RejectedRecord:
    source_relative_path: str
    reason: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_sample_id(class_name: str, sha256: str) -> str:
    return f"{class_name}_{sha256[:16]}"


def inspect_image_bytes(data: bytes) -> tuple[int, int, str]:
    from io import BytesIO

    with Image.open(BytesIO(data)) as img:
        img.verify()
    with Image.open(BytesIO(data)) as img:
        width, height = img.size
        mode = img.mode
    if width <= 0 or height <= 0:
        raise ValueError("invalid image dimensions")
    return width, height, mode


def is_target_member(member_name: str) -> bool:
    normalized = member_name.replace("\\", "/").lstrip("/")
    parts = Path(normalized).parts
    return (
        len(parts) >= 3
        and normalized.startswith(SOURCE_PREFIX)
        and parts[1] in CLASS_TO_LABEL
        and Path(parts[-1]).suffix.lower() in SUPPORTED_EXTENSIONS
    )


def iter_target_members(zf: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    members = [m for m in zf.infolist() if not m.is_dir() and is_target_member(m.filename)]
    return sorted(members, key=lambda m: m.filename.lower())


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(
    archive_path: Path,
    project_root: Path,
    *,
    fail_on_cross_class_duplicate: bool = True,
) -> dict:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    raw_root = project_root / "data" / "raw" / "classification"
    clean_root = project_root / "data" / "interim" / "classification_deduplicated"
    manifests_root = project_root / "data" / "manifests"
    reports_root = project_root / "data" / "reports"

    # Rebuild generated dataset deterministically. Source ZIP is never modified.
    reset_dir(raw_root)
    reset_dir(clean_root)
    manifests_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_TO_LABEL:
        (raw_root / class_name).mkdir(parents=True, exist_ok=True)
        (clean_root / class_name).mkdir(parents=True, exist_ok=True)

    unique_records: list[ImageRecord] = []
    duplicate_records: list[DuplicateRecord] = []
    rejected_records: list[RejectedRecord] = []

    seen_within_class: dict[tuple[str, str], str] = {}
    hash_to_classes: dict[str, set[str]] = {}
    hash_to_source_paths: dict[str, list[str]] = {}

    raw_counts = {"yes": 0, "no": 0}

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = list(iter_target_members(zf))
        if not members:
            raise RuntimeError(
                "No supported images found under brain_tumor_dataset/yes or brain_tumor_dataset/no"
            )

        for member in members:
            normalized = member.filename.replace("\\", "/")
            parts = Path(normalized).parts
            class_name = parts[1]
            label = CLASS_TO_LABEL[class_name]
            original_name = parts[-1]
            extension = Path(original_name).suffix.lower()
            raw_counts[class_name] += 1

            try:
                data = zf.read(member)
                width, height, color_mode = inspect_image_bytes(data)
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                rejected_records.append(
                    RejectedRecord(source_relative_path=normalized, reason=f"invalid_image: {exc}")
                )
                continue

            digest = sha256_bytes(data)
            hash_to_classes.setdefault(digest, set()).add(class_name)
            hash_to_source_paths.setdefault(digest, []).append(normalized)

            # Preserve the exact source image in immutable-ish raw extraction.
            raw_destination = raw_root / class_name / original_name
            if raw_destination.exists():
                # Avoid accidental filename collision while preserving source bytes.
                raw_destination = raw_root / class_name / f"{digest[:12]}_{original_name}"
            raw_destination.write_bytes(data)

            within_key = (class_name, digest)
            if within_key in seen_within_class:
                duplicate_records.append(
                    DuplicateRecord(
                        class_name=class_name,
                        label=label,
                        duplicate_source_path=normalized,
                        canonical_source_path=seen_within_class[within_key],
                        sha256=digest,
                    )
                )
                continue

            seen_within_class[within_key] = normalized
            sample_id = stable_sample_id(class_name, digest)
            clean_name = f"{sample_id}{extension}"
            clean_destination = clean_root / class_name / clean_name
            clean_destination.write_bytes(data)

            unique_records.append(
                ImageRecord(
                    sample_id=sample_id,
                    class_name=class_name,
                    label=label,
                    original_name=original_name,
                    source_relative_path=normalized,
                    clean_relative_path=clean_destination.relative_to(project_root).as_posix(),
                    extension=extension,
                    sha256=digest,
                    width=width,
                    height=height,
                    color_mode=color_mode,
                    file_size_bytes=len(data),
                )
            )

    cross_class_duplicates = {
        digest: paths
        for digest, classes in hash_to_classes.items()
        if len(classes) > 1
        for paths in [hash_to_source_paths[digest]]
    }

    if cross_class_duplicates and fail_on_cross_class_duplicate:
        report_path = reports_root / "cross_class_duplicates.json"
        report_path.write_text(json.dumps(cross_class_duplicates, indent=2), encoding="utf-8")
        raise RuntimeError(
            "Cross-class exact duplicates detected. This is a label-integrity failure. "
            f"See {report_path}"
        )

    unique_records.sort(key=lambda r: (r.class_name, r.sample_id))
    duplicate_records.sort(key=lambda r: (r.class_name, r.duplicate_source_path.lower()))
    rejected_records.sort(key=lambda r: r.source_relative_path.lower())

    write_csv(
        manifests_root / "classification_manifest.csv",
        [asdict(r) for r in unique_records],
        list(ImageRecord.__dataclass_fields__.keys()),
    )
    write_csv(
        manifests_root / "exact_duplicates.csv",
        [asdict(r) for r in duplicate_records],
        list(DuplicateRecord.__dataclass_fields__.keys()),
    )
    write_csv(
        manifests_root / "rejected_files.csv",
        [asdict(r) for r in rejected_records],
        list(RejectedRecord.__dataclass_fields__.keys()),
    )

    unique_counts = {
        class_name: sum(1 for r in unique_records if r.class_name == class_name)
        for class_name in CLASS_TO_LABEL
    }
    duplicate_counts = {
        class_name: sum(1 for r in duplicate_records if r.class_name == class_name)
        for class_name in CLASS_TO_LABEL
    }

    report = {
        "source_archive": str(archive_path),
        "source_scope": "brain_tumor_dataset/{yes,no} only",
        "class_mapping": {"yes": "GBM", "no": "No GBM"},
        "label_mapping": CLASS_TO_LABEL,
        "raw_counts": raw_counts,
        "valid_unique_counts": unique_counts,
        "valid_unique_total": len(unique_records),
        "exact_duplicate_extras": duplicate_counts,
        "exact_duplicate_total": len(duplicate_records),
        "rejected_total": len(rejected_records),
        "cross_class_exact_duplicate_hashes": len(cross_class_duplicates),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "outputs": {
            "manifest": "data/manifests/classification_manifest.csv",
            "duplicates": "data/manifests/exact_duplicates.csv",
            "rejected": "data/manifests/rejected_files.csv",
            "deduplicated_dataset": "data/interim/classification_deduplicated",
        },
    }
    (reports_root / "phase1_step1_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    LOGGER.info("Dataset preparation complete: %s", json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract, validate and exactly deduplicate the GBM 2D classification dataset."
    )
    parser.add_argument("--archive", required=True, type=Path, help="Path to archive.zip")
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="GBM project root directory"
    )
    parser.add_argument(
        "--allow-cross-class-duplicates",
        action="store_true",
        help="Do not fail if identical bytes appear in both yes and no classes (not recommended).",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    try:
        prepare_dataset(
            args.archive.resolve(),
            args.project_root.resolve(),
            fail_on_cross_class_duplicate=not args.allow_cross_class_duplicates,
        )
    except Exception as exc:
        LOGGER.error("Dataset preparation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
