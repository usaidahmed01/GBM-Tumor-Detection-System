from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

LOGGER = logging.getLogger("gbm.dataset.standardize")

TARGET_SIZE = 384
LOW_RESOLUTION_MIN_SIDE = 128
EXTREME_ASPECT_RATIO = 2.0
BLANK_STD_MAX = 3.0
BLANK_DYNAMIC_RANGE_MAX = 15.0
LOW_CONTRAST_DYNAMIC_RANGE = 35.0


@dataclass(frozen=True)
class QualityRecord:
    sample_id: str
    class_name: str
    label: int
    clean_relative_path: str
    standardized_relative_path: str
    near_duplicate_group_id: str
    original_width: int
    original_height: int
    aspect_ratio: float
    grayscale_mean: float
    grayscale_std: float
    p01: float
    p99: float
    dynamic_range_p01_p99: float
    black_pixel_fraction: float
    quality_status: str
    quality_flags: str


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Grouped manifest not found: {path}. Run Phase 1 Step 2 first."
        )
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Grouped classification manifest is empty")
    required = {"sample_id", "class_name", "label", "clean_relative_path"}
    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"Manifest is missing required columns: {sorted(missing)}")
    return rows


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def robust_quality_metrics(gray: Image.Image) -> dict[str, float]:
    arr = np.asarray(gray, dtype=np.float32)
    p01, p99 = np.percentile(arr, [1, 99])
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p01": float(p01),
        "p99": float(p99),
        "dynamic_range": float(p99 - p01),
        "black_fraction": float(np.mean(arr <= 5.0)),
    }


def quality_flags(width: int, height: int, metrics: dict[str, float]) -> tuple[str, list[str]]:
    flags: list[str] = []
    min_side = min(width, height)
    ratio = max(width / height, height / width)

    if metrics["std"] <= BLANK_STD_MAX and metrics["dynamic_range"] <= BLANK_DYNAMIC_RANGE_MAX:
        flags.append("near_blank")
    if min_side < LOW_RESOLUTION_MIN_SIDE:
        flags.append("low_resolution")
    if ratio > EXTREME_ASPECT_RATIO:
        flags.append("extreme_aspect_ratio")
    if metrics["dynamic_range"] < LOW_CONTRAST_DYNAMIC_RANGE and "near_blank" not in flags:
        flags.append("low_contrast")

    # Only a near-blank image is automatically unsuitable. Other flags require review.
    if "near_blank" in flags:
        status = "reject"
    elif flags:
        status = "review"
    else:
        status = "pass"
    return status, flags


def standardize_mri_2d(image: Image.Image, target_size: int = TARGET_SIZE) -> Image.Image:
    """Create deterministic geometry-preserving 3-channel input.

    MRI content is converted to luminance, padded to a square canvas without
    stretching, then resized to the selected classifier input size. No random
    augmentation is applied here; augmentation belongs only to the training split.
    """
    gray = ImageOps.exif_transpose(image).convert("L")
    canvas_side = max(gray.width, gray.height)
    canvas = Image.new("L", (canvas_side, canvas_side), 0)
    left = (canvas_side - gray.width) // 2
    top = (canvas_side - gray.height) // 2
    canvas.paste(gray, (left, top))
    resized = canvas.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return resized.convert("RGB")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_flagged_montage(
    project_root: Path,
    records: list[QualityRecord],
    output_path: Path,
    *,
    max_items: int = 40,
) -> None:
    flagged = [r for r in records if r.quality_status != "pass"][:max_items]
    if not flagged:
        if output_path.exists():
            output_path.unlink()
        return

    thumb = 220
    label_h = 58
    cols = 4
    rows = math.ceil(len(flagged) / cols)
    canvas = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, rec in enumerate(flagged):
        x = (idx % cols) * thumb
        y = (idx // cols) * (thumb + label_h)
        path = project_root / rec.standardized_relative_path
        with Image.open(path) as img:
            tile = img.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y))
        draw.text((x + 4, y + thumb + 3), rec.sample_id[:24], fill="black")
        draw.text((x + 4, y + thumb + 22), rec.quality_status, fill="black")
        draw.text((x + 4, y + thumb + 39), rec.quality_flags[:34], fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def standardize_dataset(project_root: Path, *, target_size: int = TARGET_SIZE) -> dict:
    project_root = project_root.resolve()
    manifests_root = project_root / "data" / "manifests"
    reports_root = project_root / "data" / "reports"
    input_manifest = manifests_root / "classification_manifest_grouped.csv"
    output_root = project_root / "data" / "processed" / f"classification_{target_size}"

    rows = read_manifest(input_manifest)
    reset_dir(output_root)
    for class_name in ("yes", "no"):
        (output_root / class_name).mkdir(parents=True, exist_ok=True)

    reports_root.mkdir(parents=True, exist_ok=True)

    records: list[QualityRecord] = []
    for row in rows:
        source = project_root / row["clean_relative_path"]
        if not source.is_file():
            raise FileNotFoundError(f"Image missing for {row['sample_id']}: {source}")

        with Image.open(source) as img:
            oriented = ImageOps.exif_transpose(img)
            width, height = oriented.size
            gray = oriented.convert("L")
            metrics = robust_quality_metrics(gray)
            status, flags = quality_flags(width, height, metrics)
            standardized = standardize_mri_2d(oriented, target_size=target_size)

        output_path = output_root / row["class_name"] / f"{row['sample_id']}.png"
        standardized.save(output_path, format="PNG", optimize=True)

        records.append(
            QualityRecord(
                sample_id=row["sample_id"],
                class_name=row["class_name"],
                label=int(row["label"]),
                clean_relative_path=row["clean_relative_path"],
                standardized_relative_path=output_path.relative_to(project_root).as_posix(),
                near_duplicate_group_id=row.get("near_duplicate_group_id", ""),
                original_width=width,
                original_height=height,
                aspect_ratio=round(width / height, 6),
                grayscale_mean=round(metrics["mean"], 4),
                grayscale_std=round(metrics["std"], 4),
                p01=round(metrics["p01"], 4),
                p99=round(metrics["p99"], 4),
                dynamic_range_p01_p99=round(metrics["dynamic_range"], 4),
                black_pixel_fraction=round(metrics["black_fraction"], 6),
                quality_status=status,
                quality_flags=";".join(flags),
            )
        )

    records.sort(key=lambda r: (r.class_name, r.sample_id))
    output_manifest = manifests_root / "classification_quality_manifest.csv"
    write_csv(
        output_manifest,
        [asdict(r) for r in records],
        list(QualityRecord.__dataclass_fields__.keys()),
    )

    status_counts = {status: sum(r.quality_status == status for r in records) for status in ("pass", "review", "reject")}
    class_counts = {
        class_name: sum(r.class_name == class_name for r in records)
        for class_name in ("yes", "no")
    }
    dimensions = [(r.original_width, r.original_height) for r in records]
    unique_dimensions = len(set(dimensions))

    config = {
        "version": "phase1_step3_v1",
        "classifier_target_size": [target_size, target_size],
        "orientation": "apply EXIF transpose before pixel processing",
        "channel_policy": "convert MRI to grayscale luminance then replicate to RGB",
        "geometry_policy": "pad shorter dimension to square with black background; never stretch anatomy",
        "resize_interpolation": "PIL LANCZOS",
        "saved_format": "PNG",
        "random_augmentation": "none in canonical preprocessing; training-only later",
        "model_normalization": "deferred to the selected TorchVision pretrained weight transform during model training",
        "automatic_removal_policy": "only near-blank inputs are marked reject; other quality flags require review",
    }
    (reports_root / "phase1_step3_preprocessing_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    report = {
        "input_manifest": input_manifest.relative_to(project_root).as_posix(),
        "sample_count": len(records),
        "class_counts": class_counts,
        "unique_original_dimensions": unique_dimensions,
        "quality_status_counts": status_counts,
        "quality_thresholds": {
            "low_resolution_min_side": LOW_RESOLUTION_MIN_SIDE,
            "extreme_aspect_ratio": EXTREME_ASPECT_RATIO,
            "blank_std_max": BLANK_STD_MAX,
            "blank_dynamic_range_max": BLANK_DYNAMIC_RANGE_MAX,
            "low_contrast_dynamic_range": LOW_CONTRAST_DYNAMIC_RANGE,
        },
        "canonical_preprocessing": config,
        "outputs": {
            "standardized_dataset": output_root.relative_to(project_root).as_posix(),
            "quality_manifest": output_manifest.relative_to(project_root).as_posix(),
            "preprocessing_config": "data/reports/phase1_step3_preprocessing_config.json",
            "flagged_review_montage": "data/reports/phase1_step3_flagged_review.jpg",
        },
    }
    report_path = reports_root / "phase1_step3_quality_audit.json"
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    create_flagged_montage(
        project_root,
        records,
        reports_root / "phase1_step3_flagged_review.jpg",
    )

    LOGGER.info("Classification standardization complete: %s", json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit image quality and create deterministic geometry-preserving 2D MRI inputs."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    try:
        standardize_dataset(args.project_root, target_size=args.target_size)
    except Exception as exc:
        LOGGER.error("Standardization failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
