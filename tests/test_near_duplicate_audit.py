from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from gbm_ai.data.audit_near_duplicates import audit_near_duplicates


def save_image(path: Path, array: np.ndarray, *, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path, quality=quality)


def write_manifest(root: Path, rows: list[dict]) -> None:
    path = root / "data" / "manifests" / "classification_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "class_name",
        "label",
        "original_name",
        "source_relative_path",
        "clean_relative_path",
        "extension",
        "sha256",
        "width",
        "height",
        "color_mode",
        "file_size_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_brain_like(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((180, 180), dtype=np.uint8)
    yy, xx = np.ogrid[:180, :180]
    mask = ((xx - 90) ** 2 / 68**2 + (yy - 90) ** 2 / 82**2) <= 1
    base[mask] = 80 + rng.integers(0, 50, size=int(mask.sum()), dtype=np.uint8)
    return base


def test_resized_reencoded_copy_is_grouped(tmp_path: Path) -> None:
    root = tmp_path
    clean = root / "data" / "interim" / "classification_deduplicated" / "yes"
    clean.mkdir(parents=True)

    original = synthetic_brain_like(42)
    p1 = clean / "yes_a.jpg"
    save_image(p1, original, quality=96)

    resized = Image.fromarray(original).resize((240, 240), Image.Resampling.BICUBIC)
    p2 = clean / "yes_b.jpg"
    resized.save(p2, quality=82)

    different = synthetic_brain_like(999)
    p3 = clean / "yes_c.jpg"
    save_image(p3, different, quality=95)

    rows = []
    for sid, name in [("yes_a", "yes_a.jpg"), ("yes_b", "yes_b.jpg"), ("yes_c", "yes_c.jpg")]:
        p = clean / name
        rows.append(
            {
                "sample_id": sid,
                "class_name": "yes",
                "label": 1,
                "original_name": name,
                "source_relative_path": f"brain_tumor_dataset/yes/{name}",
                "clean_relative_path": p.relative_to(root).as_posix(),
                "extension": ".jpg",
                "sha256": sid,
                "width": 180,
                "height": 180,
                "color_mode": "L",
                "file_size_bytes": p.stat().st_size,
            }
        )
    write_manifest(root, rows)

    report = audit_near_duplicates(root, correlation_min=0.95, rmse_max=0.12)
    assert report["near_duplicate_candidate_pairs"] >= 1
    assert report["cross_class_candidate_pairs"] == 0
    assert (root / "data" / "manifests" / "classification_manifest_grouped.csv").is_file()


def test_cross_class_near_duplicate_fails(tmp_path: Path) -> None:
    root = tmp_path
    arr = synthetic_brain_like(11)
    rows = []
    for class_name, label in [("yes", 1), ("no", 0)]:
        clean = root / "data" / "interim" / "classification_deduplicated" / class_name
        clean.mkdir(parents=True, exist_ok=True)
        p = clean / f"{class_name}_copy.jpg"
        save_image(p, arr, quality=95 if class_name == "yes" else 85)
        rows.append(
            {
                "sample_id": f"{class_name}_copy",
                "class_name": class_name,
                "label": label,
                "original_name": p.name,
                "source_relative_path": f"brain_tumor_dataset/{class_name}/{p.name}",
                "clean_relative_path": p.relative_to(root).as_posix(),
                "extension": ".jpg",
                "sha256": class_name,
                "width": 180,
                "height": 180,
                "color_mode": "L",
                "file_size_bytes": p.stat().st_size,
            }
        )
    write_manifest(root, rows)

    with pytest.raises(RuntimeError, match="Cross-class near-duplicate"):
        audit_near_duplicates(root, correlation_min=0.95, rmse_max=0.12)
