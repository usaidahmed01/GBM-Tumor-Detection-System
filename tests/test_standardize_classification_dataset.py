from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from gbm_ai.data.standardize_classification_dataset import standardize_dataset, standardize_mri_2d


def _write_grouped_manifest(root: Path, rows: list[dict[str, str]]) -> None:
    path = root / "data" / "manifests" / "classification_manifest_grouped.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "class_name",
        "label",
        "clean_relative_path",
        "near_duplicate_group_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_standardization_preserves_content_without_stretching() -> None:
    arr = np.zeros((100, 200), dtype=np.uint8)
    arr[25:75, 50:150] = 180
    out = standardize_mri_2d(Image.fromarray(arr), target_size=384)
    assert out.size == (384, 384)
    assert out.mode == "RGB"
    # A rectangular source receives black padding above/below rather than being stretched.
    pixels = np.asarray(out)
    assert pixels[:50].mean() < pixels[140:245].mean()


def test_quality_pipeline_marks_near_blank_as_reject(tmp_path: Path) -> None:
    root = tmp_path
    rows = []
    for class_name, label, value in [("yes", 1, 0), ("no", 0, 120)]:
        folder = root / "data" / "interim" / "classification_deduplicated" / class_name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{class_name}_a.png"
        if class_name == "yes":
            arr = np.zeros((200, 200), dtype=np.uint8)
        else:
            arr = np.zeros((200, 200), dtype=np.uint8)
            yy, xx = np.ogrid[:200, :200]
            mask = (xx - 100) ** 2 + (yy - 100) ** 2 < 70**2
            arr[mask] = value
        Image.fromarray(arr).save(path)
        rows.append(
            {
                "sample_id": f"{class_name}_a",
                "class_name": class_name,
                "label": str(label),
                "clean_relative_path": path.relative_to(root).as_posix(),
                "near_duplicate_group_id": "",
            }
        )
    _write_grouped_manifest(root, rows)

    report = standardize_dataset(root)
    assert report["sample_count"] == 2
    assert report["quality_status_counts"]["reject"] == 1
    assert (root / "data" / "processed" / "classification_384" / "yes" / "yes_a.png").is_file()
