from __future__ import annotations

import csv
import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from gbm_ai.data.prepare_classification_dataset import prepare_dataset


def make_jpeg(value: int) -> bytes:
    buf = BytesIO()
    Image.new("L", (16, 16), color=value).save(buf, format="JPEG")
    return buf.getvalue()


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_only_target_folder_is_used_and_duplicates_are_removed(tmp_path: Path) -> None:
    yes = make_jpeg(210)
    no = make_jpeg(20)
    zip_path = tmp_path / "archive.zip"
    write_zip(
        zip_path,
        {
            "brain_tumor_dataset/yes/a.jpg": yes,
            "brain_tumor_dataset/yes/a_copy.jpg": yes,
            "brain_tumor_dataset/no/b.jpg": no,
            # Must be ignored even though it looks like another class folder.
            "yes/ignored.jpg": make_jpeg(150),
            "no/ignored.jpg": make_jpeg(50),
        },
    )

    report = prepare_dataset(zip_path, tmp_path)

    assert report["raw_counts"] == {"yes": 2, "no": 1}
    assert report["valid_unique_counts"] == {"yes": 1, "no": 1}
    assert report["exact_duplicate_total"] == 1
    assert report["cross_class_exact_duplicate_hashes"] == 0

    manifest_path = tmp_path / "data/manifests/classification_manifest.csv"
    with manifest_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r["class_name"] for r in rows} == {"yes", "no"}


def test_cross_class_exact_duplicate_is_a_hard_failure(tmp_path: Path) -> None:
    same = make_jpeg(100)
    zip_path = tmp_path / "archive.zip"
    write_zip(
        zip_path,
        {
            "brain_tumor_dataset/yes/a.jpg": same,
            "brain_tumor_dataset/no/b.jpg": same,
        },
    )

    with pytest.raises(RuntimeError, match="Cross-class exact duplicates"):
        prepare_dataset(zip_path, tmp_path)

    report_path = tmp_path / "data/reports/cross_class_duplicates.json"
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))


def test_invalid_image_is_rejected_not_crashed(tmp_path: Path) -> None:
    zip_path = tmp_path / "archive.zip"
    write_zip(
        zip_path,
        {
            "brain_tumor_dataset/yes/good.jpg": make_jpeg(200),
            "brain_tumor_dataset/no/bad.jpg": b"not-a-real-image",
        },
    )

    report = prepare_dataset(zip_path, tmp_path)
    assert report["valid_unique_total"] == 1
    assert report["rejected_total"] == 1
