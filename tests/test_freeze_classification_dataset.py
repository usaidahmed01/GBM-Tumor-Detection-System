import csv
import json
from pathlib import Path

import pytest

from gbm_ai.data.freeze_classification_dataset import freeze_release, validate_release


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_project(tmp_path: Path):
    (tmp_path / "data/reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/processed/classification_384/yes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/processed/classification_384/no").mkdir(parents=True, exist_ok=True)

    for p in [
        tmp_path / "data/reports/phase1_step3_preprocessing_config.json",
        tmp_path / "data/reports/phase1_step4_split_config.json",
        tmp_path / "data/reports/phase1_step4_split_report.json",
    ]:
        p.write_text("{}", encoding="utf-8")

    rows = [
        {
            "sample_id": "yes_1", "class_name": "yes", "label": "1",
            "standardized_relative_path": "data/processed/classification_384/yes/yes_1.png",
            "quality_status": "PASS", "near_duplicate_group_id": "",
            "leakage_group_id": "g1", "holdout_split": "development", "cv_fold": "0",
        },
        {
            "sample_id": "no_1", "class_name": "no", "label": "0",
            "standardized_relative_path": "data/processed/classification_384/no/no_1.png",
            "quality_status": "PASS", "near_duplicate_group_id": "",
            "leakage_group_id": "g2", "holdout_split": "test", "cv_fold": "",
        },
    ]
    write_csv(tmp_path / "data/manifests/classification_split_manifest.csv", rows)
    write_csv(tmp_path / "data/manifests/classification_quality_manifest.csv", rows)

    for r in rows:
        p = tmp_path / r["standardized_relative_path"]
        p.write_bytes(b"fake-image")

    return rows


def test_release_is_created_and_hashed(tmp_path):
    make_project(tmp_path)
    metadata_path = freeze_release(tmp_path, "classification_v1.0")
    assert metadata_path.exists()

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["summary"]["total_samples"] == 2
    assert data["summary"]["outer_group_leakage"] == 0
    assert len(data["frozen_manifest_sha256"]) == 64
    assert (tmp_path / "data/releases/classification_v1.0/PHASE1_READY.txt").exists()


def test_outer_group_leakage_fails(tmp_path):
    rows = make_project(tmp_path)

    rows[1]["leakage_group_id"] = "g1"
    write_csv(tmp_path / "data/manifests/classification_split_manifest.csv", rows)

    with pytest.raises(RuntimeError, match="cross development/test"):
        validate_release(tmp_path)
