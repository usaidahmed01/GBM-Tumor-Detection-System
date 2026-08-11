from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from gbm_ai.data.classification_dataset import GBMClassificationDataset
from gbm_ai.models.efficientnet_v2_gbm import GBMEfficientNetV2S


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def make_release(tmp_path: Path) -> None:
    image_dir = tmp_path / "data" / "processed" / "classification_384"
    (image_dir / "yes").mkdir(parents=True)
    (image_dir / "no").mkdir(parents=True)

    rows = []
    specs = [
        ("yes_train", "yes", "1", "development", "1"),
        ("no_train", "no", "0", "development", "1"),
        ("yes_val", "yes", "1", "development", "0"),
        ("no_val", "no", "0", "development", "0"),
        ("yes_test", "yes", "1", "test", ""),
        ("no_test", "no", "0", "test", ""),
    ]
    for sample_id, cls, label, split, fold in specs:
        rel = f"data/processed/classification_384/{cls}/{sample_id}.png"
        Image.new("RGB", (384, 384), color=(30, 30, 30)).save(tmp_path / rel)
        rows.append(
            {
                "sample_id": sample_id,
                "class_name": cls,
                "label": label,
                "standardized_relative_path": rel,
                "quality_status": "PASS",
                "near_duplicate_group_id": "",
                "leakage_group_id": sample_id,
                "holdout_split": split,
                "cv_fold": fold,
            }
        )

    release_dir = tmp_path / "data" / "releases" / "classification_v1.0"
    release_dir.mkdir(parents=True)
    manifest = release_dir / "classification_split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "release_name": "classification_v1.0",
        "frozen_manifest_sha256": sha256(manifest),
    }
    (release_dir / "dataset_release.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_dataset_uses_frozen_fold_and_locked_test(tmp_path):
    make_release(tmp_path)

    train = GBMClassificationDataset(tmp_path, "train", fold=0)
    val = GBMClassificationDataset(tmp_path, "validation", fold=0)
    test = GBMClassificationDataset(tmp_path, "test", fold=0)

    assert len(train) == 2
    assert len(val) == 2
    assert len(test) == 2

    assert {r["sample_id"] for r in train.rows} == {"yes_train", "no_train"}
    assert {r["sample_id"] for r in val.rows} == {"yes_val", "no_val"}
    assert {r["sample_id"] for r in test.rows} == {"yes_test", "no_test"}

    sample = train[0]
    assert tuple(sample["image"].shape) == (3, 384, 384)
    assert sample["target"].dtype == torch.float32


def test_binary_model_head_and_backbone_freeze():
    model = GBMEfficientNetV2S(pretrained=False, freeze_backbone=True)
    x = torch.randn(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (1,)
    assert model.model.classifier[1].out_features == 1
    assert any(p.requires_grad for p in model.model.classifier.parameters())
    assert not any(p.requires_grad for p in model.model.features.parameters())
    assert model.trainable_parameter_count() < model.total_parameter_count()
