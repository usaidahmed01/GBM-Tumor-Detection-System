from __future__ import annotations

import csv
from pathlib import Path

from gbm_ai.data.create_classification_splits import create_leakage_safe_splits


def _write_manifest(path: Path, n_per_class: int = 35) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, class_name in [(0, "no"), (1, "yes")]:
        for i in range(n_per_class):
            # Pair a few samples into known near-duplicate groups.
            group = f"{class_name}_ndg_{i // 2}" if i < 10 else ""
            rows.append(
                {
                    "sample_id": f"{class_name}_{i:03d}",
                    "class_name": class_name,
                    "label": label,
                    "standardized_relative_path": f"data/processed/classification_384/{class_name}/{class_name}_{i:03d}.png",
                    "quality_status": "pass",
                    "near_duplicate_group_id": group,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_split_is_complete_reproducible_and_group_safe(tmp_path: Path) -> None:
    manifest = tmp_path / "data/manifests/classification_quality_manifest.csv"
    _write_manifest(manifest)

    first = create_leakage_safe_splits(tmp_path, seed=42, outer_folds=7, inner_folds=5)
    rows1 = _read(tmp_path / "data/manifests/classification_split_manifest.csv")
    second = create_leakage_safe_splits(tmp_path, seed=42, outer_folds=7, inner_folds=5)
    rows2 = _read(tmp_path / "data/manifests/classification_split_manifest.csv")

    assert first["output_split_fingerprint_sha256"] == second["output_split_fingerprint_sha256"]
    assert rows1 == rows2
    assert len(rows1) == 70
    assert {r["holdout_split"] for r in rows1} == {"development", "test"}

    outer_by_group = {}
    cv_by_group = {}
    for r in rows1:
        outer_by_group.setdefault(r["leakage_group_id"], set()).add(r["holdout_split"])
        if r["holdout_split"] == "development":
            cv_by_group.setdefault(r["leakage_group_id"], set()).add(r["cv_fold"])
        else:
            assert r["cv_fold"] == ""

    assert all(len(v) == 1 for v in outer_by_group.values())
    assert all(len(v) == 1 for v in cv_by_group.values())


def test_conflicting_labels_inside_group_fail(tmp_path: Path) -> None:
    manifest = tmp_path / "data/manifests/classification_quality_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(20):
        label = i % 2
        class_name = "yes" if label else "no"
        rows.append(
            {
                "sample_id": f"s{i:03d}",
                "class_name": class_name,
                "label": label,
                "standardized_relative_path": f"dummy/{i}.png",
                "quality_status": "pass",
                "near_duplicate_group_id": "conflict" if i in (0, 1) else "",
            }
        )
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        create_leakage_safe_splits(tmp_path, seed=42, outer_folds=5, inner_folds=3)
    except RuntimeError as exc:
        assert "multiple class labels" in str(exc)
    else:
        raise AssertionError("Expected cross-label leakage group to fail")
