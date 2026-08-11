from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = {
    "sample_id",
    "class_name",
    "label",
    "standardized_relative_path",
    "quality_status",
    "holdout_split",
    "cv_fold",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize_group(row: dict[str, str]) -> str:
    return (
        row.get("leakage_group_id")
        or row.get("near_duplicate_group_id")
        or row["sample_id"]
    ).strip()


def validate_release(project_root: Path) -> dict:
    split_manifest = project_root / "data" / "manifests" / "classification_split_manifest.csv"
    quality_manifest = project_root / "data" / "manifests" / "classification_quality_manifest.csv"
    preprocessing_config = project_root / "data" / "reports" / "phase1_step3_preprocessing_config.json"
    split_config = project_root / "data" / "reports" / "phase1_step4_split_config.json"
    split_report = project_root / "data" / "reports" / "phase1_step4_split_report.json"

    required_files = [
        split_manifest,
        quality_manifest,
        preprocessing_config,
        split_config,
        split_report,
    ]
    missing = [str(p.relative_to(project_root)) for p in required_files if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing required Phase 1 artifacts: {missing}")

    rows = read_csv(split_manifest)
    if not rows:
        raise RuntimeError("Split manifest is empty.")

    missing_cols = REQUIRED_COLUMNS.difference(rows[0].keys())
    if missing_cols:
        raise RuntimeError(f"Split manifest missing required columns: {sorted(missing_cols)}")

    sample_ids = [r["sample_id"].strip() for r in rows]
    duplicate_ids = [sid for sid, c in Counter(sample_ids).items() if c > 1]
    if duplicate_ids:
        raise RuntimeError(f"Duplicate sample_id values found: {duplicate_ids[:10]}")

    invalid_quality = [r["sample_id"] for r in rows if r["quality_status"].strip().upper() != "PASS"]
    if invalid_quality:
        raise RuntimeError(
            "Release contains non-PASS samples. Review before training: "
            + ", ".join(invalid_quality[:10])
        )

    missing_images = []
    for r in rows:
        p = project_root / r["standardized_relative_path"]
        if not p.exists():
            missing_images.append(r["standardized_relative_path"])
    if missing_images:
        raise RuntimeError(f"Missing standardized images: {missing_images[:10]}")

    outer_groups: dict[str, set[str]] = defaultdict(set)
    cv_groups: dict[str, set[str]] = defaultdict(set)

    for r in rows:
        group = normalize_group(r)
        split = r["holdout_split"].strip().lower()
        if split not in {"development", "test"}:
            raise RuntimeError(f"Invalid holdout_split for {r['sample_id']}: {split}")
        outer_groups[group].add(split)

        fold = r["cv_fold"].strip()
        if split == "test":
            if fold not in {"", "nan", "None", "none"}:
                raise RuntimeError(
                    f"Locked test sample {r['sample_id']} unexpectedly has cv_fold={fold}"
                )
        else:
            if fold == "":
                raise RuntimeError(f"Development sample {r['sample_id']} has no cv_fold.")
            cv_groups[group].add(fold)

    leaking_outer = {g: sorted(v) for g, v in outer_groups.items() if len(v) > 1}
    if leaking_outer:
        raise RuntimeError(f"Leakage groups cross development/test: {list(leaking_outer)[:10]}")

    leaking_cv = {g: sorted(v) for g, v in cv_groups.items() if len(v) > 1}
    if leaking_cv:
        raise RuntimeError(f"Leakage groups cross CV folds: {list(leaking_cv)[:10]}")

    counts = Counter()
    class_counts = Counter()
    cv_counts = Counter()
    for r in rows:
        split = r["holdout_split"].strip().lower()
        label = r["class_name"].strip().lower()
        counts[split] += 1
        class_counts[(split, label)] += 1
        if split == "development":
            cv_counts[r["cv_fold"].strip()] += 1

    release_summary = {
        "total_samples": len(rows),
        "development_samples": counts["development"],
        "test_samples": counts["test"],
        "class_distribution": {
            "development": {
                "yes": class_counts[("development", "yes")],
                "no": class_counts[("development", "no")],
            },
            "test": {
                "yes": class_counts[("test", "yes")],
                "no": class_counts[("test", "no")],
            },
        },
        "cv_fold_sizes": dict(sorted(cv_counts.items(), key=lambda kv: kv[0])),
        "outer_group_leakage": 0,
        "cv_group_leakage": 0,
        "all_quality_pass": True,
        "all_standardized_files_present": True,
    }

    return {
        "summary": release_summary,
        "artifacts": {
            "split_manifest": {
                "path": str(split_manifest.relative_to(project_root)),
                "sha256": sha256_file(split_manifest),
            },
            "quality_manifest": {
                "path": str(quality_manifest.relative_to(project_root)),
                "sha256": sha256_file(quality_manifest),
            },
            "preprocessing_config": {
                "path": str(preprocessing_config.relative_to(project_root)),
                "sha256": sha256_file(preprocessing_config),
            },
            "split_config": {
                "path": str(split_config.relative_to(project_root)),
                "sha256": sha256_file(split_config),
            },
            "split_report": {
                "path": str(split_report.relative_to(project_root)),
                "sha256": sha256_file(split_report),
            },
        },
    }


def freeze_release(project_root: Path, release_name: str = "classification_v1.0") -> Path:
    validation = validate_release(project_root)

    release_dir = project_root / "data" / "releases" / release_name
    release_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = project_root / "data" / "manifests" / "classification_split_manifest.csv"
    frozen_manifest = release_dir / "classification_split_manifest.csv"
    shutil.copy2(source_manifest, frozen_manifest)

    metadata = {
        "release_name": release_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Frozen Phase 1 classification dataset release for Phase 2 model development.",
        "ground_truth_definition": {
            "yes": "GBM (project-defined label)",
            "no": "No GBM (project-defined label)",
        },
        "important_limitations": [
            "Supplied 2D files do not independently verify pathology/molecular diagnosis.",
            "Patient identifiers/grouping are unavailable, so patient-level separation cannot be guaranteed.",
            "Near-duplicate/leakage groups are kept within a single outer split and a single CV fold.",
            "The locked test set must not be used for training, hyperparameter selection, calibration, threshold selection, or early stopping.",
        ],
        **validation,
    }
    metadata["frozen_manifest_sha256"] = sha256_file(frozen_manifest)

    metadata_path = release_dir / "dataset_release.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    ready_path = release_dir / "PHASE1_READY.txt"
    ready_path.write_text(
        "PHASE 1 DATA ENGINEERING: READY FOR PHASE 2\n"
        f"Release: {release_name}\n"
        f"Samples: {validation['summary']['total_samples']}\n"
        f"Development: {validation['summary']['development_samples']}\n"
        f"Locked test: {validation['summary']['test_samples']}\n"
        "Leakage checks: PASS\n"
        "Quality checks: PASS\n",
        encoding="utf-8",
    )

    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--release-name", default="classification_v1.0")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    metadata_path = freeze_release(root, args.release_name)
    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    print("\nPHASE 1 FINAL DATA READINESS CHECK")
    print("=" * 44)
    print(f"Release:       {data['release_name']}")
    print(f"Total:         {data['summary']['total_samples']}")
    print(f"Development:   {data['summary']['development_samples']}")
    print(f"Locked test:   {data['summary']['test_samples']}")
    print(f"Outer leakage: {data['summary']['outer_group_leakage']}")
    print(f"CV leakage:    {data['summary']['cv_group_leakage']}")
    print("Quality:       PASS")
    print("Files present: PASS")
    print("\nSTATUS: READY FOR PHASE 2")


if __name__ == "__main__":
    main()
