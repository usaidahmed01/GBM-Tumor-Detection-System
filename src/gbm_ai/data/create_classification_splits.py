from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

LOGGER = logging.getLogger("gbm.dataset.split")

DEFAULT_SEED = 42
OUTER_FOLDS = 7   # 1/7 ~= 14.3% locked holdout, close to the documented ~15% target.
INNER_FOLDS = 5   # Development-set cross-validation.


@dataclass(frozen=True)
class SplitRecord:
    sample_id: str
    class_name: str
    label: int
    standardized_relative_path: str
    quality_status: str
    near_duplicate_group_id: str
    leakage_group_id: str
    holdout_split: str
    cv_fold: str


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Quality manifest not found: {path}. Run Phase 1 Step 3 first."
        )
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Classification quality manifest is empty")
    required = {
        "sample_id",
        "class_name",
        "label",
        "standardized_relative_path",
        "quality_status",
        "near_duplicate_group_id",
    }
    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {sorted(missing)}")
    return rows


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_sha256(rows: list[SplitRecord]) -> str:
    payload = "\n".join(
        "|".join(
            [
                r.sample_id,
                r.class_name,
                str(r.label),
                r.standardized_relative_path,
                r.leakage_group_id,
                r.holdout_split,
                r.cv_fold,
            ]
        )
        for r in sorted(rows, key=lambda x: x.sample_id)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_eligible_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    eligible: list[dict[str, str]] = []
    rejected = []
    for row in rows:
        status = row["quality_status"].strip().lower()
        if status == "reject":
            rejected.append(row["sample_id"])
            continue
        if status not in {"pass", "review"}:
            raise RuntimeError(
                f"Unknown quality_status={row['quality_status']!r} for {row['sample_id']}"
            )
        row = dict(row)
        ndg = row.get("near_duplicate_group_id", "").strip()
        # Every sample belongs to exactly one leakage-control group. Near-duplicate
        # candidates share a group; otherwise a sample is its own singleton group.
        row["leakage_group_id"] = ndg if ndg else f"sample::{row['sample_id']}"
        eligible.append(row)

    if len(eligible) < 20:
        raise RuntimeError("Too few eligible samples to create robust development/test splits")

    group_labels: dict[str, set[int]] = defaultdict(set)
    for row in eligible:
        group_labels[row["leakage_group_id"]].add(int(row["label"]))
    conflicts = [g for g, labels in group_labels.items() if len(labels) != 1]
    if conflicts:
        raise RuntimeError(
            "A leakage group contains multiple class labels. Resolve label integrity before splitting: "
            + ", ".join(conflicts[:10])
        )
    return eligible


def _choose_outer_test_fold(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(y))
    global_positive = float(y.mean())
    target_fraction = 1.0 / n_splits

    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for fold_id, (dev_idx, test_idx) in enumerate(splitter.split(indices, y, groups)):
        frac = len(test_idx) / len(indices)
        positive_rate = float(y[test_idx].mean()) if len(test_idx) else 0.0
        # Prefer a fold close to the target size and global class ratio.
        score = abs(frac - target_fraction) + abs(positive_rate - global_positive)
        candidates.append((score, fold_id, dev_idx, test_idx))

    _, fold_id, dev_idx, test_idx = min(candidates, key=lambda x: (x[0], x[1]))
    return dev_idx, test_idx, fold_id


def create_leakage_safe_splits(
    project_root: Path,
    *,
    seed: int = DEFAULT_SEED,
    outer_folds: int = OUTER_FOLDS,
    inner_folds: int = INNER_FOLDS,
) -> dict:
    project_root = project_root.resolve()
    manifests_root = project_root / "data" / "manifests"
    reports_root = project_root / "data" / "reports"
    input_manifest = manifests_root / "classification_quality_manifest.csv"
    output_manifest = manifests_root / "classification_split_manifest.csv"

    raw_rows = _read_rows(input_manifest)
    rows = _prepare_eligible_rows(raw_rows)

    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    groups = np.asarray([r["leakage_group_id"] for r in rows], dtype=object)
    sample_ids = np.asarray([r["sample_id"] for r in rows], dtype=object)

    class_counts = Counter(y.tolist())
    if len(class_counts) != 2:
        raise RuntimeError(f"Expected binary labels, found: {dict(class_counts)}")
    if min(class_counts.values()) < outer_folds:
        raise RuntimeError("Not enough minority-class samples for the requested outer split")

    dev_idx, test_idx, selected_outer_fold = _choose_outer_test_fold(
        y, groups, seed=seed, n_splits=outer_folds
    )

    dev_y = y[dev_idx]
    dev_groups = groups[dev_idx]
    dev_sample_ids = sample_ids[dev_idx]
    if len(set(dev_groups.tolist())) < inner_folds:
        raise RuntimeError("Not enough development groups for inner cross-validation")

    inner = StratifiedGroupKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=seed + 1,
    )
    cv_fold_by_sample: dict[str, int] = {}
    dev_indices_local = np.arange(len(dev_idx))
    for fold_id, (_train_local, val_local) in enumerate(
        inner.split(dev_indices_local, dev_y, dev_groups)
    ):
        for local_idx in val_local:
            sid = str(dev_sample_ids[local_idx])
            if sid in cv_fold_by_sample:
                raise RuntimeError(f"Sample assigned to more than one CV fold: {sid}")
            cv_fold_by_sample[sid] = fold_id

    if len(cv_fold_by_sample) != len(dev_idx):
        raise RuntimeError("Not every development sample received exactly one validation fold")

    dev_set = set(dev_idx.tolist())
    test_set = set(test_idx.tolist())
    if dev_set & test_set or len(dev_set | test_set) != len(rows):
        raise RuntimeError("Holdout split assignment is not a complete disjoint partition")

    split_records: list[SplitRecord] = []
    for i, row in enumerate(rows):
        is_test = i in test_set
        split_records.append(
            SplitRecord(
                sample_id=row["sample_id"],
                class_name=row["class_name"],
                label=int(row["label"]),
                standardized_relative_path=row["standardized_relative_path"],
                quality_status=row["quality_status"],
                near_duplicate_group_id=row.get("near_duplicate_group_id", ""),
                leakage_group_id=row["leakage_group_id"],
                holdout_split="test" if is_test else "development",
                cv_fold="" if is_test else str(cv_fold_by_sample[row["sample_id"]]),
            )
        )

    # Hard leakage assertions: a group may exist in only one outer partition and,
    # within development, in only one validation fold.
    outer_by_group: dict[str, set[str]] = defaultdict(set)
    cv_by_group: dict[str, set[str]] = defaultdict(set)
    for r in split_records:
        outer_by_group[r.leakage_group_id].add(r.holdout_split)
        if r.holdout_split == "development":
            cv_by_group[r.leakage_group_id].add(r.cv_fold)
    bad_outer = [g for g, s in outer_by_group.items() if len(s) > 1]
    bad_cv = [g for g, s in cv_by_group.items() if len(s) > 1]
    if bad_outer or bad_cv:
        raise RuntimeError(
            f"Leakage-control invariant failed. outer={bad_outer[:5]} cv={bad_cv[:5]}"
        )

    split_records.sort(key=lambda r: (r.holdout_split, r.cv_fold, r.class_name, r.sample_id))
    _write_csv(
        output_manifest,
        [asdict(r) for r in split_records],
        list(SplitRecord.__dataclass_fields__.keys()),
    )

    # Fold summary: each CV fold is the validation partition once; the other four
    # development folds are training. The locked test set is never used here.
    summary_rows: list[dict] = []
    for fold_id in range(inner_folds):
        val = [r for r in split_records if r.holdout_split == "development" and r.cv_fold == str(fold_id)]
        train = [r for r in split_records if r.holdout_split == "development" and r.cv_fold != str(fold_id)]
        summary_rows.append(
            {
                "cv_fold": fold_id,
                "train_total": len(train),
                "train_gbm": sum(r.label == 1 for r in train),
                "train_no_gbm": sum(r.label == 0 for r in train),
                "val_total": len(val),
                "val_gbm": sum(r.label == 1 for r in val),
                "val_no_gbm": sum(r.label == 0 for r in val),
            }
        )
    _write_csv(
        manifests_root / "classification_cv_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )

    test_records = [r for r in split_records if r.holdout_split == "test"]
    dev_records = [r for r in split_records if r.holdout_split == "development"]
    manifest_fingerprint = _manifest_sha256(split_records)
    input_fingerprint = _file_sha256(input_manifest)

    config = {
        "version": "phase1_step4_v1",
        "seed": seed,
        "strategy": "locked stratified-group holdout plus 5-fold stratified-group CV on development data",
        "outer_splitter": {
            "type": "StratifiedGroupKFold",
            "n_splits": outer_folds,
            "selected_fold_as_locked_test": selected_outer_fold,
            "approximate_test_fraction": round(1.0 / outer_folds, 6),
        },
        "inner_splitter": {
            "type": "StratifiedGroupKFold",
            "n_splits": inner_folds,
            "random_state": seed + 1,
        },
        "group_policy": "near_duplicate_group_id when present; otherwise each sample is a singleton group",
        "quality_policy": "quality_status=reject excluded; pass/review retained",
        "test_set_policy": "locked: never use for training, threshold selection, calibration, early stopping, or model selection",
    }
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "phase1_step4_split_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    report = {
        "input_manifest": input_manifest.relative_to(project_root).as_posix(),
        "input_manifest_sha256": input_fingerprint,
        "output_manifest": output_manifest.relative_to(project_root).as_posix(),
        "output_split_fingerprint_sha256": manifest_fingerprint,
        "seed": seed,
        "eligible_total": len(split_records),
        "excluded_reject_total": len(raw_rows) - len(rows),
        "class_counts": {
            "gbm": sum(r.label == 1 for r in split_records),
            "no_gbm": sum(r.label == 0 for r in split_records),
        },
        "locked_test": {
            "total": len(test_records),
            "gbm": sum(r.label == 1 for r in test_records),
            "no_gbm": sum(r.label == 0 for r in test_records),
            "fraction": round(len(test_records) / len(split_records), 6),
        },
        "development": {
            "total": len(dev_records),
            "gbm": sum(r.label == 1 for r in dev_records),
            "no_gbm": sum(r.label == 0 for r in dev_records),
            "cv_folds": summary_rows,
        },
        "leakage_checks": {
            "outer_group_overlap_count": len(bad_outer),
            "development_cv_group_overlap_count": len(bad_cv),
            "all_samples_assigned_once": True,
        },
        "policy": config,
    }
    (reports_root / "phase1_step4_split_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    LOGGER.info("Leakage-safe split complete: %s", json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a locked leakage-safe test set and grouped cross-validation folds."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--outer-folds", type=int, default=OUTER_FOLDS)
    parser.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    try:
        create_leakage_safe_splits(
            args.project_root,
            seed=args.seed,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
        )
    except Exception as exc:
        LOGGER.error("Split creation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
