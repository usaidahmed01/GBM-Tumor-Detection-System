from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from gbm_ai.data.classification_dataset import (
    GBMClassificationDataset,
    create_dataloader,
    verify_frozen_release,
)
from gbm_ai.models.efficientnet_v2_gbm import GBMEfficientNetV2S
from gbm_ai.training.device import describe_device, resolve_device
from gbm_ai.training.metrics import (
    binary_metrics,
    bootstrap_metric_ci,
)
from gbm_ai.training.reproducibility import seed_everything
from gbm_ai.training.train_classifier import TrainingConfig, train


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GBMEfficientNetV2S(pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict_validation_fold(
    project_root: Path,
    fold: int,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    release_name: str,
) -> list[dict]:
    loader = create_dataloader(
        project_root,
        split="validation",
        fold=fold,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        release_name=release_name,
    )
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)

    predictions: list[dict] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()

        for sample_id, class_name, target, probability in zip(
            batch["sample_id"],
            batch["class_name"],
            batch["target"].cpu().numpy().tolist(),
            probs,
        ):
            predictions.append(
                {
                    "sample_id": sample_id,
                    "class_name": class_name,
                    "target": int(target),
                    "probability_gbm": float(probability),
                    "fold": fold,
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    "checkpoint_stage": checkpoint.get("stage"),
                }
            )

    return predictions


def read_release_rows(project_root: Path, release_name: str) -> list[dict]:
    manifest_path, _ = verify_frozen_release(project_root, release_name)
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def validate_oof_coverage(
    release_rows: list[dict],
    oof_rows: list[dict],
    expected_folds: set[int],
) -> dict:
    development_ids = {
        r["sample_id"]
        for r in release_rows
        if r["holdout_split"].strip().lower() == "development"
    }
    test_ids = {
        r["sample_id"]
        for r in release_rows
        if r["holdout_split"].strip().lower() == "test"
    }
    expected_ids = {
        r["sample_id"]
        for r in release_rows
        if r["holdout_split"].strip().lower() == "development"
        and int(r["cv_fold"]) in expected_folds
    }

    oof_ids = [r["sample_id"] for r in oof_rows]
    unique_oof_ids = set(oof_ids)

    duplicate_oof = sorted(
        {sid for sid in oof_ids if oof_ids.count(sid) > 1}
    )
    test_leak = sorted(unique_oof_ids.intersection(test_ids))
    unexpected = sorted(unique_oof_ids - development_ids)
    missing = sorted(expected_ids - unique_oof_ids)

    if duplicate_oof:
        raise RuntimeError(f"OOF predictions contain duplicate sample IDs: {duplicate_oof[:10]}")
    if test_leak:
        raise RuntimeError(f"Locked test samples leaked into OOF predictions: {test_leak[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected OOF sample IDs: {unexpected[:10]}")
    if missing:
        raise RuntimeError(f"Missing OOF predictions for requested folds: {missing[:10]}")

    return {
        "expected_oof_samples": len(expected_ids),
        "actual_oof_samples": len(oof_rows),
        "duplicate_oof_samples": 0,
        "locked_test_leakage": 0,
        "missing_oof_samples": 0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_roc(oof_rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    folds = sorted({int(r["fold"]) for r in oof_rows})
    for fold in folds:
        rows = [r for r in oof_rows if int(r["fold"]) == fold]
        y = np.asarray([int(r["target"]) for r in rows])
        p = np.asarray([float(r["probability_gbm"]) for r in rows])
        fpr, tpr, _ = roc_curve(y, p)
        auc = binary_metrics(y, p)["roc_auc"]
        ax.plot(fpr, tpr, label=f"Fold {fold} (AUC={auc:.3f})")

    y_all = np.asarray([int(r["target"]) for r in oof_rows])
    p_all = np.asarray([float(r["probability_gbm"]) for r in oof_rows])
    fpr, tpr, _ = roc_curve(y_all, p_all)
    pooled_auc = binary_metrics(y_all, p_all)["roc_auc"]
    ax.plot(fpr, tpr, linewidth=2.5, label=f"Pooled OOF (AUC={pooled_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("EfficientNetV2-S Cross-Validated ROC")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_cross_validation(
    project_root: Path,
    folds: list[int],
    device_name: str,
    batch_size: int,
    num_workers: int,
    warmup_epochs: int,
    finetune_epochs: int,
    finetune_blocks: int,
    patience: int,
    seed: int,
    skip_existing: bool,
    pretrained: bool,
    max_train_batches: int,
    max_val_batches: int,
) -> Path:
    project_root = project_root.resolve()
    seed_everything(seed)

    release_name = "classification_v1.0"
    release_rows = read_release_rows(project_root, release_name)
    device = resolve_device(device_name)

    cv_dir = (
        project_root
        / "artifacts"
        / "cross_validation"
        / f"efficientnetv2s_seed{seed}"
    )
    cv_dir.mkdir(parents=True, exist_ok=True)

    all_oof: list[dict] = []
    fold_summaries: list[dict] = []

    for fold in folds:
        print(f"\n{'=' * 60}\nCROSS-VALIDATION FOLD {fold}\n{'=' * 60}")

        config = TrainingConfig(
            release_name=release_name,
            fold=fold,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
            warmup_epochs=warmup_epochs,
            finetune_epochs=finetune_epochs,
            finetune_blocks=finetune_blocks,
            patience=patience,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        )

        experiment_dir = (
            project_root
            / "artifacts"
            / "experiments"
            / f"efficientnetv2s_fold{fold}_seed{seed}"
        )
        checkpoint_path = experiment_dir / "checkpoints" / "best_model.pt"

        if not (skip_existing and checkpoint_path.exists()):
            checkpoint_path = train(
                project_root,
                config,
                device_name=device_name,
                pretrained=pretrained,
            )
        else:
            print(f"Using existing checkpoint: {checkpoint_path}")

        fold_predictions = predict_validation_fold(
            project_root,
            fold,
            checkpoint_path,
            device,
            batch_size,
            num_workers,
            seed,
            release_name,
        )
        write_csv(experiment_dir / "validation_predictions.csv", fold_predictions)

        targets = [r["target"] for r in fold_predictions]
        probabilities = [r["probability_gbm"] for r in fold_predictions]
        metrics = binary_metrics(targets, probabilities, threshold=0.5)

        fold_summary = {
            "fold": fold,
            "validation_samples": len(fold_predictions),
            **metrics,
            "checkpoint": str(checkpoint_path.relative_to(project_root)),
        }
        fold_summaries.append(fold_summary)
        all_oof.extend(fold_predictions)

        (experiment_dir / "validation_summary.json").write_text(
            json.dumps(fold_summary, indent=2),
            encoding="utf-8",
        )

    coverage = validate_oof_coverage(release_rows, all_oof, set(folds))
    write_csv(cv_dir / "oof_predictions.csv", all_oof)
    write_csv(cv_dir / "fold_metrics.csv", fold_summaries)

    y = [r["target"] for r in all_oof]
    p = [r["probability_gbm"] for r in all_oof]
    pooled = binary_metrics(y, p, threshold=0.5)

    roc_ci = bootstrap_metric_ci(y, p, metric="roc_auc", n_bootstrap=1000, seed=seed)
    pr_ci = bootstrap_metric_ci(y, p, metric="pr_auc", n_bootstrap=1000, seed=seed)

    fold_auc = np.asarray([float(r["roc_auc"]) for r in fold_summaries], dtype=float)
    fold_pr = np.asarray([float(r["pr_auc"]) for r in fold_summaries], dtype=float)

    false_negatives = [
        r for r in all_oof
        if int(r["target"]) == 1 and float(r["probability_gbm"]) < 0.5
    ]
    false_positives = [
        r for r in all_oof
        if int(r["target"]) == 0 and float(r["probability_gbm"]) >= 0.5
    ]
    if false_negatives:
        write_csv(cv_dir / "false_negatives_at_0_5.csv", false_negatives)
    if false_positives:
        write_csv(cv_dir / "false_positives_at_0_5.csv", false_positives)

    plot_roc(all_oof, cv_dir / "roc_cross_validation.png")

    summary = {
        "status": "COMPLETE",
        "model": "EfficientNetV2-S",
        "dataset_release": release_name,
        "seed": seed,
        "folds": folds,
        "device": describe_device(device),
        "locked_test_used": False,
        "coverage": coverage,
        "fold_roc_auc_mean": float(np.nanmean(fold_auc)),
        "fold_roc_auc_std": float(np.nanstd(fold_auc, ddof=1)) if len(fold_auc) > 1 else 0.0,
        "fold_pr_auc_mean": float(np.nanmean(fold_pr)),
        "fold_pr_auc_std": float(np.nanstd(fold_pr, ddof=1)) if len(fold_pr) > 1 else 0.0,
        "pooled_oof_metrics_at_0_5": pooled,
        "pooled_oof_roc_auc_ci": roc_ci,
        "pooled_oof_pr_auc_ci": pr_ci,
        "false_negatives_at_0_5": len(false_negatives),
        "false_positives_at_0_5": len(false_positives),
        "important_note": (
            "Threshold 0.5 results are monitoring/evaluation only. "
            "Clinical thresholds and calibration are not selected in this step."
        ),
    }
    summary_path = cv_dir / "cross_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nPHASE 2 STEP 3 — CROSS-VALIDATION COMPLETE")
    print("=" * 54)
    print(f"Folds:                 {folds}")
    print(f"OOF samples:           {coverage['actual_oof_samples']}")
    print(f"Mean fold ROC-AUC:     {summary['fold_roc_auc_mean']:.4f}")
    print(f"Std fold ROC-AUC:      {summary['fold_roc_auc_std']:.4f}")
    print(f"Pooled OOF ROC-AUC:    {pooled['roc_auc']:.4f}")
    print(f"Pooled OOF PR-AUC:     {pooled['pr_auc']:.4f}")
    print(f"FN at 0.5:             {len(false_negatives)}")
    print(f"FP at 0.5:             {len(false_positives)}")
    print("Locked test used:      NO")
    print(f"Summary:               {summary_path}")

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--finetune-blocks", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()

    run_cross_validation(
        project_root=Path(args.project_root),
        folds=args.folds,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        finetune_blocks=args.finetune_blocks,
        patience=args.patience,
        seed=args.seed,
        skip_existing=args.skip_existing,
        pretrained=not args.no_pretrained,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )


if __name__ == "__main__":
    main()
