from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from gbm_ai.data.classification_dataset import create_dataloader, verify_frozen_release
from gbm_ai.models.comparison_models import GBMBenchmarkModel, SUPPORTED_BASELINES
from gbm_ai.training.device import resolve_device
from gbm_ai.training.metrics import binary_metrics, bootstrap_metric_ci
from gbm_ai.training.train_baseline import BaselineTrainingConfig, train_baseline


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def predict_fold(
    project_root: Path,
    model_name: str,
    fold: int,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    loader = create_dataloader(
        project_root,
        split="validation",
        fold=fold,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GBMBenchmarkModel(model_name, pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    rows = []
    for batch in loader:
        logits = model(batch["image"].to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy().tolist()
        for sid, cls, target, prob in zip(
            batch["sample_id"],
            batch["class_name"],
            batch["target"].cpu().numpy().tolist(),
            probabilities,
        ):
            rows.append(
                {
                    "sample_id": sid,
                    "class_name": cls,
                    "target": int(target),
                    "probability_gbm": float(prob),
                    "fold": fold,
                }
            )
    return rows


def validate_oof(project_root: Path, folds: list[int], rows: list[dict]) -> None:
    manifest_path, _ = verify_frozen_release(project_root, "classification_v1.0")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        manifest = list(csv.DictReader(f))

    expected = {
        r["sample_id"]
        for r in manifest
        if r["holdout_split"].strip().lower() == "development"
        and int(r["cv_fold"]) in set(folds)
    }
    test_ids = {
        r["sample_id"]
        for r in manifest
        if r["holdout_split"].strip().lower() == "test"
    }
    actual_list = [r["sample_id"] for r in rows]
    actual = set(actual_list)

    if len(actual_list) != len(actual):
        raise RuntimeError("Duplicate samples found in baseline OOF predictions.")
    if actual.intersection(test_ids):
        raise RuntimeError("Locked test leakage detected in baseline OOF predictions.")
    if actual != expected:
        raise RuntimeError(
            f"Baseline OOF coverage mismatch: expected {len(expected)}, got {len(actual)}"
        )


def run(
    project_root: Path,
    model_name: str,
    folds: list[int],
    device_name: str,
    batch_size: int,
    num_workers: int,
    warmup_epochs: int,
    finetune_epochs: int,
    patience: int,
    seed: int,
    skip_existing: bool,
    pretrained: bool,
    max_train_batches: int,
    max_val_batches: int,
):
    project_root = project_root.resolve()
    device = resolve_device(device_name)

    all_oof = []
    fold_metrics = []

    for fold in folds:
        experiment_dir = (
            project_root / "artifacts" / "baseline_experiments"
            / f"{model_name}_fold{fold}_seed{seed}"
        )
        checkpoint = experiment_dir / "checkpoints" / "best_model.pt"

        if not (skip_existing and checkpoint.exists()):
            checkpoint = train_baseline(
                project_root,
                BaselineTrainingConfig(
                    model_name=model_name,
                    fold=fold,
                    seed=seed,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    warmup_epochs=warmup_epochs,
                    finetune_epochs=finetune_epochs,
                    patience=patience,
                    max_train_batches=max_train_batches,
                    max_val_batches=max_val_batches,
                ),
                device_name=device_name,
                pretrained=pretrained,
            )

        predictions = predict_fold(
            project_root,
            model_name,
            fold,
            checkpoint,
            device,
            batch_size,
            num_workers,
            seed,
        )
        write_csv(experiment_dir / "validation_predictions.csv", predictions)

        metrics = binary_metrics(
            [r["target"] for r in predictions],
            [r["probability_gbm"] for r in predictions],
            threshold=0.5,
        )
        fold_metrics.append(
            {
                "fold": fold,
                "validation_samples": len(predictions),
                **metrics,
            }
        )
        all_oof.extend(predictions)

    validate_oof(project_root, folds, all_oof)

    out_dir = (
        project_root / "artifacts" / "baseline_comparison"
        / f"{model_name}_seed{seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "oof_predictions.csv", all_oof)
    write_csv(out_dir / "fold_metrics.csv", fold_metrics)

    y = [r["target"] for r in all_oof]
    p = [r["probability_gbm"] for r in all_oof]
    pooled = binary_metrics(y, p, threshold=0.5)

    aucs = np.asarray([r["roc_auc"] for r in fold_metrics], dtype=float)
    prs = np.asarray([r["pr_auc"] for r in fold_metrics], dtype=float)

    summary = {
        "status": "COMPLETE",
        "model_name": model_name,
        "folds": folds,
        "oof_samples": len(all_oof),
        "locked_test_used": False,
        "fold_roc_auc_mean": float(np.nanmean(aucs)),
        "fold_roc_auc_std": float(np.nanstd(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "fold_pr_auc_mean": float(np.nanmean(prs)),
        "fold_pr_auc_std": float(np.nanstd(prs, ddof=1)) if len(prs) > 1 else 0.0,
        "pooled_oof_metrics_at_0_5": pooled,
        "pooled_oof_roc_auc_ci": bootstrap_metric_ci(
            y, p, metric="roc_auc", n_bootstrap=1000, seed=seed
        ),
        "pooled_oof_pr_auc_ci": bootstrap_metric_ci(
            y, p, metric="pr_auc", n_bootstrap=1000, seed=seed
        ),
    }
    (out_dir / "cross_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nBASELINE CROSS-VALIDATION COMPLETE")
    print("=" * 48)
    print(f"Model:                 {model_name}")
    print(f"Folds:                 {folds}")
    print(f"OOF samples:           {len(all_oof)}")
    print(f"Mean fold ROC-AUC:     {summary['fold_roc_auc_mean']:.4f}")
    print(f"Pooled OOF ROC-AUC:    {pooled['roc_auc']:.4f}")
    print(f"Pooled OOF PR-AUC:     {pooled['pr_auc']:.4f}")
    print(f"FN at 0.5:             {pooled['fn']}")
    print(f"FP at 0.5:             {pooled['fp']}")
    print("Locked test used:      NO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model", choices=SUPPORTED_BASELINES, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()

    run(
        Path(args.project_root),
        args.model,
        args.folds,
        args.device,
        args.batch_size,
        args.num_workers,
        args.warmup_epochs,
        args.finetune_epochs,
        args.patience,
        args.seed,
        args.skip_existing,
        not args.no_pretrained,
        args.max_train_batches,
        args.max_val_batches,
    )


if __name__ == "__main__":
    main()
