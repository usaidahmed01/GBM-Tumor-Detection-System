from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from gbm_ai.data.classification_dataset import create_dataloader, verify_frozen_release
from gbm_ai.models.efficientnet_v2_gbm import GBMEfficientNetV2S
from gbm_ai.training.device import describe_device, resolve_device
from gbm_ai.training.metrics import binary_metrics
from gbm_ai.training.reproducibility import seed_everything


@dataclass
class TrainingConfig:
    release_name: str = "classification_v1.0"
    fold: int = 0
    seed: int = 42

    batch_size: int = 8
    num_workers: int = 0

    warmup_epochs: int = 5
    finetune_epochs: int = 15
    finetune_blocks: int = 3

    head_lr_warmup: float = 1e-3
    head_lr_finetune: float = 1e-4
    backbone_lr_finetune: float = 1e-5
    weight_decay: float = 1e-4

    patience: int = 5
    min_delta_auc: float = 1e-4
    monitor_threshold: float = 0.5

    use_class_weight: bool = True
    amp: bool = True

    # Diagnostic limits only. Keep 0 for full production training.
    max_train_batches: int = 0
    max_val_batches: int = 0


def _targets_from_dataset(dataset) -> list[int]:
    return [int(float(row["label"])) for row in dataset.rows]


def _build_loss(train_dataset, device: torch.device, use_class_weight: bool) -> nn.Module:
    if not use_class_weight:
        return nn.BCEWithLogitsLoss()

    labels = _targets_from_dataset(train_dataset)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError("Training fold must contain both GBM and no-GBM samples.")

    # PyTorch BCEWithLogitsLoss pos_weight: N_negative / N_positive.
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled,
    )


def _run_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    amp_enabled: bool,
    max_batches: int = 0,
) -> dict:
    training = optimizer is not None
    model.train(training)

    losses: list[float] = []
    targets: list[float] = []
    probabilities: list[float] = []

    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with _autocast_context(device, amp_enabled):
                logits = model(images)
                loss = loss_fn(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered.")

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        losses.append(float(loss.detach().cpu()))
        targets.extend(y.detach().cpu().numpy().tolist())
        probabilities.extend(torch.sigmoid(logits.detach()).cpu().numpy().tolist())

    if not losses:
        raise RuntimeError("Epoch produced no batches.")

    return {
        "loss": float(np.mean(losses)),
        "targets": targets,
        "probabilities": probabilities,
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainingConfig,
    stage: str,
    epoch: int,
    val_metrics: dict,
    dataset_metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "EfficientNetV2-S",
            "task": "GBM_vs_no_GBM",
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "stage": stage,
            "epoch": epoch,
            "validation_metrics": val_metrics,
            "dataset_release": dataset_metadata.get("release_name", config.release_name),
            "frozen_manifest_sha256": dataset_metadata.get("frozen_manifest_sha256"),
            "saved_at_unix": time.time(),
        },
        path,
    )


def _history_row(
    global_epoch: int,
    stage: str,
    train_result: dict,
    val_result: dict,
    threshold: float,
    optimizer: torch.optim.Optimizer,
) -> dict:
    train_metrics = binary_metrics(
        train_result["targets"], train_result["probabilities"], threshold
    )
    val_metrics = binary_metrics(
        val_result["targets"], val_result["probabilities"], threshold
    )

    return {
        "global_epoch": global_epoch,
        "stage": stage,
        "train_loss": train_result["loss"],
        "val_loss": val_result["loss"],
        "train_roc_auc": train_metrics["roc_auc"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_sensitivity_at_0_5": val_metrics["recall_sensitivity"],
        "val_specificity_at_0_5": val_metrics["specificity"],
        "val_f1_at_0_5": val_metrics["f1"],
        "head_lr": optimizer.param_groups[-1]["lr"],
        "backbone_lr": optimizer.param_groups[0]["lr"]
        if len(optimizer.param_groups) > 1
        else 0.0,
    }


def train(
    project_root: Path,
    config: TrainingConfig,
    device_name: str = "auto",
    pretrained: bool = True,
) -> Path:
    seed_everything(config.seed)
    project_root = project_root.resolve()

    _, dataset_metadata = verify_frozen_release(project_root, config.release_name)
    device = resolve_device(device_name)

    train_loader = create_dataloader(
        project_root,
        split="train",
        fold=config.fold,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        release_name=config.release_name,
    )
    val_loader = create_dataloader(
        project_root,
        split="validation",
        fold=config.fold,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        release_name=config.release_name,
    )

    # IMPORTANT: No test DataLoader is created here.
    # The locked test set is not part of model fitting/model selection.

    model = GBMEfficientNetV2S(
        pretrained=pretrained,
        freeze_backbone=True,
    ).to(device)

    loss_fn = _build_loss(train_loader.dataset, device, config.use_class_weight)

    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    experiment_dir = (
        project_root
        / "artifacts"
        / "experiments"
        / f"efficientnetv2s_fold{config.fold}_seed{config.seed}"
    )
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "best_model.pt"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_auc = -math.inf
    best_val_loss = math.inf
    epochs_without_improvement = 0
    global_epoch = 0
    best_stage = ""
    best_epoch = -1

    def run_stage(
        stage: str,
        epochs: int,
        optimizer,
        scheduler,
        allow_early_stop: bool,
    ) -> bool:
        nonlocal best_auc, best_val_loss, epochs_without_improvement
        nonlocal global_epoch, best_stage, best_epoch

        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1

            train_result = _run_epoch(
                model,
                train_loader,
                loss_fn,
                device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                max_batches=config.max_train_batches,
            )
            val_result = _run_epoch(
                model,
                val_loader,
                loss_fn,
                device,
                optimizer=None,
                scaler=None,
                amp_enabled=False,
                max_batches=config.max_val_batches,
            )

            row = _history_row(
                global_epoch,
                stage,
                train_result,
                val_result,
                config.monitor_threshold,
                optimizer,
            )
            history.append(row)

            val_auc = row["val_roc_auc"]
            val_loss = row["val_loss"]
            improved = (
                (not math.isnan(val_auc) and val_auc > best_auc + config.min_delta_auc)
                or (
                    not math.isnan(val_auc)
                    and abs(val_auc - best_auc) <= config.min_delta_auc
                    and val_loss < best_val_loss
                )
            )

            if improved or best_epoch == -1:
                best_auc = val_auc if not math.isnan(val_auc) else best_auc
                best_val_loss = val_loss
                best_stage = stage
                best_epoch = global_epoch
                epochs_without_improvement = 0

                val_metrics = binary_metrics(
                    val_result["targets"],
                    val_result["probabilities"],
                    config.monitor_threshold,
                )
                val_metrics["loss"] = val_loss

                _save_checkpoint(
                    checkpoint_path,
                    model,
                    config,
                    stage,
                    global_epoch,
                    val_metrics,
                    dataset_metadata,
                )
            else:
                epochs_without_improvement += 1

            if scheduler is not None:
                scheduler.step()

            print(
                f"[{stage}] epoch {stage_epoch}/{epochs} | "
                f"train_loss={row['train_loss']:.4f} | "
                f"val_loss={row['val_loss']:.4f} | "
                f"val_auc={row['val_roc_auc']:.4f}"
            )

            if allow_early_stop and epochs_without_improvement >= config.patience:
                print(
                    f"Early stopping triggered after "
                    f"{epochs_without_improvement} non-improving epochs."
                )
                return True

        return False

    # Stage A: classifier-head warm-up.
    warmup_optimizer = AdamW(
        model.model.classifier.parameters(),
        lr=config.head_lr_warmup,
        weight_decay=config.weight_decay,
    )
    warmup_scheduler = CosineAnnealingLR(
        warmup_optimizer, T_max=max(1, config.warmup_epochs)
    )

    run_stage(
        "warmup",
        config.warmup_epochs,
        warmup_optimizer,
        warmup_scheduler,
        allow_early_stop=False,
    )

    # Stage B: progressive fine-tuning.
    if config.finetune_epochs > 0:
        # Continue fine-tuning from the best warm-up state, not necessarily the last warm-up epoch.
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

        epochs_without_improvement = 0
        model.unfreeze_last_feature_blocks(config.finetune_blocks)

        backbone_params = [
            p for p in model.model.features.parameters() if p.requires_grad
        ]
        head_params = list(model.model.classifier.parameters())

        finetune_optimizer = AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": config.backbone_lr_finetune,
                },
                {
                    "params": head_params,
                    "lr": config.head_lr_finetune,
                },
            ],
            weight_decay=config.weight_decay,
        )
        finetune_scheduler = CosineAnnealingLR(
            finetune_optimizer, T_max=max(1, config.finetune_epochs)
        )

        run_stage(
            "finetune",
            config.finetune_epochs,
            finetune_optimizer,
            finetune_scheduler,
            allow_early_stop=True,
        )

    history_path = experiment_dir / "history.csv"
    if history:
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    summary = {
        "status": "COMPLETE",
        "model": "EfficientNetV2-S",
        "task": "GBM_vs_no_GBM",
        "dataset_release": config.release_name,
        "fold": config.fold,
        "seed": config.seed,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "locked_test_used": False,
        "pretrained_weights": pretrained,
        "device": describe_device(device),
        "amp_enabled": amp_enabled,
        "best_validation_roc_auc": best_auc,
        "best_validation_loss": best_val_loss,
        "best_stage": best_stage,
        "best_global_epoch": best_epoch,
        "checkpoint": str(checkpoint_path.relative_to(project_root)),
        "history": str(history_path.relative_to(project_root)),
        "config": asdict(config),
        "important_note": (
            "Threshold=0.5 metrics are monitoring metrics only. "
            "Clinical thresholds and calibration are selected later using validation data."
        ),
    }
    (experiment_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 2 STEP 2 TRAINING COMPLETE")
    print("=" * 44)
    print(f"Best validation ROC-AUC: {best_auc:.4f}")
    print(f"Best stage/epoch:         {best_stage} / {best_epoch}")
    print(f"Checkpoint:               {checkpoint_path}")
    print("Locked test set used:     NO")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--finetune-blocks", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()

    config = TrainingConfig(
        fold=args.fold,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        finetune_blocks=args.finetune_blocks,
        patience=args.patience,
        amp=not args.no_amp,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    train(
        Path(args.project_root),
        config,
        device_name=args.device,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
