from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from gbm_ai.data.classification_dataset import create_dataloader, verify_frozen_release
from gbm_ai.models.efficientnet_v2_gbm import GBMEfficientNetV2S
from gbm_ai.training.device import describe_device, resolve_device
from gbm_ai.training.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip downloading/loading pretrained weights. Intended only for code smoke tests.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    seed_everything(42)

    manifest, metadata = verify_frozen_release(project_root)
    device = resolve_device(args.device)

    train_loader = create_dataloader(
        project_root,
        split="train",
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        project_root,
        split="validation",
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = create_dataloader(
        project_root,
        split="test",
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = GBMEfficientNetV2S(
        pretrained=not args.no_pretrained,
        freeze_backbone=True,
    ).to(device)

    batch = next(iter(train_loader))
    x = batch["image"].to(device)
    y = batch["target"].to(device)

    model.train()
    logits = model(x)
    loss = nn.BCEWithLogitsLoss()(logits, y)

    if logits.shape != y.shape:
        raise RuntimeError(
            f"Binary output shape mismatch: logits={logits.shape}, target={y.shape}"
        )
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite smoke-test loss.")

    result = {
        "phase": "Phase 2 Step 1",
        "status": "PASS",
        "release": metadata.get("release_name", "classification_v1.0"),
        "manifest": str(manifest.relative_to(project_root)),
        "fold": args.fold,
        "dataset_counts": {
            "train": len(train_loader.dataset),
            "validation": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "input_shape": list(x.shape),
        "output_shape": list(logits.shape),
        "smoke_loss": float(loss.detach().cpu()),
        "pretrained_weights": not args.no_pretrained,
        "model": {
            "name": "EfficientNetV2-S",
            "binary_output_logits": 1,
            "total_parameters": model.total_parameter_count(),
            "trainable_parameters_initial_warmup": model.trainable_parameter_count(),
            "backbone_frozen": True,
        },
        "device": describe_device(device),
    }

    report_dir = project_root / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "phase2_step1_foundation_check.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nPHASE 2 STEP 1 — TRAINING FOUNDATION CHECK")
    print("=" * 50)
    print(f"Release:       {result['release']}")
    print(f"Fold:          {result['fold']}")
    print(f"Train:         {result['dataset_counts']['train']}")
    print(f"Validation:    {result['dataset_counts']['validation']}")
    print(f"Locked test:   {result['dataset_counts']['test']}")
    print(f"Input shape:   {tuple(result['input_shape'])}")
    print(f"Output shape:  {tuple(result['output_shape'])}")
    print(f"Smoke loss:    {result['smoke_loss']:.6f}")
    print(f"Pretrained:    {result['pretrained_weights']}")
    print(f"Device:        {result['device']['device']}")
    print(f"Trainable params (warm-up): {result['model']['trainable_parameters_initial_warmup']:,}")
    print("\nSTATUS: PASS — READY FOR PHASE 2 TRAINING LOOP")


if __name__ == "__main__":
    main()
