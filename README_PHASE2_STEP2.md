# Phase 2 Step 2 — Full Training Engine

No new Python dependency is required beyond the cumulative `requirements.txt`.

## Important design rules

- Train/validation are read from the frozen Phase 1 release.
- The locked test set is never instantiated by the training engine.
- EfficientNetV2-S starts with a frozen backbone.
- Stage A trains the classifier head.
- Stage B progressively unfreezes the last feature blocks.
- BCEWithLogitsLoss is used for numerical stability.
- Optional class weighting is derived from the current training fold only.
- AdamW is used for optimization.
- Gradient norm clipping is applied.
- CUDA AMP is used only when a supported CUDA GPU is actually active.
- Best checkpoint is selected by validation ROC-AUC with validation loss as a tie-breaker.
- Threshold 0.5 metrics are monitoring-only. Clinical thresholds come later.
- Training history and checkpoint metadata are stored for traceability.

## Install/update

```powershell
pip install -r requirements.txt
$env:PYTHONPATH="$PWD\src"
```

## Important-step unit test

```powershell
pytest -q tests/test_phase2_training_engine.py
```

Expected:

`2 passed`

## Local CPU diagnostic run

This proves the engine works without spending hours on full training:

```powershell
python -m gbm_ai.training.train_classifier --project-root . --fold 0 --device cpu --batch-size 2 --warmup-epochs 1 --finetune-epochs 1 --max-train-batches 2 --max-val-batches 2
```

Do NOT treat the diagnostic checkpoint as the final model.

## Full fold-0 training

On a supported modern CUDA GPU:

```powershell
python -m gbm_ai.training.train_classifier --project-root . --fold 0 --device auto --batch-size 8 --warmup-epochs 5 --finetune-epochs 15 --patience 5
```

Adjust batch size downward only if GPU memory is insufficient.

## Outputs

`artifacts/experiments/efficientnetv2s_fold0_seed42/`

contains:

- `checkpoints/best_model.pt`
- `history.csv`
- `training_summary.json`
