# Phase 2 Step 3 — 5-Fold Cross-Validation & Out-of-Fold Evaluation

## New cumulative dependency

`matplotlib`

It is added to the same project-level `requirements.txt`.

## What this step does

- Trains EfficientNetV2-S independently on the requested frozen CV folds.
- Loads each fold's best validation checkpoint.
- Generates one out-of-fold (OOF) probability for every development sample.
- Verifies no locked-test sample enters OOF evaluation.
- Computes fold-level and pooled metrics.
- Computes ROC-AUC and PR-AUC bootstrap confidence intervals.
- Saves preliminary false-negative and false-positive case lists at threshold 0.5.
- Generates a cross-validated ROC figure.
- DOES NOT evaluate or tune on the locked test set.
- DOES NOT choose clinical thresholds yet.

## Install

```powershell
pip install -r requirements.txt
$env:PYTHONPATH="$PWD\src"
```

## Important-step test

```powershell
pytest -q tests/test_phase2_cross_validation.py
```

Expected:

`3 passed`

## Local diagnostic only

Use one fold and very few batches:

```powershell
python -m gbm_ai.training.cross_validation --project-root . --folds 0 --device cpu --batch-size 2 --warmup-epochs 1 --finetune-epochs 1 --max-train-batches 2 --max-val-batches 2
```

This checks orchestration only. Do not report diagnostic metrics as model results.

## Full 5-fold training

Run this on a supported modern CUDA GPU:

```powershell
python -m gbm_ai.training.cross_validation --project-root . --folds 0 1 2 3 4 --device auto --batch-size 8 --warmup-epochs 5 --finetune-epochs 15 --patience 5
```

If checkpoints were already trained with the same configuration and you only want to rebuild OOF evaluation:

```powershell
python -m gbm_ai.training.cross_validation --project-root . --folds 0 1 2 3 4 --device auto --batch-size 8 --skip-existing
```

## Outputs

`artifacts/cross_validation/efficientnetv2s_seed42/`

- `oof_predictions.csv`
- `fold_metrics.csv`
- `cross_validation_summary.json`
- `roc_cross_validation.png`
- `false_negatives_at_0_5.csv` when any exist
- `false_positives_at_0_5.csv` when any exist

Each individual fold experiment also receives:

- `validation_predictions.csv`
- `validation_summary.json`

## Interpretation

The 0.5 threshold is used only for preliminary monitoring. Final `T_low` and `T_high` decision thresholds and probability calibration are a later step.
