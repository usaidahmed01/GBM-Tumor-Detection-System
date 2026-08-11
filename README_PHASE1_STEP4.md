# Phase 1 Step 4 — Leakage-Safe Dataset Splitting

This step creates a locked final test set and 5-fold grouped cross-validation assignments for the development data.

It intentionally does **not** copy images into physical train/val/test folders. The split manifest is the source of truth.

## Input
`data/manifests/classification_quality_manifest.csv`

## Strategy
- Keep near-duplicate groups together.
- Singletons become their own leakage groups.
- Reserve approximately 1/7 (~14.3%) as a locked test set using stratified group splitting.
- Assign the remaining development set to 5 stratified group CV folds.
- During Phase 2, for CV fold `k`: validation = `cv_fold == k`; training = all other development folds.
- Never use the locked test set for training, early stopping, threshold selection, calibration, or model selection.

## Requirements
`pip install -r requirements.txt`

## PowerShell
`$env:PYTHONPATH="$PWD\src"`

## Run
`python -m gbm_ai.data.create_classification_splits --project-root .`

## Outputs
- `data/manifests/classification_split_manifest.csv`
- `data/manifests/classification_cv_summary.csv`
- `data/reports/phase1_step4_split_config.json`
- `data/reports/phase1_step4_split_report.json`

## Important-step test
`pytest -q tests/test_classification_splits.py`

Expected: `2 passed`
