# Phase 1 Step 3 — Dataset Standardization & Quality Audit

This step consumes:
`data/manifests/classification_manifest_grouped.csv`

It does NOT split the dataset and does NOT apply random augmentation.

## What it does
- Applies EXIF orientation safely.
- Converts MRI input to grayscale luminance, then 3-channel RGB.
- Preserves anatomy by padding to square before resizing.
- Creates deterministic 384x384 PNG inputs.
- Audits low resolution, extreme aspect ratio, near-blank and low-contrast images.
- Creates a quality manifest and preprocessing configuration.
- Preserves near-duplicate group IDs for the leakage-safe split step.

## Install/update dependencies
`pip install -r requirements.txt`

## PowerShell
`$env:PYTHONPATH="$PWD\src"`

## Run
`python -m gbm_ai.data.standardize_classification_dataset --project-root .`

## Outputs
- `data/processed/classification_384/`
- `data/manifests/classification_quality_manifest.csv`
- `data/reports/phase1_step3_quality_audit.json`
- `data/reports/phase1_step3_preprocessing_config.json`
- `data/reports/phase1_step3_flagged_review.jpg` only when review/reject cases exist

## Important-step test
`pytest -q tests/test_standardize_classification_dataset.py`

Expected test result:
`2 passed`
