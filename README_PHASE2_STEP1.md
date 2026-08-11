# Phase 2 Step 1 — PyTorch Training Foundation & EfficientNetV2-S Setup

## Purpose
Build the production-style PyTorch classification foundation without performing full model training yet.

## New cumulative dependencies
- torch
- torchvision

## Install
```powershell
pip install -r requirements.txt
```

## PowerShell
```powershell
$env:PYTHONPATH="$PWD\src"
```

## Important-step automated test
Tests do not download pretrained weights:
```powershell
pytest -q tests/test_phase2_training_foundation.py
```

Expected:
`2 passed`

## Real foundation verification using pretrained weights
This may download the official EfficientNetV2-S pretrained weights on first use:
```powershell
python -m gbm_ai.training.verify_foundation --project-root . --fold 0 --batch-size 2 --device auto
```

A CPU-only smoke check can also be forced:
```powershell
python -m gbm_ai.training.verify_foundation --project-root . --fold 0 --batch-size 2 --device cpu
```

For code-only/offline checking without downloading weights:
```powershell
python -m gbm_ai.training.verify_foundation --project-root . --fold 0 --batch-size 2 --device cpu --no-pretrained
```

## Output
`artifacts/reports/phase2_step1_foundation_check.json`

## Design rules
- Data comes only from the frozen `classification_v1.0` release.
- The manifest hash is checked before loading.
- Locked test data never enters train/validation DataLoaders.
- Fold selection is controlled by the frozen CV assignment.
- Model starts with the feature backbone frozen.
- One output logit is used with BCEWithLogitsLoss.
- No random augmentation is applied to validation/test inputs.
- Training uses only conservative augmentation.
- Test set remains untouched until the final selected model is locked.
