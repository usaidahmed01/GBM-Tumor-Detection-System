# Phase 1 — Step 1: Dataset Setup, Integrity Validation & Exact Deduplication

This step uses **only** `brain_tumor_dataset/yes` and `brain_tumor_dataset/no` from the source archive.

Project label definition:
- `yes` = GBM (label 1)
- `no` = No GBM (label 0)

The source ZIP is treated as immutable. The script extracts a raw copy, verifies image readability, computes SHA-256 hashes, removes exact within-class duplicate copies from the clean dataset, fails on cross-class exact duplicates, and emits manifests/audit reports.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-phase1.txt
$env:PYTHONPATH = "$PWD\src"
python -m gbm_ai.data.prepare_classification_dataset --archive data/source/archive.zip --project-root .
pytest -q
```

## Expected real-dataset audit

- Source files: 253
- GBM (`yes`): 155
- No-GBM (`no`): 98
- Unique GBM: 141
- Unique No-GBM: 87
- Unique total: 228
- Exact duplicate extras: 25
- Cross-class exact duplicates: 0
