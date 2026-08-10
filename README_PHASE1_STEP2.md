# Phase 1 Step 2 — Near-Duplicate Audit

1. Rename/remove the old `requirements-phase1.txt`. Keep only the cumulative `requirements.txt`.
2. Copy the files in this patch into the same paths in your GBM project.
3. Install/update dependencies:
   `pip install -r requirements.txt`
4. Set source path in PowerShell:
   `$env:PYTHONPATH="$PWD\src"`
5. Run:
   `python -m gbm_ai.data.audit_near_duplicates --project-root .`
6. Review:
   `data/reports/phase1_step2_near_duplicate_audit.json`
   `data/manifests/near_duplicate_candidates.csv`
   `data/manifests/near_duplicate_groups.csv`
   `data/manifests/classification_manifest_grouped.csv`
   `data/reports/near_duplicate_review/`
7. Important-step test:
   `pytest -q tests/test_near_duplicate_audit.py`
