# NeuroGlioma AI — Release Checklist

## Source control

- [ ] `git status` works without index lock errors.
- [ ] `git ls-files frontend/.next` returns no output.
- [ ] `git ls-files frontend/node_modules` returns no output.
- [ ] No `.env`, MRI, DICOM/NIfTI, model weights, local database or runtime storage is tracked.

## Backend

- [ ] `alembic current` is at the single head.
- [ ] `python -m gbm_ai.api.verify_phase10_release` reports READY.
- [ ] `python -m gbm_ai.api.verify_phase10_validation` reports READY foundation.
- [ ] Full Phase 10 validation passes.

## Frontend

- [ ] `npm ci` succeeds from a clean install.
- [ ] `npm run build` succeeds.
- [ ] Homepage, intake, clinical viewer and report are readable at 100% zoom on desktop and responsive on narrow screens.
- [ ] No prototype/debug/internal-ID explanatory copy is present in normal product screens.

## Model/runtime assets

- [ ] Five frozen 2D classifier checkpoints are installed before claiming 2D classifier runtime READY.
- [ ] MONAI bundle checksum/runtime gate passes before 3D inference.
- [ ] Localization atlas/template assets pass their checksum/runtime gate.
- [ ] Missing assets produce a controlled unavailable/indeterminate state.

## Deployment

- [ ] Free-tier-first deployment plan reviewed.
- [ ] Production environment uses `GBM_DEBUG=false`.
- [ ] Secrets are configured outside Git.
- [ ] Public demo contains no real patient data.
- [ ] Free-tier usage/quotas are monitored.

## Final academic boundary

- [ ] No claim of clinical validation, medical-device clearance, autonomous diagnosis or guaranteed clinical performance.
- [ ] Three-state GBM wording is preserved.
- [ ] Final report retains traceability and clinician review state.
