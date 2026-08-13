# Model Card — GBM 2D Classifier + Classifier Safety Layer

**Model card version:** 1.0
**Generated:** 2026-08-13T20:36:58.842172+00:00
**Code commit:** `e437622c948617deee489ff63d7f31ced66afdfe`
**Selected architecture:** efficientnet_v2_s
**Development status:** PHASE 3 TECHNICAL COMPLETE — MEDICAL FALSE-NEGATIVE REVIEW PENDING

## Intended Use

Academic/production-minded pre-biopsy 2D MRI decision-support assessment for the project-defined GBM versus no-GBM label, within the documented image domain and with clinician review.

## Not Intended For

- Definitive pathological or molecular diagnosis.
- Replacing radiologist interpretation or multidisciplinary review.
- Classifying every alternative brain tumor type.
- Treatment recommendation, drug selection, or survival prediction.
- Physical tumor volume or validated anatomical localization from standalone 2D JPG/JPEG/PNG.
- Clinical deployment without prospective/external validation and required governance/regulatory work.

## Training Data

- Frozen release: `classification_v1.0`
- Unique deduplicated samples: 228
- GBM: 141
- No-GBM: 87
- Development: 196
- Locked test: 32 (not used during Phases 2–3)
- Patient-level separation guaranteed: False

## Model / Training

- Architecture: EfficientNetV2-S, ImageNet-pretrained transfer learning.
- Binary one-logit head for project GBM vs no-GBM label.
- Warm-up: classifier head with frozen backbone.
- Fine-tuning: upper feature blocks progressively unfrozen with lower backbone learning rate.
- Loss: BCEWithLogitsLoss with fold-training-only class weighting.
- Optimizer: AdamW.
- Best checkpoint selected on validation ROC-AUC with validation-loss tie-breaker.

### OOF fold checkpoints

- Fold 0: `artifacts\experiments\efficientnetv2s_fold0_seed42\checkpoints\best_model.pt` — SHA-256 `e996bea99a7b1e2ca60e96fc8b5dc783afc0d6671325bb83486d458819f2520c`
- Fold 1: `artifacts\experiments\efficientnetv2s_fold1_seed42\checkpoints\best_model.pt` — SHA-256 `3ed6b41b22056898ef4c4c48daa5dd4368a9389ef4ed5d831573f027f9b62e08`
- Fold 2: `artifacts\experiments\efficientnetv2s_fold2_seed42\checkpoints\best_model.pt` — SHA-256 `1ab1059a0e1f5bbf00123c1059ffa81cdd876cf0e7f95b53183fd23d29a1d247`
- Fold 3: `artifacts\experiments\efficientnetv2s_fold3_seed42\checkpoints\best_model.pt` — SHA-256 `6c25e266adf89e0a1bee72dbbb0392a456e4ec1235969d8c9fce1d94c6aa5a91`
- Fold 4: `artifacts\experiments\efficientnetv2s_fold4_seed42\checkpoints\best_model.pt` — SHA-256 `9f4952120de8e202c2b795f41f7dc96999bec1046b6dee5e9b76eba59c81b106`

## Development Evaluation

- Raw OOF ROC-AUC: 0.9334
- Raw OOF PR-AUC: 0.9528
- Calibrated OOF ROC-AUC: 0.9319
- Calibrated OOF PR-AUC: 0.9514
- Raw NLL: 0.4509
- Calibrated NLL: 0.4335
- Raw Brier: 0.1442
- Calibrated Brier: 0.1412
- Raw ECE: 0.1858
- Calibrated ECE: 0.1499

## Thresholds

- `T_low = 0.13`
- `T_high = 0.57`
- Probability ≤ T_low: GBM not suspected, unless a safety signal downgrades to indeterminate.
- Probability ≥ T_high: GBM suspected, unless a safety signal downgrades to indeterminate.
- Middle band: indeterminate.
- Thresholds were selected using development OOF evidence, not the locked test set.

## Classifier Safety Evaluation

- OOF samples: 196
- Determinate coverage: 0.454
- Indeterminate/abstention rate: 0.546
- Safety overrides to indeterminate: 36
- Residual actual-GBM cases ending as GBM not suspected: 2
- No-GBM cases ending as GBM suspected: 2

Safety signals currently include:
- calibrated three-band probability;
- deterministic TTA instability / high uncertainty reference;
- internal feature-embedding OOD-likeness;
- Phase 1 image-quality status.

## Explainability / False-Negative Review

- Grad-CAM is a supporting 2D explanation aid only; it is not a segmentation mask or anatomical localization.
- Safety-critical false-negative review cases generated: 3
- Medical-review pending cases: 3

## Failure Modes and Limitations

- Small supplied 2D dataset; generalization to other hospitals, scanner vendors, protocols, MRI views or preprocessing is not established.
- No patient identifiers are available, so patient-level split separation cannot be guaranteed.
- No dedicated external OOD dataset is available; the current embedding-distance mechanism is an internal OOD-likeness signal.
- TTA uncertainty references are internal development OOF engineering quantiles, not validated clinical thresholds.
- Grad-CAM may highlight correlates or artifacts and is not a tumor boundary, physical measurement or validated anatomical localization.
- Low GBM probability does not establish normal brain or absence of another intracranial abnormality.
- Applying the 2D classifier to slices extracted from DICOM/NIfTI would introduce a separate domain/processing shift requiring validation.
- Current Phase 1 quality status is not yet the final clinical upload-time DICOM/NIfTI/JPG QC implementation.
- Five OOF fold checkpoints exist; a final single deployment checkpoint/ensemble strategy has not yet been frozen.

## Fairness / Subgroups

Meaningful age/sex/site/scanner subgroup evaluation is not supported by the supplied classification dataset because reliable linked subgroup metadata are unavailable.

## Licensing / Dependencies

- PyTorch / TorchVision are used for the 2D classifier.
- Selected classifier architecture uses TorchVision EfficientNetV2-S pretrained weights.
- Third-party and dataset license/use terms must be independently verified before public/commercial distribution.

## Clinical Validation Statement

This model and safety layer are an academic/production-minded prototype evaluated on internal development OOF data. They are not clinically validated, are not a medical-device approval claim, and must not be used as a substitute for qualified clinical interpretation.

## Phase 3 Gate

**PHASE 3 TECHNICAL COMPLETE — MEDICAL FALSE-NEGATIVE REVIEW PENDING**

Technical classifier-safety artifacts are complete. 3 safety-critical Grad-CAM false-negative review case(s) still require medical-collaborator review.
