from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Required file missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def read_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Required JSON missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "UNAVAILABLE"


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safety_metrics(rows: list[dict[str, str]]) -> dict:
    total = len(rows)
    final_counts = Counter(row["final_safety_state"] for row in rows)
    base_counts = Counter(row["base_probability_state"] for row in rows)

    gbm = [row for row in rows if int(float(row["target"])) == 1]
    no_gbm = [row for row in rows if int(float(row["target"])) == 0]

    def count(group, state):
        return sum(row["final_safety_state"] == state for row in group)

    overrides = sum(as_bool(row["safety_override_applied"]) for row in rows)

    determinate = (
        final_counts.get("GBM_SUSPECTED", 0)
        + final_counts.get("GBM_NOT_SUSPECTED", 0)
    )
    indeterminate = final_counts.get("INDETERMINATE", 0)

    residual_fn = count(gbm, "GBM_NOT_SUSPECTED")
    residual_fp = count(no_gbm, "GBM_SUSPECTED")

    return {
        "total_oof_samples": total,
        "gbm_samples": len(gbm),
        "no_gbm_samples": len(no_gbm),
        "base_state_counts": dict(base_counts),
        "final_state_counts": dict(final_counts),
        "determinate_coverage": determinate / total if total else 0.0,
        "abstention_indeterminate_rate": indeterminate / total if total else 0.0,
        "safety_override_count": overrides,
        "gbm_final_suspected": count(gbm, "GBM_SUSPECTED"),
        "gbm_final_indeterminate": count(gbm, "INDETERMINATE"),
        "gbm_final_not_suspected": residual_fn,
        "no_gbm_final_not_suspected": count(no_gbm, "GBM_NOT_SUSPECTED"),
        "no_gbm_final_indeterminate": count(no_gbm, "INDETERMINATE"),
        "no_gbm_final_suspected": residual_fp,
        "residual_safety_critical_false_negatives": residual_fn,
        "residual_false_positive_suspected_cases": residual_fp,
    }


def validate_locked_test_not_used(named_summaries: dict[str, dict]) -> None:
    offenders = []
    for name, summary in named_summaries.items():
        value = summary.get("locked_test_used")
        if value is not False:
            offenders.append(f"{name}={value!r}")
    if offenders:
        raise RuntimeError(
            "Safety/model development gate found non-false locked_test_used: "
            + ", ".join(offenders)
        )


def review_status(review_rows: list[dict[str, str]]) -> dict:
    statuses = Counter(
        (row.get("medical_review_status") or "PENDING").strip().upper()
        for row in review_rows
    )
    pending = sum(
        count
        for status, count in statuses.items()
        if status in {"", "PENDING", "NOT_REVIEWED"}
    )
    return {
        "case_count": len(review_rows),
        "status_counts": dict(statuses),
        "pending_count": pending,
        "complete": pending == 0,
    }


def checkpoint_inventory(project_root: Path, folds=(0, 1, 2, 3, 4)) -> list[dict]:
    inventory = []
    for fold in folds:
        path = (
            project_root
            / "artifacts"
            / "experiments"
            / f"efficientnetv2s_fold{fold}_seed42"
            / "checkpoints"
            / "best_model.pt"
        )
        if not path.exists():
            raise RuntimeError(f"Selected-model fold checkpoint missing: {path}")
        inventory.append(
            {
                "fold": fold,
                "relative_path": str(path.relative_to(project_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return inventory


def markdown_model_card(card: dict) -> str:
    metrics = card["development_evaluation"]
    safety = card["classifier_safety_evaluation"]
    thresholds = card["thresholds"]
    review = card["false_negative_review"]

    lines = [
        "# Model Card — GBM 2D Classifier + Classifier Safety Layer",
        "",
        f"**Model card version:** {card['model_card_version']}",
        f"**Generated:** {card['generated_at_utc']}",
        f"**Code commit:** `{card['code_commit']}`",
        f"**Selected architecture:** {card['model']['selected_architecture']}",
        f"**Development status:** {card['phase3_status']}",
        "",
        "## Intended Use",
        "",
        card["intended_use"],
        "",
        "## Not Intended For",
        "",
        *[f"- {item}" for item in card["not_intended_for"]],
        "",
        "## Training Data",
        "",
        f"- Frozen release: `{card['training_data']['dataset_release']}`",
        f"- Unique deduplicated samples: {card['training_data']['total_unique_samples']}",
        f"- GBM: {card['training_data']['gbm_samples']}",
        f"- No-GBM: {card['training_data']['no_gbm_samples']}",
        f"- Development: {card['training_data']['development_samples']}",
        f"- Locked test: {card['training_data']['locked_test_samples']} (not used during Phases 2–3)",
        f"- Patient-level separation guaranteed: {card['training_data']['patient_level_split_guaranteed']}",
        "",
        "## Model / Training",
        "",
        "- Architecture: EfficientNetV2-S, ImageNet-pretrained transfer learning.",
        "- Binary one-logit head for project GBM vs no-GBM label.",
        "- Warm-up: classifier head with frozen backbone.",
        "- Fine-tuning: upper feature blocks progressively unfrozen with lower backbone learning rate.",
        "- Loss: BCEWithLogitsLoss with fold-training-only class weighting.",
        "- Optimizer: AdamW.",
        "- Best checkpoint selected on validation ROC-AUC with validation-loss tie-breaker.",
        "",
        "### OOF fold checkpoints",
        "",
        *[
            f"- Fold {item['fold']}: `{item['relative_path']}` — SHA-256 `{item['sha256']}`"
            for item in card["model"]["fold_checkpoints"]
        ],
        "",
        "## Development Evaluation",
        "",
        f"- Raw OOF ROC-AUC: {metrics['roc_auc']:.4f}",
        f"- Raw OOF PR-AUC: {metrics['pr_auc']:.4f}",
        f"- Calibrated OOF ROC-AUC: {metrics['calibrated_roc_auc']:.4f}",
        f"- Calibrated OOF PR-AUC: {metrics['calibrated_pr_auc']:.4f}",
        f"- Raw NLL: {metrics['raw_nll']:.4f}",
        f"- Calibrated NLL: {metrics['calibrated_nll']:.4f}",
        f"- Raw Brier: {metrics['raw_brier']:.4f}",
        f"- Calibrated Brier: {metrics['calibrated_brier']:.4f}",
        f"- Raw ECE: {metrics['raw_ece']:.4f}",
        f"- Calibrated ECE: {metrics['calibrated_ece']:.4f}",
        "",
        "## Thresholds",
        "",
        f"- `T_low = {thresholds['T_low']:.2f}`",
        f"- `T_high = {thresholds['T_high']:.2f}`",
        "- Probability ≤ T_low: GBM not suspected, unless a safety signal downgrades to indeterminate.",
        "- Probability ≥ T_high: GBM suspected, unless a safety signal downgrades to indeterminate.",
        "- Middle band: indeterminate.",
        "- Thresholds were selected using development OOF evidence, not the locked test set.",
        "",
        "## Classifier Safety Evaluation",
        "",
        f"- OOF samples: {safety['total_oof_samples']}",
        f"- Determinate coverage: {safety['determinate_coverage']:.3f}",
        f"- Indeterminate/abstention rate: {safety['abstention_indeterminate_rate']:.3f}",
        f"- Safety overrides to indeterminate: {safety['safety_override_count']}",
        f"- Residual actual-GBM cases ending as GBM not suspected: {safety['residual_safety_critical_false_negatives']}",
        f"- No-GBM cases ending as GBM suspected: {safety['residual_false_positive_suspected_cases']}",
        "",
        "Safety signals currently include:",
        "- calibrated three-band probability;",
        "- deterministic TTA instability / high uncertainty reference;",
        "- internal feature-embedding OOD-likeness;",
        "- Phase 1 image-quality status.",
        "",
        "## Explainability / False-Negative Review",
        "",
        "- Grad-CAM is a supporting 2D explanation aid only; it is not a segmentation mask or anatomical localization.",
        f"- Safety-critical false-negative review cases generated: {review['case_count']}",
        f"- Medical-review pending cases: {review['pending_count']}",
        "",
        "## Failure Modes and Limitations",
        "",
        *[f"- {item}" for item in card["failure_modes_and_limitations"]],
        "",
        "## Fairness / Subgroups",
        "",
        card["fairness_subgroups"],
        "",
        "## Licensing / Dependencies",
        "",
        *[f"- {item}" for item in card["licenses_dependencies"]],
        "",
        "## Clinical Validation Statement",
        "",
        card["clinical_validation_statement"],
        "",
        "## Phase 3 Gate",
        "",
        f"**{card['phase3_status']}**",
        "",
        card["phase3_gate_note"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    release_manifest_path = (
        root / "data" / "releases" / "classification_v1.0"
        / "classification_split_manifest.csv"
    )
    release_json_path = (
        root / "data" / "releases" / "classification_v1.0"
        / "dataset_release.json"
    )
    calibration_summary_path = (
        root / "artifacts" / "calibration" / "efficientnetv2s_seed42"
        / "calibration_threshold_summary.json"
    )
    comparison_json_path = (
        root / "artifacts" / "model_comparison"
        / "classifier_comparison.json"
    )
    architecture_freeze_path = (
        root / "artifacts" / "model_selection"
        / "classifier_architecture_freeze.json"
    )
    threshold_path = (
        root / "artifacts" / "model_selection"
        / "threshold_engineering_freeze_candidate.json"
    )
    gradcam_summary_path = (
        root / "artifacts" / "safety" / "gradcam_false_negative_review"
        / "phase3_step1_summary.json"
    )
    gradcam_review_path = (
        root / "artifacts" / "safety" / "gradcam_false_negative_review"
        / "false_negative_review_manifest.csv"
    )
    tta_summary_path = (
        root / "artifacts" / "safety" / "tta_uncertainty"
        / "phase3_step2_summary.json"
    )
    ood_summary_path = (
        root / "artifacts" / "safety" / "ood_embeddings"
        / "phase3_step3_summary.json"
    )
    fusion_summary_path = (
        root / "artifacts" / "safety" / "fusion"
        / "phase3_step4_summary.json"
    )
    fusion_csv_path = (
        root / "artifacts" / "safety" / "fusion"
        / "oof_safety_fusion.csv"
    )
    safety_policy_path = (
        root / "artifacts" / "safety" / "fusion"
        / "classifier_safety_policy_v1.json"
    )

    release_rows = read_csv(release_manifest_path)
    release_json = read_json(release_json_path)
    calibration = read_json(calibration_summary_path)
    comparison = read_json(comparison_json_path)
    architecture = read_json(architecture_freeze_path)
    threshold_package = read_json(threshold_path)
    gradcam = read_json(gradcam_summary_path)
    gradcam_review = read_csv(gradcam_review_path)
    tta = read_json(tta_summary_path)
    ood = read_json(ood_summary_path)
    fusion = read_json(fusion_summary_path)
    safety_policy = read_json(safety_policy_path)
    fusion_rows = read_csv(fusion_csv_path)

    validate_locked_test_not_used(
        {
            "calibration": calibration,
            "architecture_freeze": architecture,
            "gradcam": gradcam,
            "tta": tta,
            "ood": ood,
            "fusion": fusion,
        }
    )

    if len(fusion_rows) != 196:
        raise RuntimeError(
            f"Expected 196 development OOF safety rows, got {len(fusion_rows)}."
        )

    split_counts = Counter(
        row["holdout_split"].strip().lower() for row in release_rows
    )
    label_counts = Counter(int(float(row["label"])) for row in release_rows)

    review = review_status(gradcam_review)
    safety = safety_metrics(fusion_rows)
    checkpoints = checkpoint_inventory(root)

    raw_metrics = next(
        row for row in comparison["ranked_by_roc_auc_then_pr_auc"]
        if row["model"] == "efficientnet_v2_s"
    )
    raw_cal = calibration["raw_oof"]
    cal_cal = calibration["cross_fitted_calibrated_oof"]

    phase3_status = (
        "PHASE 3 COMPLETE"
        if review["complete"]
        else "PHASE 3 TECHNICAL COMPLETE — MEDICAL FALSE-NEGATIVE REVIEW PENDING"
    )

    card = {
        "model_card_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(root),
        "phase3_status": phase3_status,
        "intended_use": (
            "Academic/production-minded pre-biopsy 2D MRI decision-support "
            "assessment for the project-defined GBM versus no-GBM label, within "
            "the documented image domain and with clinician review."
        ),
        "not_intended_for": [
            "Definitive pathological or molecular diagnosis.",
            "Replacing radiologist interpretation or multidisciplinary review.",
            "Classifying every alternative brain tumor type.",
            "Treatment recommendation, drug selection, or survival prediction.",
            "Physical tumor volume or validated anatomical localization from standalone 2D JPG/JPEG/PNG.",
            "Clinical deployment without prospective/external validation and required governance/regulatory work.",
        ],
        "model": {
            "selected_architecture": architecture["selected_architecture"],
            "architecture_selection_status": architecture["status"],
            "fold_checkpoints": checkpoints,
            "deployment_single_checkpoint_frozen": False,
            "deployment_note": (
                "Phases 2–3 evaluate five fold-specific OOF models. A single "
                "deployment-weight strategy has not yet been frozen."
            ),
        },
        "training_data": {
            "dataset_release": "classification_v1.0",
            "release_metadata": release_json,
            "total_unique_samples": len(release_rows),
            "gbm_samples": label_counts.get(1, 0),
            "no_gbm_samples": label_counts.get(0, 0),
            "development_samples": split_counts.get("development", 0),
            "locked_test_samples": split_counts.get("test", 0),
            "patient_level_split_guaranteed": False,
            "provenance_limitation": (
                "Project labels are accepted from the supplied dataset but are "
                "not independently pathology/molecularly verified in the image files."
            ),
        },
        "development_evaluation": {
            "roc_auc": float(raw_metrics["roc_auc"]),
            "pr_auc": float(raw_metrics["pr_auc"]),
            "roc_auc_ci_lower": float(raw_metrics["roc_auc_ci_lower"]),
            "roc_auc_ci_upper": float(raw_metrics["roc_auc_ci_upper"]),
            "calibrated_roc_auc": float(cal_cal["roc_auc"]),
            "calibrated_pr_auc": float(cal_cal["pr_auc"]),
            "raw_nll": float(raw_cal["negative_log_likelihood"]),
            "calibrated_nll": float(cal_cal["negative_log_likelihood"]),
            "raw_brier": float(raw_cal["brier_score"]),
            "calibrated_brier": float(cal_cal["brier_score"]),
            "raw_ece": float(raw_cal["expected_calibration_error_10bin"]),
            "calibrated_ece": float(
                cal_cal["expected_calibration_error_10bin"]
            ),
        },
        "thresholds": {
            "T_low": float(threshold_package["T_low"]),
            "T_high": float(threshold_package["T_high"]),
            "status": threshold_package["status"],
            "clinical_review_required": threshold_package[
                "clinical_review_required"
            ],
            "selection_policy": threshold_package["selection_policy"],
        },
        "classifier_safety_evaluation": safety,
        "safety_policy": safety_policy,
        "false_negative_review": review,
        "ood": {
            "validated_external_ood_detector": ood[
                "validated_external_ood_detector"
            ],
            "candidate_count": ood["ood_likeness_candidate_count"],
            "method": ood["ood_signal"],
        },
        "tta_uncertainty": {
            "candidate_count": tta[
                "engineering_uncertainty_candidate_cases"
            ],
            "band_instability_cases": tta["band_instability_cases"],
            "tta_variants": tta["tta_variants"],
            "threshold_status": tta["threshold_status"],
        },
        "failure_modes_and_limitations": [
            (
                "Small supplied 2D dataset; generalization to other hospitals, "
                "scanner vendors, protocols, MRI views or preprocessing is not established."
            ),
            (
                "No patient identifiers are available, so patient-level split "
                "separation cannot be guaranteed."
            ),
            (
                "No dedicated external OOD dataset is available; the current "
                "embedding-distance mechanism is an internal OOD-likeness signal."
            ),
            (
                "TTA uncertainty references are internal development OOF "
                "engineering quantiles, not validated clinical thresholds."
            ),
            (
                "Grad-CAM may highlight correlates or artifacts and is not a tumor "
                "boundary, physical measurement or validated anatomical localization."
            ),
            (
                "Low GBM probability does not establish normal brain or absence "
                "of another intracranial abnormality."
            ),
            (
                "Applying the 2D classifier to slices extracted from DICOM/NIfTI "
                "would introduce a separate domain/processing shift requiring validation."
            ),
            (
                "Current Phase 1 quality status is not yet the final clinical "
                "upload-time DICOM/NIfTI/JPG QC implementation."
            ),
            (
                "Five OOF fold checkpoints exist; a final single deployment "
                "checkpoint/ensemble strategy has not yet been frozen."
            ),
        ],
        "fairness_subgroups": (
            "Meaningful age/sex/site/scanner subgroup evaluation is not supported "
            "by the supplied classification dataset because reliable linked subgroup "
            "metadata are unavailable."
        ),
        "licenses_dependencies": [
            "PyTorch / TorchVision are used for the 2D classifier.",
            "Selected classifier architecture uses TorchVision EfficientNetV2-S pretrained weights.",
            "Third-party and dataset license/use terms must be independently verified before public/commercial distribution.",
        ],
        "clinical_validation_statement": (
            "This model and safety layer are an academic/production-minded "
            "prototype evaluated on internal development OOF data. They are not "
            "clinically validated, are not a medical-device approval claim, and "
            "must not be used as a substitute for qualified clinical interpretation."
        ),
        "phase3_gate_note": (
            "Technical classifier-safety artifacts are complete. "
            + (
                "The safety-critical Grad-CAM false-negative review has been completed."
                if review["complete"]
                else (
                    f"{review['pending_count']} safety-critical Grad-CAM "
                    "false-negative review case(s) still require medical-collaborator review."
                )
            )
        ),
    }

    output_dir = root / "artifacts" / "safety" / "model_card"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "model_card_v1.json"
    md_path = output_dir / "MODEL_CARD.md"
    validation_path = output_dir / "phase3_safety_validation.json"
    gate_path = output_dir / "PHASE3_STATUS.txt"

    json_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    md_path.write_text(markdown_model_card(card), encoding="utf-8")

    validation = {
        "phase": "Phase 3 Step 5",
        "status": phase3_status,
        "locked_test_used": False,
        "required_artifacts_present": True,
        "oof_safety_rows": len(fusion_rows),
        "classifier_safety_metrics": safety,
        "false_negative_medical_review": review,
        "model_card_json": str(json_path.relative_to(root)),
        "model_card_markdown": str(md_path.relative_to(root)),
        "next_phase": "Phase 4 — Backend foundation",
        "next_phase_allowed": True,
        "important_note": (
            "Backend foundation may proceed while medical review is pending, "
            "but Phase 3 must not be represented as fully clinically reviewed "
            "until those cases are completed."
        ),
    }
    validation_path.write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )

    gate_text = (
        f"{phase3_status}\n"
        f"OOF safety samples: {len(fusion_rows)}\n"
        f"Residual GBM-not-suspected cases: "
        f"{safety['residual_safety_critical_false_negatives']}\n"
        f"Medical false-negative review pending: {review['pending_count']}\n"
        "Locked test used: NO\n"
        "Next phase allowed: YES — Phase 4 Backend Foundation\n"
    )
    gate_path.write_text(gate_text, encoding="utf-8")

    print("\nPHASE 3 STEP 5 — SAFETY VALIDATION + MODEL CARD")
    print("=" * 62)
    print(f"OOF safety samples:              {len(fusion_rows)}")
    print(
        f"Determinate coverage:            "
        f"{safety['determinate_coverage']:.3f}"
    )
    print(
        f"Indeterminate rate:              "
        f"{safety['abstention_indeterminate_rate']:.3f}"
    )
    print(
        f"Safety overrides:                "
        f"{safety['safety_override_count']}"
    )
    print(
        f"Residual GBM not suspected:      "
        f"{safety['residual_safety_critical_false_negatives']}"
    )
    print(
        f"Medical FN reviews pending:      "
        f"{review['pending_count']}"
    )
    print("Locked test used:                NO")
    print(f"Phase 3 status:                  {phase3_status}")
    print(f"Model card:                      {md_path}")
    print("\nNEXT: Phase 4 — Backend Foundation")


if __name__ == "__main__":
    main()
