from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from gbm_ai.safety.gradcam import GradCAM, make_review_panel


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Required file missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_safety_critical_false_negatives(
    rows: list[dict],
    t_low: float,
) -> list[dict]:
    selected = []
    for row in rows:
        target = int(float(row["target"]))
        probability = float(row["probability_gbm_calibrated"])
        if target == 1 and probability <= t_low:
            selected.append(row)
    return sorted(selected, key=lambda r: float(r["probability_gbm_calibrated"]))


def checkpoint_for_fold(project_root: Path, fold: int, seed: int = 42) -> Path:
    return (
        project_root
        / "artifacts"
        / "experiments"
        / f"efficientnetv2s_fold{fold}_seed{seed}"
        / "checkpoints"
        / "best_model.pt"
    )


def load_fold_model(checkpoint_path: Path, device: torch.device):
    from gbm_ai.models.efficientnet_v2_gbm import GBMEfficientNetV2S
    if not checkpoint_path.exists():
        raise RuntimeError(f"Fold checkpoint missing: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = GBMEfficientNetV2S(pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    from gbm_ai.data.classification_dataset import build_transform, verify_frozen_release
    from gbm_ai.training.device import resolve_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    device = resolve_device(args.device)

    release_manifest, _ = verify_frozen_release(root, "classification_v1.0")
    release_rows = read_csv(release_manifest)
    release_by_id = {row["sample_id"]: row for row in release_rows}

    thresholds_path = (
        root
        / "artifacts"
        / "model_selection"
        / "threshold_engineering_freeze_candidate.json"
    )
    if not thresholds_path.exists():
        raise RuntimeError(
            "Phase 2 threshold freeze artifact is missing. "
            "Run finalize_classifier first."
        )

    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    t_low = float(thresholds["T_low"])
    t_high = float(thresholds["T_high"])

    calibrated_path = (
        root
        / "artifacts"
        / "calibration"
        / "efficientnetv2s_seed42"
        / "cross_fitted_calibrated_oof_predictions.csv"
    )
    calibrated_rows = read_csv(calibrated_path)

    cases = select_safety_critical_false_negatives(calibrated_rows, t_low)
    if not cases:
        raise RuntimeError(
            "No safety-critical false negatives found at the current T_low."
        )

    output_dir = root / "artifacts" / "safety" / "gradcam_false_negative_review"
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform(training=False)
    review_rows: list[dict] = []
    loaded_models: dict[int, torch.nn.Module] = {}

    for case in cases:
        sample_id = case["sample_id"]
        fold = int(case["fold"])

        release_row = release_by_id.get(sample_id)
        if release_row is None:
            raise RuntimeError(f"Sample not found in frozen release: {sample_id}")
        if release_row["holdout_split"].strip().lower() != "development":
            raise RuntimeError(f"Locked-test sample reached review: {sample_id}")
        if int(release_row["cv_fold"]) != fold:
            raise RuntimeError(
                f"OOF fold mismatch for {sample_id}: "
                f"prediction fold={fold}, release fold={release_row['cv_fold']}"
            )

        if fold not in loaded_models:
            loaded_models[fold] = load_fold_model(
                checkpoint_for_fold(root, fold, args.seed),
                device,
            )

        model = loaded_models[fold]
        target_layer = model.model.features[-1]

        image_path = root / release_row["standardized_relative_path"]
        if not image_path.exists():
            raise RuntimeError(f"Standardized MRI missing: {image_path}")

        with Image.open(image_path) as im:
            original = im.convert("RGB")
            tensor = transform(original).unsqueeze(0).to(device)

        with GradCAM(model, target_layer) as gradcam:
            result = gradcam.generate(tensor)

        panel = make_review_panel(original, result.cam)
        panel_path = image_dir / f"{sample_id}_gradcam_review.png"
        panel.save(panel_path)

        raw_probability = float(
            case.get("probability_gbm_raw", case.get("probability_gbm", "nan"))
        )
        calibrated_probability = float(case["probability_gbm_calibrated"])

        review_rows.append(
            {
                "sample_id": sample_id,
                "fold": fold,
                "ground_truth": "GBM",
                "raw_probability_gbm": raw_probability,
                "calibrated_probability_gbm": calibrated_probability,
                "T_low": t_low,
                "T_high": t_high,
                "current_three_band_state": "GBM_NOT_SUSPECTED",
                "review_priority": "SAFETY_CRITICAL_FALSE_NEGATIVE",
                "standardized_image": release_row["standardized_relative_path"],
                "gradcam_review_image": str(panel_path.relative_to(root)),
                "cam_peak_x_normalized": result.peak_x_normalized,
                "cam_peak_y_normalized": result.peak_y_normalized,
                "cam_border_energy_fraction": result.border_energy_fraction,
                "medical_review_status": "PENDING",
                "reviewer": "",
                "gradcam_focus_assessment": "",
                "possible_failure_mode": "",
                "clinician_comment": "",
                "action_after_review": "",
            }
        )

    write_csv(output_dir / "false_negative_review_manifest.csv", review_rows)

    summary = {
        "phase": "Phase 3 Step 1",
        "status": "GENERATED_FOR_MEDICAL_REVIEW",
        "selected_architecture": "efficientnet_v2_s",
        "dataset_release": "classification_v1.0",
        "T_low": t_low,
        "T_high": t_high,
        "safety_critical_false_negative_count": len(review_rows),
        "locked_test_used": False,
        "gradcam_role": (
            "Supporting 2D classifier visualization only. "
            "Not a tumor boundary, physical measurement, or validated anatomical localization."
        ),
        "review_requirement": (
            "Review each case with the medical collaborator and record whether "
            "attention is plausibly intracranial/lesion-related, border/background-driven, "
            "diffuse/unclear, or otherwise suspicious."
        ),
        "methodology": (
            "Each OOF case uses its own fold checkpoint, which did not train on "
            "that validation image."
        ),
    }
    (output_dir / "phase3_step1_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 3 STEP 1 — GRAD-CAM FALSE-NEGATIVE REVIEW")
    print("=" * 60)
    print("Selected architecture:      EfficientNetV2-S")
    print(f"T_low:                      {t_low:.2f}")
    print(f"T_high:                     {t_high:.2f}")
    print(f"Safety-critical FN cases:   {len(review_rows)}")
    print("Fold-specific OOF models:   YES")
    print("Locked test used:           NO")
    print("Medical review status:      PENDING")
    print(f"Review folder:              {output_dir}")
    print("\nIMPORTANT: Grad-CAM is an explanation aid, not a segmentation mask.")


if __name__ == "__main__":
    main()
