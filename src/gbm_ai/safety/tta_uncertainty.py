from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TTASpec:
    name: str
    angle: float = 0.0
    translate_x_fraction: float = 0.0
    translate_y_fraction: float = 0.0


TTA_SPECS = (
    TTASpec("identity"),
    TTASpec("rotate_minus_3", angle=-3.0),
    TTASpec("rotate_plus_3", angle=3.0),
    TTASpec("translate_left_2pct", translate_x_fraction=-0.02),
    TTASpec("translate_right_2pct", translate_x_fraction=0.02),
    TTASpec("translate_up_2pct", translate_y_fraction=-0.02),
    TTASpec("translate_down_2pct", translate_y_fraction=0.02),
)


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


def probability_to_logit(probability: float) -> float:
    p = min(max(float(probability), 1e-7), 1.0 - 1e-7)
    return math.log(p / (1.0 - p))


def calibrated_probability(probability: float, temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logit = probability_to_logit(probability) / temperature
    logit = max(min(logit, 60.0), -60.0)
    return float(1.0 / (1.0 + math.exp(-logit)))


def probability_entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return float(-(p * math.log(p) + (1 - p) * math.log(1 - p)))


def three_band_state(probability: float, t_low: float, t_high: float) -> str:
    if probability <= t_low:
        return "GBM_NOT_SUSPECTED"
    if probability >= t_high:
        return "GBM_SUSPECTED"
    return "INDETERMINATE"


def summarize_tta_probabilities(
    calibrated_probabilities: list[float],
    t_low: float,
    t_high: float,
) -> dict:
    if not calibrated_probabilities:
        raise ValueError("No TTA probabilities supplied.")

    values = np.asarray(calibrated_probabilities, dtype=np.float64)
    states = [three_band_state(float(p), t_low, t_high) for p in values]
    unique_states = sorted(set(states))

    mean_p = float(values.mean())
    std_p = float(values.std(ddof=0))
    min_p = float(values.min())
    max_p = float(values.max())

    return {
        "tta_count": int(len(values)),
        "probability_mean": mean_p,
        "probability_std": std_p,
        "probability_min": min_p,
        "probability_max": max_p,
        "probability_range": float(max_p - min_p),
        "predictive_entropy_mean_probability": probability_entropy(mean_p),
        "mean_three_band_state": three_band_state(mean_p, t_low, t_high),
        "tta_unique_state_count": len(unique_states),
        "tta_states_seen": "|".join(unique_states),
        "band_instability": len(unique_states) > 1,
        "crosses_T_low": bool(min_p <= t_low < max_p),
        "crosses_T_high": bool(min_p < t_high <= max_p),
    }


def apply_tta(image: Image.Image, spec: TTASpec) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    translate = [
        int(round(width * spec.translate_x_fraction)),
        int(round(height * spec.translate_y_fraction)),
    ]

    return TF.affine(
        image,
        angle=spec.angle,
        translate=translate,
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )


def image_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor


def load_fold_temperature(project_root: Path) -> dict[int, float]:
    path = (
        project_root
        / "artifacts"
        / "calibration"
        / "efficientnetv2s_seed42"
        / "cross_fitted_temperatures.csv"
    )
    rows = read_csv(path)
    result = {}
    for row in rows:
        result[int(row["held_out_fold"])] = float(row["temperature"])
    return result


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
        raise RuntimeError(f"Checkpoint missing: {checkpoint_path}")

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


@torch.inference_mode()
def predict_tta_set(
    model,
    original: Image.Image,
    device: torch.device,
    temperature: float,
) -> tuple[list[float], list[float]]:
    tensors = [
        image_to_normalized_tensor(apply_tta(original, spec))
        for spec in TTA_SPECS
    ]
    batch = torch.stack(tensors, dim=0).to(device)

    logits = model(batch)
    raw_probabilities = torch.sigmoid(logits).detach().cpu().numpy().tolist()
    calibrated = [
        calibrated_probability(float(p), temperature)
        for p in raw_probabilities
    ]
    return [float(p) for p in raw_probabilities], calibrated


def derive_uncertainty_reference(rows: list[dict]) -> dict:
    std_values = np.asarray([float(r["probability_std"]) for r in rows])
    range_values = np.asarray([float(r["probability_range"]) for r in rows])
    entropy_values = np.asarray(
        [float(r["predictive_entropy_mean_probability"]) for r in rows]
    )

    return {
        "method": "development_OOF_quantile_reference",
        "probability_std_q90": float(np.quantile(std_values, 0.90)),
        "probability_std_q95": float(np.quantile(std_values, 0.95)),
        "probability_std_q99": float(np.quantile(std_values, 0.99)),
        "probability_range_q90": float(np.quantile(range_values, 0.90)),
        "probability_range_q95": float(np.quantile(range_values, 0.95)),
        "probability_range_q99": float(np.quantile(range_values, 0.99)),
        "predictive_entropy_q90": float(np.quantile(entropy_values, 0.90)),
        "predictive_entropy_q95": float(np.quantile(entropy_values, 0.95)),
        "predictive_entropy_q99": float(np.quantile(entropy_values, 0.99)),
        "important_note": (
            "These are engineering reference quantiles from OOF development data, "
            "not clinically validated uncertainty thresholds."
        ),
    }


def process_folds(
    project_root: Path,
    folds: list[int],
    device_name: str,
    seed: int,
    max_samples: int,
) -> None:
    from gbm_ai.data.classification_dataset import verify_frozen_release
    from gbm_ai.training.device import resolve_device

    root = project_root.resolve()
    device = resolve_device(device_name)

    manifest_path, _ = verify_frozen_release(root, "classification_v1.0")
    release_rows = read_csv(manifest_path)

    threshold_path = (
        root
        / "artifacts"
        / "model_selection"
        / "threshold_engineering_freeze_candidate.json"
    )
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    t_low = float(thresholds["T_low"])
    t_high = float(thresholds["T_high"])

    temperatures = load_fold_temperature(root)

    output_root = root / "artifacts" / "safety" / "tta_uncertainty"
    per_fold_dir = output_root / "per_fold"
    per_fold_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        if fold not in temperatures:
            raise RuntimeError(f"No cross-fitted temperature for fold {fold}.")

        fold_rows = [
            row
            for row in release_rows
            if row["holdout_split"].strip().lower() == "development"
            and int(row["cv_fold"]) == fold
        ]
        if max_samples > 0:
            fold_rows = fold_rows[:max_samples]

        model = load_fold_model(checkpoint_for_fold(root, fold, seed), device)
        output_rows = []

        for index, row in enumerate(fold_rows, start=1):
            image_path = root / row["standardized_relative_path"]
            with Image.open(image_path) as im:
                original = im.convert("RGB")

            raw_probs, calibrated_probs = predict_tta_set(
                model,
                original,
                device,
                temperatures[fold],
            )
            summary = summarize_tta_probabilities(
                calibrated_probs,
                t_low,
                t_high,
            )

            out = {
                "sample_id": row["sample_id"],
                "class_name": row["class_name"],
                "target": int(float(row["label"])),
                "fold": fold,
                "temperature": temperatures[fold],
                "T_low": t_low,
                "T_high": t_high,
                **summary,
            }

            for spec, raw_p, cal_p in zip(
                TTA_SPECS,
                raw_probs,
                calibrated_probs,
            ):
                out[f"raw_{spec.name}"] = raw_p
                out[f"calibrated_{spec.name}"] = cal_p

            output_rows.append(out)

            if index % 10 == 0 or index == len(fold_rows):
                print(
                    f"Fold {fold}: {index}/{len(fold_rows)} "
                    "samples processed"
                )

        suffix = (
            f"_diagnostic_{max_samples}"
            if max_samples > 0
            else ""
        )
        write_csv(
            per_fold_dir / f"fold{fold}_tta_uncertainty{suffix}.csv",
            output_rows,
        )

        print(
            f"Fold {fold} complete | samples={len(output_rows)} | "
            f"band-instability={sum(bool(r['band_instability']) for r in output_rows)}"
        )


def aggregate(
    project_root: Path,
    folds: list[int],
) -> None:
    from gbm_ai.data.classification_dataset import verify_frozen_release

    root = project_root.resolve()
    output_root = root / "artifacts" / "safety" / "tta_uncertainty"
    per_fold_dir = output_root / "per_fold"

    combined = []
    for fold in folds:
        path = per_fold_dir / f"fold{fold}_tta_uncertainty.csv"
        rows = read_csv(path)
        combined.extend(rows)

    manifest_path, _ = verify_frozen_release(root, "classification_v1.0")
    release_rows = read_csv(manifest_path)

    expected_ids = {
        row["sample_id"]
        for row in release_rows
        if row["holdout_split"].strip().lower() == "development"
        and int(row["cv_fold"]) in set(folds)
    }
    actual_ids = [row["sample_id"] for row in combined]
    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError("Duplicate sample IDs in TTA aggregation.")
    if set(actual_ids) != expected_ids:
        raise RuntimeError(
            f"TTA OOF coverage mismatch: expected {len(expected_ids)}, "
            f"got {len(set(actual_ids))}"
        )

    if len(folds) == 5 and set(folds) == {0, 1, 2, 3, 4}:
        if len(combined) != 196:
            raise RuntimeError(
                f"Expected 196 full-development OOF cases, got {len(combined)}"
            )

    reference = derive_uncertainty_reference(combined)

    enriched = []
    for row in combined:
        std_value = float(row["probability_std"])
        range_value = float(row["probability_range"])
        entropy_value = float(row["predictive_entropy_mean_probability"])
        band_instability = str(row["band_instability"]).lower() == "true"

        engineering_uncertainty_candidate = bool(
            band_instability
            or std_value >= reference["probability_std_q95"]
            or range_value >= reference["probability_range_q95"]
        )

        enriched.append(
            {
                **row,
                "above_std_q95": std_value >= reference["probability_std_q95"],
                "above_range_q95": range_value >= reference["probability_range_q95"],
                "above_entropy_q95": (
                    entropy_value >= reference["predictive_entropy_q95"]
                ),
                "engineering_uncertainty_candidate": (
                    engineering_uncertainty_candidate
                ),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "oof_tta_uncertainty.csv", enriched)

    uncertain_rows = [
        row for row in enriched
        if str(row["engineering_uncertainty_candidate"]).lower() == "true"
    ]
    if uncertain_rows:
        write_csv(
            output_root / "uncertainty_review_candidates.csv",
            uncertain_rows,
        )

    summary = {
        "phase": "Phase 3 Step 2",
        "status": "COMPLETE",
        "folds": folds,
        "oof_samples": len(enriched),
        "tta_variants": [spec.name for spec in TTA_SPECS],
        "tta_count_per_sample": len(TTA_SPECS),
        "locked_test_used": False,
        "reference_quantiles": reference,
        "band_instability_cases": sum(
            str(row["band_instability"]).lower() == "true"
            for row in enriched
        ),
        "engineering_uncertainty_candidate_cases": len(uncertain_rows),
        "threshold_status": "REFERENCE_ONLY_NOT_FINAL_SAFETY_GATE",
        "important_note": (
            "Phase 3 Step 4 will combine uncertainty with calibrated probability, "
            "OOD and QC. This step does not independently force a clinical state."
        ),
    }
    (output_root / "phase3_step2_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_root / "uncertainty_reference.json").write_text(
        json.dumps(reference, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 3 STEP 2 — TEST-TIME UNCERTAINTY")
    print("=" * 56)
    print(f"Folds aggregated:                   {folds}")
    print(f"OOF samples:                        {len(enriched)}")
    print(f"TTA variants/sample:                {len(TTA_SPECS)}")
    print(
        f"Band-instability cases:             "
        f"{summary['band_instability_cases']}"
    )
    print(
        f"Engineering uncertainty candidates: "
        f"{summary['engineering_uncertainty_candidate_cases']}"
    )
    print(
        f"Std q95 reference:                  "
        f"{reference['probability_std_q95']:.4f}"
    )
    print(
        f"Range q95 reference:                "
        f"{reference['probability_range_q95']:.4f}"
    )
    print("Locked test used:                   NO")
    print("Final safety uncertainty gate:      NOT YET")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Aggregate already-generated per-fold outputs without inference.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Diagnostic only. 0 means all samples in requested folds.",
    )
    args = parser.parse_args()

    root = Path(args.project_root)
    if args.aggregate_only:
        aggregate(root, args.folds)
    else:
        process_folds(
            root,
            args.folds,
            args.device,
            args.seed,
            args.max_samples,
        )


if __name__ == "__main__":
    main()
