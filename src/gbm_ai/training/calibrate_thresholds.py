from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

from gbm_ai.data.classification_dataset import verify_frozen_release
from gbm_ai.training.metrics import binary_metrics

EPS = 1e-7


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Required file not found: {path}")
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


def probability_to_logit(probabilities) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(values) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def temperature_scale(probabilities, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = probability_to_logit(probabilities)
    return np.clip(sigmoid(logits / temperature), EPS, 1.0 - EPS)


def fit_temperature(targets, probabilities) -> float:
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.size == 0 or y.shape != p.shape:
        raise ValueError("Invalid targets/probabilities.")
    if len(np.unique(y)) < 2:
        raise ValueError("Temperature fitting requires both classes.")

    logits = probability_to_logit(p)

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        calibrated = sigmoid(logits / temperature)
        return float(log_loss(y, calibrated, labels=[0, 1]))

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(math.exp(result.x))


def load_release_rows(project_root: Path, release_name: str) -> list[dict[str, str]]:
    manifest_path, _ = verify_frozen_release(project_root, release_name)
    return read_csv(manifest_path)


def collect_fold_oof(
    project_root: Path,
    folds: list[int],
    seed: int,
    release_name: str,
) -> list[dict]:
    release_rows = load_release_rows(project_root, release_name)
    release_by_id = {r["sample_id"]: r for r in release_rows}

    all_rows: list[dict] = []
    seen: set[str] = set()

    for fold in folds:
        path = (
            project_root / "artifacts" / "experiments"
            / f"efficientnetv2s_fold{fold}_seed{seed}"
            / "validation_predictions.csv"
        )
        rows = read_csv(path)

        expected_ids = {
            r["sample_id"]
            for r in release_rows
            if r["holdout_split"].strip().lower() == "development"
            and int(r["cv_fold"]) == fold
        }
        actual_ids = {r["sample_id"] for r in rows}
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"Fold {fold} OOF coverage mismatch. "
                f"Missing={sorted(expected_ids-actual_ids)[:10]}, "
                f"extra={sorted(actual_ids-expected_ids)[:10]}"
            )

        for row in rows:
            sid = row["sample_id"]
            if sid in seen:
                raise RuntimeError(f"Duplicate OOF sample found: {sid}")
            manifest_row = release_by_id.get(sid)
            if manifest_row is None:
                raise RuntimeError(f"OOF sample not in frozen release: {sid}")
            if manifest_row["holdout_split"].strip().lower() == "test":
                raise RuntimeError(f"Locked test leakage detected: {sid}")

            seen.add(sid)
            all_rows.append(
                {
                    **row,
                    "fold": int(row["fold"]),
                    "target": int(float(row["target"])),
                    "probability_gbm": float(row["probability_gbm"]),
                }
            )

    expected_all = {
        r["sample_id"]
        for r in release_rows
        if r["holdout_split"].strip().lower() == "development"
        and int(r["cv_fold"]) in set(folds)
    }
    if seen != expected_all:
        raise RuntimeError(
            f"Combined OOF coverage mismatch: expected {len(expected_all)}, got {len(seen)}"
        )

    return all_rows


def cross_fitted_temperature_calibration(
    oof_rows: list[dict],
    folds: list[int],
) -> tuple[list[dict], list[dict]]:
    calibrated_rows: list[dict] = []
    temperature_rows: list[dict] = []

    for held_out_fold in folds:
        fit_rows = [r for r in oof_rows if int(r["fold"]) != held_out_fold]
        apply_rows = [r for r in oof_rows if int(r["fold"]) == held_out_fold]
        if not fit_rows or not apply_rows:
            raise RuntimeError(f"Insufficient rows for fold {held_out_fold}")

        temperature = fit_temperature(
            [r["target"] for r in fit_rows],
            [r["probability_gbm"] for r in fit_rows],
        )
        calibrated = temperature_scale(
            [r["probability_gbm"] for r in apply_rows], temperature
        )

        temperature_rows.append(
            {
                "held_out_fold": held_out_fold,
                "temperature": temperature,
                "calibration_samples_other_folds": len(fit_rows),
                "applied_samples_held_out_fold": len(apply_rows),
            }
        )

        for row, p_cal in zip(apply_rows, calibrated):
            calibrated_rows.append(
                {
                    **row,
                    "probability_gbm_raw": float(row["probability_gbm"]),
                    "probability_gbm_calibrated": float(p_cal),
                }
            )

    calibrated_rows.sort(key=lambda r: (int(r["fold"]), str(r["sample_id"])))
    return calibrated_rows, temperature_rows


def threshold_sweep(targets, probabilities) -> list[dict]:
    rows = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        m = binary_metrics(targets, probabilities, threshold=float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": m["recall_sensitivity"],
                "specificity": m["specificity"],
                "precision_ppv": m["precision_ppv"],
                "negative_predictive_value": m["negative_predictive_value"],
                "f1": m["f1"],
                "accuracy": m["accuracy"],
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
        )
    return rows


def threshold_candidates(sweep_rows: list[dict]) -> list[dict]:
    candidates = []

    for target_sensitivity in (0.90, 0.95):
        feasible = [r for r in sweep_rows if r["sensitivity"] >= target_sensitivity]
        if feasible:
            chosen = max(feasible, key=lambda r: (r["threshold"], r["specificity"]))
            candidates.append(
                {
                    "candidate_role": "T_low_candidate",
                    "criterion": f"sensitivity >= {target_sensitivity:.2f}",
                    **chosen,
                }
            )

    for target_specificity in (0.90, 0.95):
        feasible = [r for r in sweep_rows if r["specificity"] >= target_specificity]
        if feasible:
            chosen = min(feasible, key=lambda r: (r["threshold"], -r["sensitivity"]))
            candidates.append(
                {
                    "candidate_role": "T_high_candidate",
                    "criterion": f"specificity >= {target_specificity:.2f}",
                    **chosen,
                }
            )

    return candidates


def plot_reliability(targets, raw_probs, calibrated_probs, output_path: Path) -> None:
    y = np.asarray(targets, dtype=np.int64)
    raw_true, raw_pred = calibration_curve(
        y, np.asarray(raw_probs), n_bins=10, strategy="quantile"
    )
    cal_true, cal_pred = calibration_curve(
        y, np.asarray(calibrated_probs), n_bins=10, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(raw_pred, raw_true, marker="o", label="Raw OOF")
    ax.plot(cal_pred, cal_true, marker="o", label="Cross-fitted temperature scaled")
    ax.set_xlabel("Mean predicted GBM probability")
    ax.set_ylabel("Observed GBM frequency")
    ax.set_title("OOF Probability Calibration")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(
    project_root: Path,
    folds: list[int],
    seed: int = 42,
    release_name: str = "classification_v1.0",
) -> Path:
    project_root = project_root.resolve()
    oof_rows = collect_fold_oof(project_root, folds, seed, release_name)
    calibrated_rows, temperatures = cross_fitted_temperature_calibration(oof_rows, folds)

    targets = [r["target"] for r in calibrated_rows]
    raw_probs = [r["probability_gbm_raw"] for r in calibrated_rows]
    cal_probs = [r["probability_gbm_calibrated"] for r in calibrated_rows]

    raw_metrics = binary_metrics(targets, raw_probs, threshold=0.5)
    cal_metrics = binary_metrics(targets, cal_probs, threshold=0.5)

    raw_nll = float(log_loss(targets, raw_probs, labels=[0, 1]))
    cal_nll = float(log_loss(targets, cal_probs, labels=[0, 1]))

    sweep = threshold_sweep(targets, cal_probs)
    candidates = threshold_candidates(sweep)

    output_dir = (
        project_root / "artifacts" / "calibration" / f"efficientnetv2s_seed{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "all_folds_oof_predictions.csv", oof_rows)
    write_csv(output_dir / "cross_fitted_calibrated_oof_predictions.csv", calibrated_rows)
    write_csv(output_dir / "cross_fitted_temperatures.csv", temperatures)
    write_csv(output_dir / "threshold_sweep.csv", sweep)
    if candidates:
        write_csv(output_dir / "threshold_candidates.csv", candidates)

    plot_reliability(
        targets, raw_probs, cal_probs, output_dir / "reliability_diagram.png"
    )

    summary = {
        "status": "COMPLETE",
        "model": "EfficientNetV2-S",
        "dataset_release": release_name,
        "seed": seed,
        "folds": folds,
        "oof_samples": len(oof_rows),
        "locked_test_used": False,
        "calibration_method": "cross-fitted temperature scaling",
        "raw_oof": {"negative_log_likelihood": raw_nll, **raw_metrics},
        "cross_fitted_calibrated_oof": {
            "negative_log_likelihood": cal_nll,
            **cal_metrics,
        },
        "threshold_candidates_are_final": False,
        "threshold_note": (
            "Candidate thresholds are analysis aids only. Final T_low/T_high "
            "must be frozen after reviewing full OOF trade-offs and false negatives "
            "with the medical collaborator."
        ),
    }
    summary_path = output_dir / "calibration_threshold_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nPHASE 2 STEP 4 — OOF CALIBRATION & THRESHOLD ANALYSIS")
    print("=" * 62)
    print(f"Folds combined:             {folds}")
    print(f"OOF samples:                {len(oof_rows)}")
    print(f"Raw NLL:                    {raw_nll:.4f}")
    print(f"Calibrated NLL:             {cal_nll:.4f}")
    print(f"Raw Brier:                  {raw_metrics['brier_score']:.4f}")
    print(f"Calibrated Brier:           {cal_metrics['brier_score']:.4f}")
    print(f"Raw ECE:                    {raw_metrics['expected_calibration_error_10bin']:.4f}")
    print(f"Calibrated ECE:             {cal_metrics['expected_calibration_error_10bin']:.4f}")
    print(f"Calibrated ROC-AUC:         {cal_metrics['roc_auc']:.4f}")
    print(f"Calibrated PR-AUC:          {cal_metrics['pr_auc']:.4f}")
    print("Locked test used:           NO")
    print("Final T_low/T_high frozen:  NO")
    print(f"Summary:                    {summary_path}")

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(Path(args.project_root), folds=args.folds, seed=args.seed)


if __name__ == "__main__":
    main()
