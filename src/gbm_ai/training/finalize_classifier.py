from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

MODEL_PATHS = {
    "efficientnet_v2_s": "artifacts/calibration/efficientnetv2s_seed42/all_folds_oof_predictions.csv",
    "efficientnet_b0": "artifacts/baseline_comparison/efficientnet_b0_seed42/oof_predictions.csv",
    "convnext_tiny": "artifacts/baseline_comparison/convnext_tiny_seed42/oof_predictions.csv",
}
CALIBRATED_V2_PATH = "artifacts/calibration/efficientnetv2s_seed42/cross_fitted_calibrated_oof_predictions.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Required file does not exist: {path}")
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


def prediction_map(rows):
    result = {}
    for row in rows:
        sid = row["sample_id"]
        if sid in result:
            raise RuntimeError(f"Duplicate prediction sample_id: {sid}")
        pkey = "probability_gbm_raw" if "probability_gbm_raw" in row else "probability_gbm"
        result[sid] = (int(float(row["target"])), float(row[pkey]))
    return result


def align_models(model_rows):
    maps = {name: prediction_map(rows) for name, rows in model_rows.items()}
    reference_name = next(iter(maps))
    reference_ids = set(maps[reference_name])
    for name, values in maps.items():
        if set(values) != reference_ids:
            raise RuntimeError(f"OOF sample IDs differ for {name}.")
    sample_ids = sorted(reference_ids)
    targets, probabilities = [], {name: [] for name in maps}
    for sid in sample_ids:
        ref_target = maps[reference_name][sid][0]
        targets.append(ref_target)
        for name, values in maps.items():
            target, probability = values[sid]
            if target != ref_target:
                raise RuntimeError(f"Target disagreement for {sid}.")
            probabilities[name].append(probability)
    return sample_ids, np.asarray(targets, dtype=np.int64), {
        name: np.asarray(values, dtype=np.float64) for name, values in probabilities.items()
    }


def paired_bootstrap_difference(targets, pa, pb, metric, n_bootstrap=5000, seed=42):
    if metric not in {"roc_auc", "pr_auc"}:
        raise ValueError("metric must be roc_auc or pr_auc")
    if not (targets.shape == pa.shape == pb.shape):
        raise ValueError("Input arrays must have identical shape.")
    scorer = roc_auc_score if metric == "roc_auc" else average_precision_score
    point_a = float(scorer(targets, pa))
    point_b = float(scorer(targets, pb))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(targets), size=len(targets))
        y = targets[idx]
        if len(np.unique(y)) < 2:
            continue
        deltas.append(float(scorer(y, pa[idx]) - scorer(y, pb[idx])))
    values = np.asarray(deltas)
    lower, upper = float(np.quantile(values, .025)), float(np.quantile(values, .975))
    return {
        "metric": metric,
        "model_a_score": point_a,
        "model_b_score": point_b,
        "delta_a_minus_b": point_a - point_b,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "n_valid_bootstrap": int(len(values)),
        "challenger_significantly_better": bool(upper < 0.0),
    }


def choose_thresholds(targets, probabilities, min_negative_sensitivity=0.975, min_positive_specificity=0.95):
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    rows = []
    for threshold in np.round(np.arange(.01, 1.00, .01), 2):
        pred = (p >= threshold).astype(np.int64)
        tp = int(((y == 1) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        rows.append({"threshold": float(threshold), "sensitivity": float(sensitivity), "specificity": float(specificity), "tp": tp, "tn": tn, "fp": fp, "fn": fn})
    low_feasible = [r for r in rows if r["sensitivity"] >= min_negative_sensitivity]
    high_feasible = [r for r in rows if r["specificity"] >= min_positive_specificity]
    if not low_feasible or not high_feasible:
        raise RuntimeError("Requested threshold constraints are not feasible.")
    valid_pairs = [
        (low, high)
        for low in low_feasible
        for high in high_feasible
        if low["threshold"] < high["threshold"]
    ]
    if not valid_pairs:
        raise RuntimeError(
            "No ordered T_low/T_high pair satisfies both requested constraints."
        )

    # Choose the narrowest indeterminate interval that still preserves both
    # safety constraints. Ties prefer the higher T_low (better specificity
    # for the not-suspected boundary).
    t_low_row, t_high_row = min(
        valid_pairs,
        key=lambda pair: (
            pair[1]["threshold"] - pair[0]["threshold"],
            -pair[0]["threshold"],
        ),
    )
    t_low, t_high = t_low_row["threshold"], t_high_row["threshold"]
    negative = p <= t_low
    suspected = p >= t_high
    indeterminate = ~(negative | suspected)
    def band(mask):
        total = int(mask.sum())
        return {"total": total, "gbm": int(((y == 1) & mask).sum()), "no_gbm": int(((y == 0) & mask).sum()), "fraction_of_development": float(total / len(y))}
    n, i, s = band(negative), band(indeterminate), band(suspected)
    n["negative_predictive_value"] = float(n["no_gbm"] / n["total"]) if n["total"] else float("nan")
    s["positive_predictive_value"] = float(s["gbm"] / s["total"]) if s["total"] else float("nan")
    return {
        "selection_policy": {
            "constraints": (
                f"T_low sensitivity >= {min_negative_sensitivity:.3f}; "
                f"T_high specificity >= {min_positive_specificity:.3f}; "
                "T_low < T_high"
            ),
            "pair_selection": (
                "Choose the narrowest ordered indeterminate interval satisfying "
                "both constraints; ties prefer the higher T_low."
            ),
        },
        "T_low": t_low,
        "T_high": t_high,
        "T_low_operating_point": t_low_row,
        "T_high_operating_point": t_high_row,
        "three_band_development_behavior": {
            "gbm_not_suspected_band": n,
            "indeterminate_band": i,
            "gbm_suspected_band": s,
        },
        "locked_test_used": False,
        "status": "ENGINEERING_FREEZE_CANDIDATE",
        "clinical_review_required": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-negative-sensitivity", type=float, default=0.975)
    parser.add_argument("--min-positive-specificity", type=float, default=0.95)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    model_rows = {name: read_csv(root / rel) for name, rel in MODEL_PATHS.items()}
    sample_ids, targets, probabilities = align_models(model_rows)

    comparisons = []
    for challenger in ("convnext_tiny", "efficientnet_b0"):
        for metric in ("roc_auc", "pr_auc"):
            comparisons.append({
                "model_a": "efficientnet_v2_s",
                "model_b": challenger,
                **paired_bootstrap_difference(targets, probabilities["efficientnet_v2_s"], probabilities[challenger], metric, 5000, args.seed),
            })

    metrics = {
        name: {
            "roc_auc": float(roc_auc_score(targets, probs)),
            "pr_auc": float(average_precision_score(targets, probs)),
        }
        for name, probs in probabilities.items()
    }
    challenger_proven_superior = any(r["challenger_significantly_better"] for r in comparisons)

    calibrated_rows = read_csv(root / CALIBRATED_V2_PATH)
    if set(r["sample_id"] for r in calibrated_rows) != set(sample_ids):
        raise RuntimeError("Calibrated V2-S OOF set does not match model-comparison OOF set.")
    calibrated_rows.sort(key=lambda r: r["sample_id"])
    cal_targets = np.asarray([int(float(r["target"])) for r in calibrated_rows])
    cal_probs = np.asarray([float(r["probability_gbm_calibrated"]) for r in calibrated_rows])
    threshold_package = choose_thresholds(cal_targets, cal_probs, args.min_negative_sensitivity, args.min_positive_specificity)

    output_dir = root / "artifacts" / "model_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_bootstrap_model_comparison.csv", comparisons)
    (output_dir / "threshold_engineering_freeze_candidate.json").write_text(json.dumps(threshold_package, indent=2), encoding="utf-8")

    selection = {
        "phase": "Phase 2 final classifier selection",
        "status": "SELECTED_WITH_REVIEW_FLAG" if challenger_proven_superior else "SELECTED",
        "selected_architecture": "efficientnet_v2_s",
        "selection_reason": "EfficientNetV2-S has the highest observed full-OOF ROC-AUC and PR-AUC among the three evaluated architectures. Paired bootstrap comparisons quantify uncertainty; no locked-test result is used for selection.",
        "oof_samples": len(sample_ids),
        "model_metrics_raw_oof": metrics,
        "paired_bootstrap_comparisons": comparisons,
        "threshold_engineering_freeze_candidate": threshold_package,
        "locked_test_used": False,
        "model_weights_frozen_for_deployment": False,
        "note": "Phase 3 still adds Grad-CAM, OOD/indeterminate logic, test-time uncertainty, false-negative review and the model card.",
    }
    path = output_dir / "classifier_architecture_freeze.json"
    path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (output_dir / "PHASE2_READY_FOR_SAFETY.txt").write_text(
        f"PHASE 2 READY FOR CLASSIFIER SAFETY\nSelected architecture: efficientnet_v2_s\nOOF samples: {len(sample_ids)}\nT_low candidate: {threshold_package['T_low']:.2f}\nT_high candidate: {threshold_package['T_high']:.2f}\nLocked test used: NO\n",
        encoding="utf-8",
    )

    bands = threshold_package["three_band_development_behavior"]
    print("\nPHASE 2 FINAL CLASSIFIER SELECTION")
    print("=" * 58)
    print("Selected architecture:       efficientnet_v2_s")
    print(f"V2-S OOF ROC-AUC:            {metrics['efficientnet_v2_s']['roc_auc']:.4f}")
    print(f"V2-S OOF PR-AUC:             {metrics['efficientnet_v2_s']['pr_auc']:.4f}")
    print(f"Challenger proven superior:  {challenger_proven_superior}")
    print(f"T_low engineering candidate: {threshold_package['T_low']:.2f}")
    print(f"T_high engineering candidate:{threshold_package['T_high']:.2f}")
    print(f"Not-suspected band:          {bands['gbm_not_suspected_band']['total']} cases ({bands['gbm_not_suspected_band']['gbm']} GBM)")
    print(f"Indeterminate band:          {bands['indeterminate_band']['total']} cases ({bands['indeterminate_band']['gbm']} GBM)")
    print(f"Suspected band:              {bands['gbm_suspected_band']['total']} cases ({bands['gbm_suspected_band']['gbm']} GBM)")
    print("Locked test used:            NO")
    print("\nSTATUS: PHASE 2 READY FOR CLASSIFIER SAFETY")
    print(f"Report: {path}")

if __name__ == "__main__":
    main()
