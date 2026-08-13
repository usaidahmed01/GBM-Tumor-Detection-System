from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gbm_ai.training.metrics import binary_metrics, bootstrap_metric_ci


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def metric_row(name: str, path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Missing OOF predictions for {name}: {path}")
    rows = read_csv(path)
    targets = [int(float(r["target"])) for r in rows]

    probability_column = (
        "probability_gbm_raw"
        if "probability_gbm_raw" in rows[0]
        else "probability_gbm"
    )
    probs = [float(r[probability_column]) for r in rows]
    metrics = binary_metrics(targets, probs, threshold=0.5)
    ci = bootstrap_metric_ci(
        targets, probs, metric="roc_auc", n_bootstrap=1000, seed=42
    )
    return {
        "model": name,
        "oof_samples": len(rows),
        "roc_auc": metrics["roc_auc"],
        "roc_auc_ci_lower": ci["lower"],
        "roc_auc_ci_upper": ci["upper"],
        "pr_auc": metrics["pr_auc"],
        "sensitivity_at_0_5": metrics["recall_sensitivity"],
        "specificity_at_0_5": metrics["specificity"],
        "false_negatives_at_0_5": metrics["fn"],
        "false_positives_at_0_5": metrics["fp"],
        "brier_score_raw": metrics["brier_score"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    rows = [
        metric_row(
            "efficientnet_v2_s",
            root / "artifacts" / "calibration" / "efficientnetv2s_seed42"
            / "all_folds_oof_predictions.csv",
        ),
        metric_row(
            "efficientnet_b0",
            root / "artifacts" / "baseline_comparison"
            / "efficientnet_b0_seed42" / "oof_predictions.csv",
        ),
        metric_row(
            "convnext_tiny",
            root / "artifacts" / "baseline_comparison"
            / "convnext_tiny_seed42" / "oof_predictions.csv",
        ),
    ]

    out_dir = root / "artifacts" / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "classifier_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # This ranking is evidence support, not an automatic clinical selection.
    ranked = sorted(rows, key=lambda r: (r["roc_auc"], r["pr_auc"]), reverse=True)
    summary = {
        "status": "COMPLETE",
        "locked_test_used": False,
        "comparison_basis": "raw full-development OOF predictions",
        "ranked_by_roc_auc_then_pr_auc": ranked,
        "automatic_final_model_selection": False,
        "note": (
            "Final model selection must also consider sensitivity, calibration, "
            "fold stability, computational cost, and false-negative review."
        ),
    }
    (out_dir / "classifier_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nCLASSIFIER COMPARISON")
    print("=" * 72)
    for r in ranked:
        print(
            f"{r['model']:20s} ROC-AUC={r['roc_auc']:.4f} "
            f"PR-AUC={r['pr_auc']:.4f} "
            f"FN@0.5={r['false_negatives_at_0_5']}"
        )
    print("\nLocked test used: NO")
    print("Automatic final selection: NO")


if __name__ == "__main__":
    main()
