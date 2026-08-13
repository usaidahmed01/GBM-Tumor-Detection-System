from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


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


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def probability_state(calibrated_probability: float, t_low: float, t_high: float) -> str:
    if calibrated_probability <= t_low:
        return "GBM_NOT_SUSPECTED"
    if calibrated_probability >= t_high:
        return "GBM_SUSPECTED"
    return "INDETERMINATE"


def fuse_case(
    calibrated_probability: float,
    t_low: float,
    t_high: float,
    qc_status: str,
    uncertainty_candidate: bool,
    band_instability: bool,
    ood_likeness_candidate: bool,
) -> dict:
    """Conservative fusion: safety signals may only downgrade to INDETERMINATE."""
    base_state = probability_state(calibrated_probability, t_low, t_high)
    reasons: list[str] = []

    qc_normalized = qc_status.strip().upper()
    if qc_normalized != "PASS":
        reasons.append(f"QC_{qc_normalized or 'UNKNOWN'}")
    if ood_likeness_candidate:
        reasons.append("OOD_LIKENESS")
    if band_instability:
        reasons.append("TTA_BAND_INSTABILITY")
    if uncertainty_candidate and not band_instability:
        reasons.append("HIGH_TTA_UNCERTAINTY")

    if base_state == "INDETERMINATE":
        reasons.insert(0, "PROBABILITY_MIDDLE_BAND")
        final_state = "INDETERMINATE"
    elif reasons:
        final_state = "INDETERMINATE"
    else:
        final_state = base_state

    return {
        "base_probability_state": base_state,
        "final_safety_state": final_state,
        "safety_override_applied": final_state == "INDETERMINATE" and base_state != "INDETERMINATE",
        "safety_reason_codes": "|".join(reasons),
    }


def index_rows(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in result:
            raise RuntimeError(f"Duplicate sample_id {sample_id!r} in {label}.")
        result[sample_id] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    threshold_path = root / "artifacts" / "model_selection" / "threshold_engineering_freeze_candidate.json"
    if not threshold_path.exists():
        raise RuntimeError("Threshold engineering freeze artifact missing. Complete Phase 2 finalization first.")
    threshold_package = json.loads(threshold_path.read_text(encoding="utf-8"))
    t_low = float(threshold_package["T_low"])
    t_high = float(threshold_package["T_high"])

    calibrated_path = root / "artifacts" / "calibration" / "efficientnetv2s_seed42" / "cross_fitted_calibrated_oof_predictions.csv"
    tta_path = root / "artifacts" / "safety" / "tta_uncertainty" / "oof_tta_uncertainty.csv"
    ood_path = root / "artifacts" / "safety" / "ood_embeddings" / "oof_ood_scores.csv"
    release_path = root / "data" / "releases" / "classification_v1.0" / "classification_split_manifest.csv"

    calibrated = index_rows(read_csv(calibrated_path), "calibrated OOF")
    tta = index_rows(read_csv(tta_path), "TTA OOF")
    ood = index_rows(read_csv(ood_path), "OOD OOF")
    release = index_rows(read_csv(release_path), "frozen release")

    development_ids = {sid for sid, row in release.items() if row["holdout_split"].strip().lower() == "development"}
    for label, mapping in [("calibrated", calibrated), ("TTA", tta), ("OOD", ood)]:
        if set(mapping) != development_ids:
            raise RuntimeError(f"{label} OOF coverage mismatch: expected {len(development_ids)}, got {len(mapping)}")

    fused_rows = []
    for sample_id in sorted(development_ids):
        cal_row, tta_row, ood_row, release_row = calibrated[sample_id], tta[sample_id], ood[sample_id], release[sample_id]
        target = int(float(cal_row["target"]))
        if target != int(float(release_row["label"])):
            raise RuntimeError(f"Target mismatch for {sample_id}")

        probability = float(cal_row["probability_gbm_calibrated"])
        qc_status = release_row["quality_status"]
        uncertainty_candidate = as_bool(tta_row["engineering_uncertainty_candidate"])
        band_instability = as_bool(tta_row["band_instability"])
        ood_candidate = as_bool(ood_row["ood_likeness_candidate"])
        decision = fuse_case(probability, t_low, t_high, qc_status, uncertainty_candidate, band_instability, ood_candidate)

        fused_rows.append({
            "sample_id": sample_id,
            "fold": int(cal_row["fold"]),
            "target": target,
            "ground_truth": "GBM" if target == 1 else "NO_GBM",
            "calibrated_probability_gbm": probability,
            "T_low": t_low,
            "T_high": t_high,
            "quality_status": qc_status,
            "tta_probability_mean": float(tta_row["probability_mean"]),
            "tta_probability_std": float(tta_row["probability_std"]),
            "tta_probability_range": float(tta_row["probability_range"]),
            "tta_band_instability": band_instability,
            "tta_uncertainty_candidate": uncertainty_candidate,
            "mahalanobis_distance": float(ood_row["mahalanobis_distance"]),
            "cosine_knn_distance": float(ood_row["cosine_knn_distance"]),
            "ood_likeness_candidate": ood_candidate,
            **decision,
        })

    output_dir = root / "artifacts" / "safety" / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "oof_safety_fusion.csv", fused_rows)

    residual_false_negatives = [r for r in fused_rows if r["target"] == 1 and r["final_safety_state"] == "GBM_NOT_SUSPECTED"]
    safety_overrides = [r for r in fused_rows if r["safety_override_applied"]]
    if residual_false_negatives:
        write_csv(output_dir / "residual_false_negatives.csv", residual_false_negatives)
    if safety_overrides:
        write_csv(output_dir / "safety_overrides_to_indeterminate.csv", safety_overrides)

    final_counts = Counter(r["final_safety_state"] for r in fused_rows)
    base_counts = Counter(r["base_probability_state"] for r in fused_rows)
    gbm_rows = [r for r in fused_rows if r["target"] == 1]
    no_gbm_rows = [r for r in fused_rows if r["target"] == 0]
    reason_counts = Counter()
    for row in fused_rows:
        for reason in str(row["safety_reason_codes"]).split("|"):
            if reason:
                reason_counts[reason] += 1

    def count_state(rows, state):
        return sum(r["final_safety_state"] == state for r in rows)

    policy = {
        "name": "classifier_safety_fusion_v1",
        "selected_architecture": "efficientnet_v2_s",
        "probability_thresholds": {"T_low": t_low, "T_high": t_high},
        "base_probability_logic": {
            f"p <= {t_low:.2f}": "GBM_NOT_SUSPECTED",
            f"{t_low:.2f} < p < {t_high:.2f}": "INDETERMINATE",
            f"p >= {t_high:.2f}": "GBM_SUSPECTED",
        },
        "safety_downgrade_rules": [
            "quality_status != PASS -> INDETERMINATE",
            "ood_likeness_candidate == true -> INDETERMINATE",
            "tta_band_instability == true -> INDETERMINATE",
            "tta_uncertainty_candidate == true -> INDETERMINATE",
        ],
        "monotonic_safety_rule": "Safety signals may downgrade a determinate probability result to INDETERMINATE; they may never upgrade it to the opposite determinate class.",
        "important_limitations": [
            "OOD-likeness is internally referenced on development OOF embeddings and is not externally validated.",
            "TTA uncertainty cutoffs are internal OOF engineering references, not clinical standards.",
            "Current quality_status comes from the Phase 1 2D dataset audit, not final upload-time clinical QC.",
            "The locked test set remains unused.",
        ],
    }
    (output_dir / "classifier_safety_policy_v1.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    summary = {
        "phase": "Phase 3 Step 4",
        "status": "COMPLETE",
        "oof_samples": len(fused_rows),
        "locked_test_used": False,
        "base_probability_state_counts": dict(base_counts),
        "final_safety_state_counts": dict(final_counts),
        "safety_override_count": len(safety_overrides),
        "safety_reason_counts": dict(reason_counts),
        "gbm_ground_truth": {
            "total": len(gbm_rows),
            "final_gbm_suspected": count_state(gbm_rows, "GBM_SUSPECTED"),
            "final_indeterminate": count_state(gbm_rows, "INDETERMINATE"),
            "final_gbm_not_suspected": count_state(gbm_rows, "GBM_NOT_SUSPECTED"),
        },
        "no_gbm_ground_truth": {
            "total": len(no_gbm_rows),
            "final_gbm_not_suspected": count_state(no_gbm_rows, "GBM_NOT_SUSPECTED"),
            "final_indeterminate": count_state(no_gbm_rows, "INDETERMINATE"),
            "final_gbm_suspected": count_state(no_gbm_rows, "GBM_SUSPECTED"),
        },
        "residual_safety_critical_false_negatives": len(residual_false_negatives),
        "important_interpretation": "The safety layer is allowed to abstain. More indeterminate cases can be appropriate when uncertainty/OOD/QC signals indicate a determinate classifier output should not be trusted.",
    }
    summary_path = output_dir / "phase3_step4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nPHASE 3 STEP 4 — UNIFIED CLASSIFIER SAFETY FUSION")
    print("=" * 66)
    print(f"OOF samples:                  {len(fused_rows)}")
    print(f"T_low / T_high:              {t_low:.2f} / {t_high:.2f}")
    print(f"Base not suspected:          {base_counts.get('GBM_NOT_SUSPECTED', 0)}")
    print(f"Base indeterminate:          {base_counts.get('INDETERMINATE', 0)}")
    print(f"Base suspected:              {base_counts.get('GBM_SUSPECTED', 0)}")
    print(f"Final not suspected:         {final_counts.get('GBM_NOT_SUSPECTED', 0)}")
    print(f"Final indeterminate:         {final_counts.get('INDETERMINATE', 0)}")
    print(f"Final suspected:             {final_counts.get('GBM_SUSPECTED', 0)}")
    print(f"Safety overrides:            {len(safety_overrides)}")
    print(f"Residual GBM not suspected:  {len(residual_false_negatives)}")
    print("Locked test used:            NO")
    print(f"Summary:                     {summary_path}")
    print("\nSTATUS: SAFETY FUSION READY FOR PHASE 3 VALIDATION")


if __name__ == "__main__":
    main()
