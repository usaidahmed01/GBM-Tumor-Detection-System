from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn



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


def checkpoint_for_fold(project_root: Path, fold: int, seed: int = 42) -> Path:
    return (
        project_root
        / "artifacts"
        / "experiments"
        / f"efficientnetv2s_fold{fold}_seed{seed}"
        / "checkpoints"
        / "best_model.pt"
    )


def load_fold_model(
    checkpoint_path: Path,
    device: torch.device,
):
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


class EfficientNetFeatureExtractor(nn.Module):
    """
    Extract penultimate pooled EfficientNetV2-S embeddings.

    These embeddings are used only for OOD-likeness analysis. They are not
    anatomical coordinates and must not be interpreted as lesion locations.
    """

    def __init__(self, classifier_model: nn.Module) -> None:
        super().__init__()
        self.features = classifier_model.model.features
        self.avgpool = classifier_model.model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


@torch.inference_mode()
def extract_embeddings(
    extractor: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    embeddings = []
    sample_ids = []
    targets = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        z = extractor(x).detach().float().cpu().numpy()

        embeddings.append(z)
        sample_ids.extend(batch["sample_id"])
        targets.extend(batch["target"].cpu().numpy().astype(int).tolist())

    if not embeddings:
        raise RuntimeError("Embedding extraction produced no batches.")

    matrix = np.concatenate(embeddings, axis=0)
    return matrix, sample_ids, np.asarray(targets, dtype=np.int64)


def cosine_knn_distance(
    train_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(k, len(train_embeddings))

    train = train_embeddings.astype(np.float64)
    query = query_embeddings.astype(np.float64)

    train_norm = np.linalg.norm(train, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    train = train / np.clip(train_norm, 1e-12, None)
    query = query / np.clip(query_norm, 1e-12, None)

    similarities = query @ train.T
    nearest = np.partition(similarities, kth=similarities.shape[1] - k, axis=1)[:, -k:]
    return 1.0 - nearest.mean(axis=1)


def fit_ood_reference(
    train_embeddings: np.ndarray,
    pca_components: int = 32,
) -> dict:
    if train_embeddings.ndim != 2:
        raise ValueError("train_embeddings must be 2D.")
    if len(train_embeddings) < 8:
        raise ValueError("Too few training embeddings for OOD reference.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_embeddings)

    max_components = min(
        int(pca_components),
        scaled.shape[1],
        scaled.shape[0] - 1,
    )
    if max_components < 2:
        raise RuntimeError("Insufficient rank for PCA OOD reference.")

    pca = PCA(
        n_components=max_components,
        svd_solver="auto",
        random_state=42,
    )
    reduced = pca.fit_transform(scaled)

    covariance = LedoitWolf().fit(reduced)

    return {
        "scaler": scaler,
        "pca": pca,
        "covariance": covariance,
        "train_reduced": reduced,
        "pca_components": max_components,
        "explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
    }


def mahalanobis_distance(
    reference: dict,
    query_embeddings: np.ndarray,
) -> np.ndarray:
    scaled = reference["scaler"].transform(query_embeddings)
    reduced = reference["pca"].transform(scaled)
    return np.sqrt(
        np.clip(
            reference["covariance"].mahalanobis(reduced),
            0.0,
            None,
        )
    )


def process_fold(
    project_root: Path,
    fold: int,
    device_name: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    pca_components: int,
    knn_k: int,
) -> Path:
    from gbm_ai.data.classification_dataset import create_dataloader
    from gbm_ai.training.device import resolve_device

    root = project_root.resolve()
    device = resolve_device(device_name)

    train_loader = create_dataloader(
        root,
        split="train",
        fold=fold,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        release_name="classification_v1.0",
    )
    val_loader = create_dataloader(
        root,
        split="validation",
        fold=fold,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        release_name="classification_v1.0",
    )

    model = load_fold_model(checkpoint_for_fold(root, fold, seed), device)
    extractor = EfficientNetFeatureExtractor(model).to(device).eval()

    train_z, train_ids, train_targets = extract_embeddings(
        extractor, train_loader, device
    )
    val_z, val_ids, val_targets = extract_embeddings(
        extractor, val_loader, device
    )

    reference = fit_ood_reference(
        train_z,
        pca_components=pca_components,
    )
    mahal = mahalanobis_distance(reference, val_z)
    knn = cosine_knn_distance(train_z, val_z, k=knn_k)

    output_rows = []
    for sid, target, m_dist, k_dist in zip(
        val_ids,
        val_targets,
        mahal,
        knn,
    ):
        output_rows.append(
            {
                "sample_id": sid,
                "fold": fold,
                "target": int(target),
                "mahalanobis_distance": float(m_dist),
                "cosine_knn_distance": float(k_dist),
            }
        )

    output_dir = root / "artifacts" / "safety" / "ood_embeddings"
    per_fold_dir = output_dir / "per_fold"
    per_fold_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        per_fold_dir / f"fold{fold}_ood_scores.csv",
        output_rows,
    )

    fold_summary = {
        "phase": "Phase 3 Step 3",
        "fold": fold,
        "train_embeddings": int(len(train_ids)),
        "validation_embeddings": int(len(val_ids)),
        "embedding_dimension_raw": int(train_z.shape[1]),
        "pca_components": int(reference["pca_components"]),
        "pca_explained_variance_ratio_sum": float(
            reference["explained_variance_ratio_sum"]
        ),
        "knn_k": int(min(knn_k, len(train_z))),
        "locked_test_used": False,
        "reference_fit_scope": "training portion of this CV fold only",
    }
    (
        per_fold_dir / f"fold{fold}_ood_reference_summary.json"
    ).write_text(
        json.dumps(fold_summary, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 3 STEP 3 — OOD EMBEDDING SCORES")
    print("=" * 54)
    print(f"Fold:                    {fold}")
    print(f"Training embeddings:     {len(train_ids)}")
    print(f"Validation embeddings:   {len(val_ids)}")
    print(f"Raw embedding dimension: {train_z.shape[1]}")
    print(f"PCA components:          {reference['pca_components']}")
    print(
        f"PCA variance retained:   "
        f"{reference['explained_variance_ratio_sum']:.4f}"
    )
    print("Locked test used:        NO")
    print(f"Saved:                   {per_fold_dir}")

    return per_fold_dir / f"fold{fold}_ood_scores.csv"


def aggregate(
    project_root: Path,
    folds: list[int],
) -> Path:
    from gbm_ai.data.classification_dataset import verify_frozen_release

    root = project_root.resolve()
    output_dir = root / "artifacts" / "safety" / "ood_embeddings"
    per_fold_dir = output_dir / "per_fold"

    combined = []
    for fold in folds:
        path = per_fold_dir / f"fold{fold}_ood_scores.csv"
        combined.extend(read_csv(path))

    manifest_path, _ = verify_frozen_release(
        root,
        "classification_v1.0",
    )
    release_rows = read_csv(manifest_path)

    expected_ids = {
        row["sample_id"]
        for row in release_rows
        if row["holdout_split"].strip().lower() == "development"
        and int(row["cv_fold"]) in set(folds)
    }
    actual_ids = [row["sample_id"] for row in combined]

    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError("Duplicate OOD OOF sample IDs.")
    if set(actual_ids) != expected_ids:
        raise RuntimeError(
            f"OOD OOF coverage mismatch: expected {len(expected_ids)}, "
            f"got {len(set(actual_ids))}"
        )

    mahal = np.asarray(
        [float(row["mahalanobis_distance"]) for row in combined],
        dtype=np.float64,
    )
    knn = np.asarray(
        [float(row["cosine_knn_distance"]) for row in combined],
        dtype=np.float64,
    )

    reference = {
        "method": "OOF_internal_embedding_distance_reference",
        "mahalanobis_q90": float(np.quantile(mahal, 0.90)),
        "mahalanobis_q95": float(np.quantile(mahal, 0.95)),
        "mahalanobis_q99": float(np.quantile(mahal, 0.99)),
        "cosine_knn_q90": float(np.quantile(knn, 0.90)),
        "cosine_knn_q95": float(np.quantile(knn, 0.95)),
        "cosine_knn_q99": float(np.quantile(knn, 0.99)),
        "important_limitation": (
            "This is an internal distribution-distance reference derived from "
            "GBM/no-GBM OOF development data. Without a separate external OOD "
            "dataset, it is an OOD-likeness signal, not a validated OOD detector."
        ),
    }

    enriched = []
    for row in combined:
        m = float(row["mahalanobis_distance"])
        k = float(row["cosine_knn_distance"])

        both_q95 = (
            m >= reference["mahalanobis_q95"]
            and k >= reference["cosine_knn_q95"]
        )
        either_q99 = (
            m >= reference["mahalanobis_q99"]
            or k >= reference["cosine_knn_q99"]
        )
        candidate = both_q95 or either_q99

        enriched.append(
            {
                **row,
                "mahalanobis_above_q95": (
                    m >= reference["mahalanobis_q95"]
                ),
                "knn_above_q95": (
                    k >= reference["cosine_knn_q95"]
                ),
                "mahalanobis_above_q99": (
                    m >= reference["mahalanobis_q99"]
                ),
                "knn_above_q99": (
                    k >= reference["cosine_knn_q99"]
                ),
                "ood_likeness_candidate": candidate,
            }
        )

    write_csv(
        output_dir / "oof_ood_scores.csv",
        enriched,
    )

    candidates = [
        row for row in enriched
        if str(row["ood_likeness_candidate"]).lower() == "true"
    ]
    if candidates:
        write_csv(
            output_dir / "ood_review_candidates.csv",
            candidates,
        )

    summary = {
        "phase": "Phase 3 Step 3",
        "status": "COMPLETE",
        "folds": folds,
        "oof_samples": len(enriched),
        "locked_test_used": False,
        "ood_signal": (
            "Fold-specific EfficientNetV2-S penultimate embeddings -> "
            "training-only StandardScaler/PCA/LedoitWolf Mahalanobis + "
            "cosine kNN distance"
        ),
        "reference_quantiles": reference,
        "ood_likeness_candidate_rule": (
            "(Mahalanobis >= q95 AND cosine-kNN >= q95) "
            "OR either signal >= q99"
        ),
        "ood_likeness_candidate_count": len(candidates),
        "validated_external_ood_detector": False,
        "important_note": (
            "Phase 3 Step 4 will fuse this signal with calibrated probability, "
            "TTA uncertainty and QC into an indeterminate decision policy."
        ),
    }
    (
        output_dir / "phase3_step3_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (
        output_dir / "ood_reference.json"
    ).write_text(
        json.dumps(reference, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 3 STEP 3 — OOD-LIKENESS AGGREGATION")
    print("=" * 58)
    print(f"Folds aggregated:         {folds}")
    print(f"OOF samples:              {len(enriched)}")
    print(
        f"Mahalanobis q95:          "
        f"{reference['mahalanobis_q95']:.4f}"
    )
    print(
        f"Cosine-kNN q95:           "
        f"{reference['cosine_knn_q95']:.4f}"
    )
    print(
        f"OOD-likeness candidates:  {len(candidates)}"
    )
    print("Locked test used:         NO")
    print("External OOD validated:   NO")
    print(
        "Status: internal OOD-likeness signal ready for safety fusion"
    )

    return output_dir / "phase3_step3_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root)
    if args.aggregate_only:
        aggregate(root, args.folds)
        return

    for fold in args.folds:
        process_fold(
            root,
            fold,
            args.device,
            args.seed,
            args.batch_size,
            args.num_workers,
            args.pca_components,
            args.knn_k,
        )


if __name__ == "__main__":
    main()
