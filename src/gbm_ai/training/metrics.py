from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    targets: Iterable[float],
    probabilities: Iterable[float],
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(list(targets), dtype=np.int64)
    y_prob = np.asarray(list(probabilities), dtype=np.float64)

    if y_true.size == 0:
        raise ValueError("No targets supplied.")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])

        if not np.any(mask):
            continue

        bin_accuracy = float(np.mean(y_true[mask]))
        bin_confidence = float(np.mean(y_prob[mask]))
        ece += (np.sum(mask) / y_true.size) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def binary_metrics(
    targets: Iterable[float],
    probabilities: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(list(targets), dtype=np.int64)
    y_prob = np.asarray(list(probabilities), dtype=np.float64)

    if y_true.size == 0:
        raise ValueError("No targets supplied.")
    if y_true.shape != y_prob.shape:
        raise ValueError("targets and probabilities must have the same shape.")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("probabilities must be between 0 and 1.")

    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    npv = tn / (tn + fn) if (tn + fn) else math.nan

    has_both_classes = len(np.unique(y_true)) == 2
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else math.nan
    pr_auc = (
        float(average_precision_score(y_true, y_prob))
        if has_both_classes
        else math.nan
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "negative_predictive_value": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "expected_calibration_error_10bin": expected_calibration_error(
            y_true, y_prob, n_bins=10
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def bootstrap_metric_ci(
    targets: Iterable[float],
    probabilities: Iterable[float],
    metric: str = "roc_auc",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    y_true = np.asarray(list(targets), dtype=np.int64)
    y_prob = np.asarray(list(probabilities), dtype=np.float64)

    if y_true.shape != y_prob.shape or y_true.size == 0:
        raise ValueError("Invalid targets/probabilities.")
    if metric not in {"roc_auc", "pr_auc"}:
        raise ValueError("metric must be roc_auc or pr_auc")

    rng = np.random.default_rng(seed)
    scores: list[float] = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, y_true.size, size=y_true.size)
        bt = y_true[indices]
        bp = y_prob[indices]

        if len(np.unique(bt)) < 2:
            continue

        if metric == "roc_auc":
            score = roc_auc_score(bt, bp)
        else:
            score = average_precision_score(bt, bp)
        scores.append(float(score))

    if not scores:
        raise RuntimeError("No valid bootstrap samples contained both classes.")

    alpha = 1.0 - confidence
    return {
        "metric": metric,
        "confidence": confidence,
        "n_requested": n_bootstrap,
        "n_valid": len(scores),
        "lower": float(np.quantile(scores, alpha / 2)),
        "upper": float(np.quantile(scores, 1 - alpha / 2)),
        "median": float(np.median(scores)),
    }
