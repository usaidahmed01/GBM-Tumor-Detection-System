from __future__ import annotations

import pytest

from gbm_ai.training.cross_validation import validate_oof_coverage
from gbm_ai.training.metrics import binary_metrics, bootstrap_metric_ci


def release_rows():
    return [
        {"sample_id": "a", "holdout_split": "development", "cv_fold": "0"},
        {"sample_id": "b", "holdout_split": "development", "cv_fold": "1"},
        {"sample_id": "c", "holdout_split": "test", "cv_fold": ""},
    ]


def test_oof_coverage_excludes_locked_test_and_is_complete():
    oof = [
        {"sample_id": "a", "fold": 0},
        {"sample_id": "b", "fold": 1},
    ]
    result = validate_oof_coverage(release_rows(), oof, {0, 1})
    assert result["actual_oof_samples"] == 2
    assert result["locked_test_leakage"] == 0
    assert result["missing_oof_samples"] == 0


def test_oof_coverage_rejects_test_leakage():
    oof = [
        {"sample_id": "a", "fold": 0},
        {"sample_id": "b", "fold": 1},
        {"sample_id": "c", "fold": 1},
    ]
    with pytest.raises(RuntimeError, match="Locked test"):
        validate_oof_coverage(release_rows(), oof, {0, 1})


def test_extended_metrics_and_bootstrap_ci():
    targets = [0, 0, 0, 1, 1, 1]
    probs = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    metrics = binary_metrics(targets, probs, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["expected_calibration_error_10bin"] <= 1.0

    ci = bootstrap_metric_ci(
        targets, probs, metric="roc_auc", n_bootstrap=100, seed=42
    )
    assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0
