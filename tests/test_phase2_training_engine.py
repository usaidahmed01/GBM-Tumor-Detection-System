from __future__ import annotations

import math

from gbm_ai.training.metrics import binary_metrics


def test_binary_metrics_known_case():
    m = binary_metrics(
        targets=[0, 0, 1, 1],
        probabilities=[0.1, 0.8, 0.7, 0.9],
        threshold=0.5,
    )
    assert m["tp"] == 2
    assert m["tn"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 0
    assert math.isclose(m["recall_sensitivity"], 1.0)
    assert math.isclose(m["specificity"], 0.5)
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_binary_metrics_threshold_changes_predictions():
    low = binary_metrics([0, 1], [0.4, 0.6], threshold=0.5)
    high = binary_metrics([0, 1], [0.4, 0.6], threshold=0.7)

    assert low["tp"] == 1
    assert high["tp"] == 0
    assert high["fn"] == 1
