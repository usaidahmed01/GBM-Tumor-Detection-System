from __future__ import annotations

import numpy as np

from gbm_ai.training.calibrate_thresholds import (
    fit_temperature,
    temperature_scale,
    threshold_candidates,
    threshold_sweep,
)


def test_temperature_scaling_returns_valid_probabilities():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.001, 0.05, 0.20, 0.80, 0.95, 0.999]
    t = fit_temperature(y, p)
    calibrated = temperature_scale(p, t)
    assert t > 0
    assert np.all(calibrated > 0)
    assert np.all(calibrated < 1)


def test_threshold_sweep_has_99_rows_and_safety_metrics():
    rows = threshold_sweep(
        [0, 0, 0, 1, 1, 1],
        [0.05, 0.15, 0.40, 0.55, 0.75, 0.95],
    )
    assert len(rows) == 99
    assert all("sensitivity" in r and "specificity" in r for r in rows)
    assert all("fn" in r for r in rows)


def test_threshold_candidates_created_for_separable_data():
    sweep = threshold_sweep(
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95],
    )
    candidates = threshold_candidates(sweep)
    roles = {r["candidate_role"] for r in candidates}
    assert "T_low_candidate" in roles
    assert "T_high_candidate" in roles
