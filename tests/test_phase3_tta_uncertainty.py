from __future__ import annotations

from PIL import Image

from gbm_ai.safety.tta_uncertainty import (
    TTA_SPECS,
    apply_tta,
    calibrated_probability,
    summarize_tta_probabilities,
    three_band_state,
)


def test_tta_set_is_deterministic_and_conservative():
    assert len(TTA_SPECS) == 7
    names = [spec.name for spec in TTA_SPECS]
    assert len(names) == len(set(names))
    assert "identity" in names

    image = Image.new("RGB", (384, 384), color=(100, 100, 100))
    out = apply_tta(image, TTA_SPECS[1])
    assert out.size == (384, 384)


def test_temperature_calibration_stays_in_probability_range():
    for probability in (0.01, 0.25, 0.5, 0.9, 0.99):
        calibrated = calibrated_probability(probability, temperature=1.5)
        assert 0.0 < calibrated < 1.0


def test_tta_summary_detects_three_band_instability():
    summary = summarize_tta_probabilities(
        [0.10, 0.12, 0.14, 0.20, 0.60, 0.11, 0.15],
        t_low=0.13,
        t_high=0.57,
    )
    assert summary["band_instability"] is True
    assert summary["crosses_T_low"] is True
    assert summary["crosses_T_high"] is True
    assert summary["tta_unique_state_count"] == 3


def test_stable_case_remains_single_band():
    summary = summarize_tta_probabilities(
        [0.80, 0.82, 0.79, 0.84, 0.81, 0.83, 0.80],
        t_low=0.13,
        t_high=0.57,
    )
    assert summary["band_instability"] is False
    assert summary["mean_three_band_state"] == "GBM_SUSPECTED"
    assert summary["probability_std"] < 0.03
