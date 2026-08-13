from __future__ import annotations

import numpy as np
import torch
from torch import nn

from gbm_ai.safety.gradcam import GradCAM
from gbm_ai.safety.review_false_negatives import (
    checkpoint_for_fold,
    select_safety_critical_false_negatives,
)


class TinyBinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 6, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(6, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(-1)


def test_gradcam_returns_normalized_spatial_map():
    torch.manual_seed(42)
    model = TinyBinaryCNN()
    x = torch.randn(1, 3, 32, 32, requires_grad=True)

    with GradCAM(model, model.features[-2]) as gradcam:
        result = gradcam.generate(x)

    assert result.cam.shape == (32, 32)
    assert np.isfinite(result.cam).all()
    assert 0.0 <= float(result.cam.min()) <= 1.0
    assert 0.0 <= float(result.cam.max()) <= 1.0
    assert 0.0 <= result.peak_x_normalized <= 1.0
    assert 0.0 <= result.peak_y_normalized <= 1.0
    assert 0.0 <= result.border_energy_fraction <= 1.0


def test_false_negative_selection_uses_t_low():
    rows = [
        {"sample_id": "a", "target": "1", "probability_gbm_calibrated": "0.12"},
        {"sample_id": "b", "target": "1", "probability_gbm_calibrated": "0.13"},
        {"sample_id": "c", "target": "1", "probability_gbm_calibrated": "0.14"},
        {"sample_id": "d", "target": "0", "probability_gbm_calibrated": "0.05"},
    ]
    selected = select_safety_critical_false_negatives(rows, t_low=0.13)
    assert [row["sample_id"] for row in selected] == ["a", "b"]


def test_checkpoint_routing_is_fold_specific(tmp_path):
    path = checkpoint_for_fold(tmp_path, fold=4, seed=42)
    assert "efficientnetv2s_fold4_seed42" in str(path)
    assert path.name == "best_model.pt"
