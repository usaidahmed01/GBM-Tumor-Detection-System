from __future__ import annotations

import torch

from gbm_ai.models.comparison_models import (
    GBMBenchmarkModel,
    SUPPORTED_BASELINES,
)


def test_efficientnet_b0_binary_head_and_freeze():
    model = GBMBenchmarkModel("efficientnet_b0", pretrained=False, freeze_backbone=True)
    assert model.model.classifier[-1].out_features == 1
    assert not any(p.requires_grad for p in model.features.parameters())
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_convnext_tiny_binary_head_and_freeze():
    model = GBMBenchmarkModel("convnext_tiny", pretrained=False, freeze_backbone=True)
    assert model.model.classifier[-1].out_features == 1
    assert not any(p.requires_grad for p in model.features.parameters())
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_supported_baselines_are_exactly_the_frozen_comparison_models():
    assert SUPPORTED_BASELINES == ("efficientnet_b0", "convnext_tiny")
