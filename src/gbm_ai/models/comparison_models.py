from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    convnext_tiny,
    efficientnet_b0,
)

SUPPORTED_BASELINES = ("efficientnet_b0", "convnext_tiny")


class GBMBenchmarkModel(nn.Module):
    """Binary GBM wrapper for supported TorchVision comparison architectures."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        model_name = model_name.lower().strip()
        if model_name not in SUPPORTED_BASELINES:
            raise ValueError(
                f"Unsupported comparison model {model_name!r}. "
                f"Choose one of {SUPPORTED_BASELINES}."
            )

        self.model_name = model_name

        if model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = efficientnet_b0(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, 1),
            )

        elif model_name == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.model = convnext_tiny(weights=weights)
            in_features = self.model.classifier[2].in_features
            # Keep ConvNeXt's normalization + flattening and replace only
            # the final ImageNet classifier with our binary GBM head.
            self.model.classifier = nn.Sequential(
                self.model.classifier[0],
                self.model.classifier[1],
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1),
            )

        if freeze_backbone:
            self.freeze_backbone()

    @property
    def features(self) -> nn.Module:
        return self.model.features

    @property
    def classifier(self) -> nn.Module:
        return self.model.classifier

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_last_feature_blocks(self, count: int = 3) -> None:
        self.freeze_backbone()
        blocks = list(self.features.children())
        if count <= 0:
            return
        for block in blocks[-count:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "total_parameters": self.total_parameter_count(),
            "trainable_parameters_current": self.trainable_parameter_count(),
        }
