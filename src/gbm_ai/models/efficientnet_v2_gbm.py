from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class GBMEfficientNetV2S(nn.Module):
    """EfficientNetV2-S adapted to one GBM logit."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_v2_s(weights=weights)

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 1),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.model.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.model.features.parameters():
            p.requires_grad = True

    def unfreeze_last_feature_blocks(self, count: int = 2) -> None:
        self.freeze_backbone()
        blocks = list(self.model.features.children())
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
