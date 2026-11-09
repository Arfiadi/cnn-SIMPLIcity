"""Transfer learning model builders using torchvision.models."""

from __future__ import annotations

import logging

import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


def build_transfer_model(
    model_name: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a transfer learning model with modified classification head.

    Args:
        model_name: One of 'resnet18', 'mobilenet_v3_small', 'efficientnet_b0'.
        num_classes: Target number of classes (default 10).
        pretrained: Whether to load ImageNet pre-trained weights.
        freeze_backbone: Whether to freeze feature extractor layers.

    Returns:
        nn.Module: Configured PyTorch model.
    """
    name = model_name.lower().strip()

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if name in {"mobilenet_v3_small", "mobilenet_v3"}:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    if name in {"efficientnet_b0", "efficientnet"}:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported transfer learning backbone: '{model_name}'. "
        f"Choose from: ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']"
    )
