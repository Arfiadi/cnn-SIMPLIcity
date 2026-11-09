"""Model factory unifying custom scratch models and pretrained backbones."""

from __future__ import annotations

import logging

import torch.nn as nn

from simplicity_cnn.models.baseline_cnn import SimplicityBaselineCNN
from simplicity_cnn.models.custom_scratch import SimplicityCustomCNN
from simplicity_cnn.models.transfer import build_transfer_model

logger = logging.getLogger(__name__)

AVAILABLE_MODELS: list[str] = [
    "baseline_custom",
    "custom_scratch",
    "resnet18",
    "mobilenet_v3_small",
    "efficientnet_b0",
]


def build_model(
    model_name: str,
    num_classes: int = 10,
    input_size: tuple[int, int] = (128, 128),
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **kwargs,
) -> nn.Module:
    """Build a CNN model instance by name.

    Args:
        model_name: Identifier from AVAILABLE_MODELS.
        num_classes: Number of output logits (default 10).
        input_size: Input spatial dimensions (H, W).
        pretrained: For transfer models, load pretrained ImageNet weights.
        freeze_backbone: For transfer models, freeze backbone weights.
        **kwargs: Additional model parameters.

    Returns:
        nn.Module: Instantiated PyTorch model.
    """
    name = model_name.lower().strip()

    if name == "baseline_custom":
        activation = kwargs.get("activation", "relu")
        logger.info(
            "Instantiating SimplicityBaselineCNN (Slide 6 architecture, activation=%s)", activation
        )
        return SimplicityBaselineCNN(
            num_classes=num_classes, input_size=input_size, activation=activation
        )

    if name in {"custom_scratch", "custom_deep"}:
        logger.info("Instantiating SimplicityCustomCNN (regularized scratch architecture)")
        return SimplicityCustomCNN(num_classes=num_classes)

    if name in {
        "resnet18",
        "mobilenet_v3_small",
        "mobilenet_v3",
        "efficientnet_b0",
        "efficientnet",
    }:
        logger.info(
            "Instantiating transfer model '%s' (pretrained=%s, freeze=%s)",
            name,
            pretrained,
            freeze_backbone,
        )
        return build_transfer_model(
            model_name=name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    raise ValueError(f"Unknown model name: '{model_name}'. Available: {AVAILABLE_MODELS}")
