"""Transfer learning model builders using torchvision.models."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)

# Registry of supported pretrained backbones
SUPPORTED_BACKBONES: list[str] = [
    "resnet18",
    "resnet34",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "convnext_tiny",
]


class GeMPool2d(nn.Module):
    """Generalized Mean (GeM) Pooling 2D layer.

    Computes: f(x) = (1/|x| * sum(x^p))^(1/p)
    Emphasizes salient localized peak activations over diffuse homogeneous backgrounds.
    When p=1, it is identical to standard average pooling.
    When p -> infinity, it approaches global max pooling.

    Args:
        p: Initial power value (default 3.0).
        eps: Small numerical stabilizer epsilon.
        learnable: Whether power p is a trainable parameter.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = False) -> None:
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.tensor(p, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = x.clamp(min=self.eps)
        x_pow = x_clamped.pow(self.p)
        pooled = F.adaptive_avg_pool2d(x_pow, (1, 1))
        return pooled.pow(1.0 / self.p)

    def __repr__(self) -> str:
        p_val = self.p.item() if isinstance(self.p, torch.Tensor) else float(self.p)
        return f"GeMPool2d(p={p_val:.2f}, eps={self.eps})"


def freeze_backbone_batchnorm(model: nn.Module) -> None:
    """Freeze BatchNorm layers to preserve ImageNet running statistics.

    When fine-tuning with frozen backbone or small datasets, BatchNorm running_mean
    and running_var must NOT be updated by small-batch statistics. This sets all
    BatchNorm2d / BatchNorm1d modules in the backbone to eval mode and disables
    their parameter gradients.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def get_parameter_groups(
    model: nn.Module,
    base_lr: float,
    backbone_lr_ratio: float = 0.1,
    weight_decay: float = 1e-4,
) -> list[dict[str, Any]]:
    """Split model parameters into backbone (low LR) and head (base LR) groups.

    Identifies head parameters by checking for common classifier attribute names
    used across torchvision architectures: ``fc.``, ``classifier.``, ``head.``.

    Args:
        model: PyTorch model (unwrapped from DataParallel if needed).
        base_lr: Learning rate for the classification head.
        backbone_lr_ratio: Multiplier for backbone LR (e.g. 0.1 means 10x lower).
        weight_decay: L2 penalty applied to both groups.

    Returns:
        List of parameter group dicts suitable for ``torch.optim`` constructors.
    """
    head_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []
    head_keywords = {"fc.", "classifier.", "head."}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in head_keywords):
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups: list[dict[str, Any]] = []
    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": base_lr * backbone_lr_ratio,
                "weight_decay": weight_decay,
            }
        )
    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
        )

    logger.debug(
        "Parameter groups: backbone=%d params (lr=%.6f), head=%d params (lr=%.6f)",
        sum(p.numel() for p in backbone_params),
        base_lr * backbone_lr_ratio,
        sum(p.numel() for p in head_params),
        base_lr,
    )
    return groups


def build_transfer_model(
    model_name: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    head_type: str = "standard",
    dropout_rate: float = 0.25,
    pooling: str = "avg",
    freeze_bn: bool = False,
) -> nn.Module:
    """Build a transfer learning model with modified classification head.

    Args:
        model_name: One of :data:`SUPPORTED_BACKBONES`.
        num_classes: Target number of classes (default 10).
        pretrained: Whether to load ImageNet pre-trained weights.
        freeze_backbone: Whether to freeze feature extractor layers.
        head_type: Classification head architecture ("standard", "linear", or "enhanced").
        dropout_rate: Dropout probability for classification head (default 0.25).
        pooling: Spatial pooling mode ("avg" for standard AdaptiveAvgPool2d or "gem" for GeMPool2d).
        freeze_bn: Whether to freeze BatchNorm statistics (eval mode) during fine-tuning.

    Returns:
        nn.Module: Configured PyTorch model ready for fine-tuning.
    """
    name = model_name.lower().strip()

    # --- ResNet Family ---
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            freeze_backbone_batchnorm(model)
        elif freeze_bn:
            freeze_backbone_batchnorm(model)
        if pooling.lower() == "gem":
            model.avgpool = GeMPool2d(p=3.0)
        in_features = model.fc.in_features
        if head_type.lower() == "enhanced":
            model.fc = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes),
            )
        else:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        return model

    if name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            freeze_backbone_batchnorm(model)
        elif freeze_bn:
            freeze_backbone_batchnorm(model)
        if pooling.lower() == "gem":
            model.avgpool = GeMPool2d(p=3.0)
        in_features = model.fc.in_features
        if head_type.lower() == "enhanced":
            model.fc = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes),
            )
        else:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        return model

    # --- MobileNetV3 Family ---
    if name in {"mobilenet_v3_small", "mobilenet_v3"}:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            freeze_backbone_batchnorm(model)
        elif freeze_bn:
            freeze_backbone_batchnorm(model)
        if pooling.lower() == "gem":
            model.avgpool = GeMPool2d(p=3.0)
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout_rate)
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    if name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            freeze_backbone_batchnorm(model)
        elif freeze_bn:
            freeze_backbone_batchnorm(model)
        if pooling.lower() == "gem":
            model.avgpool = GeMPool2d(p=3.0)
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout_rate)
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    # --- EfficientNet-B0 ---
    if name in {"efficientnet_b0", "efficientnet"}:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            freeze_backbone_batchnorm(model)
        elif freeze_bn:
            freeze_backbone_batchnorm(model)

        if pooling.lower() == "gem":
            logger.info("Replacing EfficientNet-B0 avgpool with GeMPool2d(p=3.0)")
            model.avgpool = GeMPool2d(p=3.0)

        in_features = model.classifier[1].in_features
        if head_type.lower() == "enhanced":
            model.classifier = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.SiLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes),
            )
        else:
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        return model

    # --- ConvNeXt-Tiny ---
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        if pooling.lower() == "gem":
            model.avgpool = GeMPool2d(p=3.0)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported transfer learning backbone: '{model_name}'. "
        f"Choose from: {SUPPORTED_BACKBONES}"
    )
