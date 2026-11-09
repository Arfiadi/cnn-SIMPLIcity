"""Improved Custom CNN from scratch with Batch Normalization, Dropout, and Adaptive Pooling."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d -> BatchNorm2d -> LeakyReLU -> Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d -> Dropout2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimplicityCustomCNN(nn.Module):
    """Custom deep CNN designed from scratch with modern regularization for small datasets.

    Mitigates overfitting via:
    - Batch Normalization across all convolutional stages.
    - Dropout in both spatial features and dense classifier.
    - Adaptive Average Pooling to prevent FC layer parameter explosion.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        hidden_dim: int = 128,
        fc_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, dropout_rate=0.10),  # (B, 32, H/2, W/2)
            ConvBlock(32, 64, dropout_rate=0.15),  # (B, 64, H/4, W/4)
            ConvBlock(64, 128, dropout_rate=0.20),  # (B, 128, H/8, W/8)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        flat_dim = 128 * 2 * 2  # 512 units regardless of input size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=fc_dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.global_pool(feat)
        logits = self.classifier(pooled)
        return logits
