"""Baseline CNN architecture directly matching Slide 6 of PENS SIMPLIcity lecture assignment."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimplicityBaselineCNN(nn.Module):
    """Baseline 2-layer ConvNet as shown in Slide 6 of the course PDF.

    Architecture:
        - Input: (B, 3, 128, 128)
        - 1st Conv: Conv2d(3, 64, 3x3, pad=1) + ReLU + MaxPool2d(2x2) -> (B, 64, 64, 64)
        - 2nd Conv: Conv2d(64, 128, 3x3, pad=1) + ReLU + MaxPool2d(2x2) -> (B, 128, 32, 32)
        - Flatten -> (B, 128 * 32 * 32) = (B, 131072)
        - Hidden FC: Linear(131072, 128) + Activation (ReLU or Sigmoid)
        - Output FC: Linear(128, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_size: tuple[int, int] = (128, 128),
        hidden_dim: int = 128,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        act_fn = nn.Sigmoid() if activation.lower() == "sigmoid" else nn.ReLU(inplace=True)

        self.features = nn.Sequential(
            # 1st Convolution Layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 2nd Convolution Layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        h_out = input_size[0] // 4
        w_out = input_size[1] // 4
        flat_dim = 128 * h_out * w_out

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits
