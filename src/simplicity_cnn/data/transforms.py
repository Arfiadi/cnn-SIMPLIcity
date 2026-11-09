"""Image transformation pipelines using torchvision transforms."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

try:
    from torchvision.transforms import v2

    HAS_V2 = True
except ImportError:
    import torchvision.transforms as v2  # type: ignore

    HAS_V2 = False


class LetterboxPad(nn.Module):
    """Resize an image preserving its aspect ratio and symmetrically pad to target (H, W).

    This prevents geometric distortion and eliminates the cutting off of tall objects
    (e.g. building towers/spires) or wide landscapes (e.g. beaches/mountains).
    """

    def __init__(
        self,
        target_size: int | tuple[int, int] = (240, 240),
        fill: int = 0,
    ) -> None:
        super().__init__()
        self.target_size = (
            (target_size, target_size)
            if isinstance(target_size, int)
            else (int(target_size[0]), int(target_size[1]))
        )
        self.fill = fill

    def forward(self, img: Any) -> Any:
        target_h, target_w = self.target_size
        if isinstance(img, Image.Image):
            w, h = img.size
        elif isinstance(img, torch.Tensor):
            h, w = img.shape[-2], img.shape[-1]
        else:
            return img

        scale = min(target_h / max(h, 1), target_w / max(w, 1))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        resized = TF.resize(img, [new_h, new_w], antialias=True)
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return TF.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)


def build_transforms(
    img_size: int | tuple[int, int] = (128, 128),
    augment: bool = True,
    mean: tuple[float, float, float] | list[float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] | list[float] = (0.229, 0.224, 0.225),
    flip_prob: float = 0.5,
    rotation_deg: int = 10,
    color_jitter_params: tuple[float, float, float] | list[float] | None = (0.1, 0.1, 0.1),
    crop_scale: tuple[float, float] | list[float] | None = None,
    random_erasing_prob: float = 0.0,
    preserve_aspect_ratio: bool = False,
) -> Callable:
    """Build train or validation/test transformation pipeline with EDA mitigations.

    Args:
        img_size: Target image size (H, W) or single int for square.
        augment: If True, apply training augmentations. If False, apply deterministic resize and normalization only.
        mean: Normalization RGB mean.
        std: Normalization RGB standard deviation.
        flip_prob: Horizontal flip probability.
        rotation_deg: Rotation degree range.
        color_jitter_params: Tuple of (brightness, contrast, saturation) jitter factors.
        crop_scale: Optional scale range for RandomResizedCrop to perturb aspect ratio.
        random_erasing_prob: Probability of random erasing to destroy uniform backgrounds.
        preserve_aspect_ratio: If True, uses LetterboxPad to preserve 100% of image geometry without cropping.

    Returns:
        Callable: Composed transformation pipeline.
    """
    target_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
    mean_tuple = tuple(mean)
    std_tuple = tuple(std)

    if augment:
        ops = []
        if HAS_V2:
            ops.append(v2.ToImage())

            # Geometrical: Letterbox vs Crop vs Resize
            if preserve_aspect_ratio:
                ops.append(LetterboxPad(target_size=target_size, fill=0))
            elif crop_scale and crop_scale[0] < 1.0:
                ops.append(v2.RandomResizedCrop(target_size, scale=tuple(crop_scale), antialias=True))
            else:
                ops.append(v2.Resize(target_size, antialias=True))

            if flip_prob > 0:
                ops.append(v2.RandomHorizontalFlip(p=flip_prob))

            if rotation_deg > 0:
                ops.append(v2.RandomRotation(degrees=rotation_deg))

            # Photometric: Color jitter to mitigate brightness shortcut
            if color_jitter_params and any(v > 0 for v in color_jitter_params):
                ops.append(
                    v2.ColorJitter(
                        brightness=color_jitter_params[0],
                        contrast=color_jitter_params[1],
                        saturation=color_jitter_params[2] if len(color_jitter_params) > 2 else 0.0,
                    )
                )

            ops.append(v2.ToDtype(torch.float32, scale=True))
            ops.append(v2.Normalize(mean=mean_tuple, std=std_tuple))

            # Random erasing for solid background perturbation
            if random_erasing_prob > 0:
                ops.append(v2.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.20)))

            return v2.Compose(ops)

        # Fallback for v1
        if preserve_aspect_ratio:
            ops.append(LetterboxPad(target_size=target_size, fill=0))
        elif crop_scale and crop_scale[0] < 1.0:
            ops.append(v2.RandomResizedCrop(target_size, scale=tuple(crop_scale)))
        else:
            ops.append(v2.Resize(target_size))

        if flip_prob > 0:
            ops.append(v2.RandomHorizontalFlip(p=flip_prob))
        if rotation_deg > 0:
            ops.append(v2.RandomRotation(degrees=rotation_deg))

        if color_jitter_params and any(v > 0 for v in color_jitter_params):
            ops.append(
                v2.ColorJitter(
                    brightness=color_jitter_params[0],
                    contrast=color_jitter_params[1],
                    saturation=color_jitter_params[2] if len(color_jitter_params) > 2 else 0.0,
                )
            )

        ops.append(v2.ToTensor())
        ops.append(v2.Normalize(mean=mean_tuple, std=std_tuple))
        if random_erasing_prob > 0:
            ops.append(v2.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.20)))

        return v2.Compose(ops)

    # Validation / Test transform:
    # If preserve_aspect_ratio is True: use LetterboxPad
    # Otherwise: deterministic resize of shorter edge followed by CenterCrop
    if preserve_aspect_ratio:
        if HAS_V2:
            return v2.Compose(
                [
                    v2.ToImage(),
                    LetterboxPad(target_size=target_size, fill=0),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=mean_tuple, std=std_tuple),
                ]
            )
        return v2.Compose(
            [
                LetterboxPad(target_size=target_size, fill=0),
                v2.ToTensor(),
                v2.Normalize(mean=mean_tuple, std=std_tuple),
            ]
        )

    short_edge = min(target_size)
    if HAS_V2:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(short_edge, antialias=True),
                v2.CenterCrop(target_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean_tuple, std=std_tuple),
            ]
        )
    return v2.Compose(
        [
            v2.Resize(short_edge),
            v2.CenterCrop(target_size),
            v2.ToTensor(),
            v2.Normalize(mean=mean_tuple, std=std_tuple),
        ]
    )
