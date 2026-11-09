"""SIMPLIcity Dataset implementation for PyTorch."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = [
    "People",  # Class 0: 0 - 99
    "Beach",  # Class 1: 100 - 199
    "Building",  # Class 2: 200 - 299
    "Bus",  # Class 3: 300 - 399
    "Dinosaur",  # Class 4: 400 - 499
    "Elephant",  # Class 5: 500 - 599
    "Flower",  # Class 6: 600 - 699
    "Horse",  # Class 7: 700 - 799
    "Mountain",  # Class 8: 800 - 899
    "Food",  # Class 9: 900 - 999
]

CLASS_NAME_TO_IDX: dict[str, int] = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}


class SIMPLIcityDataset(Dataset):
    """PyTorch Dataset for SIMPLIcity 1,000 images across 10 semantic classes.

    Supports both flat numeric image files (e.g. 0.jpg - 999.jpg with class = num // 100)
    and class-named subdirectories (e.g. train/Beach/101.jpg).
    """

    def __init__(
        self,
        root_dir: Path | str,
        split: str = "train",
        transform: Callable | None = None,
        sample_size: int | None = None,
    ) -> None:
        """Initialize SIMPLIcityDataset.

        Args:
            root_dir: Root dataset directory containing train/validation/test folders.
            split: One of 'train', 'validation', or 'test'.
            transform: Optional transform to apply on PIL image.
            sample_size: Optional integer to take only the first N samples (for CPU smoke testing).
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower().strip()
        self.transform = transform

        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                f"Available in root: {[p.name for p in self.root_dir.iterdir() if p.is_dir()]}"
            )

        self.samples: list[tuple[Path, int]] = self._collect_samples(split_dir)

        if not self.samples:
            raise RuntimeError(f"No valid image files found in {split_dir}")

        if sample_size is not None and sample_size > 0 and sample_size < len(self.samples):
            step = max(len(self.samples) // sample_size, 1)
            self.samples = self.samples[::step][:sample_size]
            logger.info(
                "Subsampled %s split to %d items (smoke-test mode).", self.split, len(self.samples)
            )

    def _collect_samples(self, split_dir: Path) -> list[tuple[Path, int]]:
        """Collect and index image paths with corresponding integer labels."""
        samples: list[tuple[Path, int]] = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        # Case 1: Check for class subfolders (e.g. train/People/0.jpg)
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        matched_subdirs = [d for d in subdirs if d.name.lower() in CLASS_NAME_TO_IDX]

        if matched_subdirs:
            for subdir in sorted(matched_subdirs, key=lambda d: CLASS_NAME_TO_IDX[d.name.lower()]):
                label = CLASS_NAME_TO_IDX[subdir.name.lower()]
                for img_path in sorted(subdir.iterdir()):
                    if img_path.suffix.lower() in valid_extensions:
                        samples.append((img_path, label))
            return samples

        # Case 2: Standard SIMPLIcity numeric naming (e.g. 0.jpg -> class 0, 950.jpg -> class 9)
        for img_path in sorted(
            split_dir.iterdir(), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name
        ):
            if img_path.suffix.lower() in valid_extensions:
                if img_path.stem.isdigit():
                    num = int(img_path.stem)
                    label = num // 100
                    if 0 <= label < len(CLASS_NAMES):
                        samples.append((img_path, label))
                    else:
                        logger.warning(
                            "Image %s has out-of-range class index %d. Skipping.",
                            img_path.name,
                            label,
                        )
                else:
                    logger.warning(
                        "Non-numeric image file %s encountered without class subfolder. Skipping.",
                        img_path.name,
                    )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """Fetch item at index.

        Returns:
            Tuple of (transformed image tensor, label integer, filename string).
        """
        img_path, label = self.samples[idx]

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, img_path.name
