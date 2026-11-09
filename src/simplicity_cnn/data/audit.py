"""Dataset sanity and integrity audit utilities."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def audit_dataset(data_dir: Path | str) -> dict:
    """Perform a comprehensive integrity and sanity audit on the dataset.

    Args:
        data_dir: Root dataset directory.

    Returns:
        dict: Summary statistics including file counts, corrupt counts, dimensions, and class distributions.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root directory not found: {root}")

    report: dict = {
        "root_dir": str(root.resolve()),
        "splits": {},
        "corrupted_files": [],
        "dimension_stats": {},
        "channel_stats": Counter(),
        "is_healthy": True,
    }

    expected_splits = ["train", "validation", "test"]

    for split in expected_splits:
        split_path = root / split
        if not split_path.exists():
            report["is_healthy"] = False
            report["splits"][split] = {"error": "Directory missing"}
            continue

        images = [p for p in split_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        class_counts: Counter[int | str] = Counter()
        widths: list[int] = []
        heights: list[int] = []

        for img_path in images:
            # Check 0-byte file
            if img_path.stat().st_size == 0:
                report["corrupted_files"].append(str(img_path))
                report["is_healthy"] = False
                continue

            # Verify readability and integrity
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
                    report["channel_stats"][img.mode] += 1
            except Exception as err:
                report["corrupted_files"].append(f"{img_path}: {err}")
                report["is_healthy"] = False
                continue

            # Determine class
            if img_path.stem.isdigit():
                class_counts[int(img_path.stem) // 100] += 1
            elif img_path.parent.name != split:
                class_counts[img_path.parent.name] += 1

        report["splits"][split] = {
            "total_images": len(images),
            "class_distribution": dict(sorted(class_counts.items())),
            "min_dimension": (min(widths), min(heights)) if widths else None,
            "max_dimension": (max(widths), max(heights)) if widths else None,
        }

    return report
