"""Environment and device management utilities (path-agnostic and device-agnostic)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def detect_environment() -> str:
    """Detect runtime environment: 'kaggle', 'colab', or 'local'.

    Returns:
        str: Detected environment identifier.
    """
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists():
        return "kaggle"
    if "google.colab" in sys.modules or Path("/content").exists():
        return "colab"
    return "local"


def get_default_data_dir() -> Path:
    """Locate the SIMPLIcity dataset directory across local, Colab, or Kaggle.

    Returns:
        Path: Resolved absolute path to the dataset directory containing train/val/test.
    """
    repo_root = Path(__file__).resolve().parents[3]

    candidate_paths: list[Path] = [
        # Local workspace relative paths
        repo_root / "Dataset" / "SIMPLIcity",
        repo_root / "Dataset" / "SIMPLIcity" / "SIMPLIcity Dataset",
        repo_root / "Dataset" / "SIMPLIcity Dataset",
        repo_root / "Dataset",
        # Colab typical paths
        Path("/content/Dataset/SIMPLIcity"),
        Path("/content/Dataset/SIMPLIcity/SIMPLIcity Dataset"),
        Path("/content/Dataset/SIMPLIcity Dataset"),
        Path("/content/Dataset"),
        Path("/content/data"),
        # Kaggle input paths
        Path("/kaggle/input/simplicity-dataset/SIMPLIcity"),
        Path("/kaggle/input/simplicity-dataset/SIMPLIcity Dataset"),
        Path("/kaggle/input/simplicity-dataset"),
        Path("/kaggle/input/simplicity/SIMPLIcity Dataset"),
    ]

    for path in candidate_paths:
        if path.exists() and (path / "train").exists():
            logger.info("Auto-detected dataset directory: %s", path)
            return path.resolve()

    # Dynamic recursive scan under repo_root / "Dataset" if present
    dataset_base = repo_root / "Dataset"
    if dataset_base.exists():
        for train_dir in dataset_base.rglob("train"):
            parent = train_dir.parent
            if (parent / "validation").exists() or (parent / "test").exists():
                logger.info("Auto-detected dataset via recursive scan: %s", parent)
                return parent.resolve()

    # Fallback to local default even if not yet verified
    default_fallback = repo_root / "Dataset" / "SIMPLIcity"
    logger.warning(
        "Could not auto-detect dataset with 'train/' subfolder. Using default: %s", default_fallback
    )
    return default_fallback


def get_device(preference: str = "auto") -> torch.device:
    """Determine compute device with fallback to CPU.

    Args:
        preference: 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        torch.device: PyTorch device object.
    """
    pref = preference.lower().strip()
    if pref == "auto":
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("CUDA GPU detected: %s. Using cuda.", device_name)
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS detected. Using mps.")
            return torch.device("mps")
        logger.info("No GPU detected. Falling back to CPU.")
        return torch.device("cpu")

    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")
