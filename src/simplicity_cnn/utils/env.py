"""Environment and device management utilities (path-agnostic and device-agnostic)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

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


def _find_dataset_root(search_dir: Path | str) -> Path | None:
    """Recursively search for directory containing 'train' and 'validation'/'test' subfolders.

    Uses os.walk(followlinks=True) to reliably follow symlinks on Kaggle and handles
    case-insensitive directory naming.
    """
    base = Path(search_dir)
    if not base.exists():
        return None
    try:
        for root, dirs, _ in os.walk(base, followlinks=True):
            parts = Path(root).parts
            if any(
                p.startswith(".") or p in {"__pycache__", "site-packages", "node_modules"}
                for p in parts
            ):
                continue
            dirs_lower = {d.lower(): d for d in dirs}
            if "train" in dirs_lower:
                if any(k in dirs_lower for k in ["validation", "val", "valid", "test"]):
                    return Path(root).resolve()
    except Exception as err:
        logger.debug("Error while scanning %s: %s", search_dir, err)
    return None


def get_default_data_dir() -> Path:
    """Locate the SIMPLIcity dataset directory across local, Colab, or Kaggle.

    Checks:
      1. Environment variables 'DATA_DIR' or 'SIMPLICITY_DATA_DIR'.
      2. Dynamic os.walk scan of '/kaggle/input' (follows Kaggle symlinks and casing).
      3. Candidate path list across typical local, Colab, and Kaggle mount paths.
      4. Dynamic os.walk scan of '/content' (Google Colab).
      5. Dynamic os.walk scan under repository 'Dataset' folder and current working directory.

    Returns:
        Path: Resolved absolute path to the dataset directory containing train/val/test.
    """
    # 1. Environment variable check
    env_dir = os.environ.get("DATA_DIR") or os.environ.get("SIMPLICITY_DATA_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
        if p.exists() and any((p / sub).exists() for sub in ["train", "Train"]):
            logger.info("Using dataset directory from environment variable: %s", p)
            return p

    # 2. Dynamic scan under /kaggle/input (handles Kaggle symlinks, any dataset name or case)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        detected_kaggle = _find_dataset_root(kaggle_input)
        if detected_kaggle is not None:
            logger.info("Auto-detected Kaggle dataset: %s", detected_kaggle)
            return detected_kaggle

    repo_root = Path(__file__).resolve().parents[3]

    # 3. Known candidate paths
    candidate_paths: list[Path] = [
        # Local workspace paths
        repo_root / "Dataset" / "SIMPLIcity",
        repo_root / "Dataset" / "SIMPLIcity" / "SIMPLIcity Dataset",
        repo_root / "Dataset" / "SIMPLIcity Dataset",
        repo_root / "Dataset",
        # Kaggle input known paths (case-sensitive variants)
        Path("/kaggle/input/SIMPLIcity"),
        Path("/kaggle/input/simplicity"),
        Path("/kaggle/input/simplicity-dataset"),
        Path("/kaggle/input/simplicity/SIMPLIcity"),
        Path("/kaggle/input/simplicity-dataset/SIMPLIcity"),
        Path("/kaggle/input/datasets/arfiadi/simplicity"),
        # Colab typical paths
        Path("/content/Dataset/SIMPLIcity"),
        Path("/content/Dataset"),
        Path("/content/data"),
        # Current working directory relative paths
        Path.cwd() / "Dataset" / "SIMPLIcity",
        Path.cwd() / "Dataset",
    ]

    for path in candidate_paths:
        if path.exists() and any((path / sub).exists() for sub in ["train", "Train"]):
            logger.info("Auto-detected dataset directory: %s", path)
            return path.resolve()

    # 4. Dynamic scan under repository 'Dataset' directory
    dataset_base = repo_root / "Dataset"
    if dataset_base.exists():
        detected_local = _find_dataset_root(dataset_base)
        if detected_local is not None:
            logger.info("Auto-detected local dataset: %s", detected_local)
            return detected_local

    # 5. Dynamic scan under Colab /content
    colab_content = Path("/content")
    if colab_content.exists():
        detected_colab = _find_dataset_root(colab_content)
        if detected_colab is not None:
            logger.info("Auto-detected Colab dataset: %s", detected_colab)
            return detected_colab

    # 6. Dynamic scan under current working directory
    detected_cwd = _find_dataset_root(Path.cwd())
    if detected_cwd is not None:
        logger.info("Auto-detected dataset in working directory: %s", detected_cwd)
        return detected_cwd

    # Fallback default
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


def get_cuda_device_info() -> dict[str, Any]:
    """Retrieve detailed CUDA device topology for experiment logging.

    Returns:
        dict with keys: available, count, devices (list of name strings),
        is_multi_gpu (bool).
    """
    if not torch.cuda.is_available():
        return {"available": False, "count": 0, "devices": [], "is_multi_gpu": False}
    count = torch.cuda.device_count()
    devices: list[str] = []
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        mem_bytes = getattr(props, "total_memory", getattr(props, "total_mem", 0))
        mem_gb = mem_bytes / (1024**3)
        devices.append(f"{name} ({mem_gb:.1f} GB)")
    return {
        "available": True,
        "count": count,
        "devices": devices,
        "is_multi_gpu": count > 1,
    }
