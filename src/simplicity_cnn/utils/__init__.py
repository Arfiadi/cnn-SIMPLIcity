"""Utility modules: environment detection, device resolution, seeding, and logging."""

from simplicity_cnn.utils.env import detect_environment, get_default_data_dir, get_device
from simplicity_cnn.utils.logging_utils import setup_logger
from simplicity_cnn.utils.seed import seed_everything

__all__ = [
    "detect_environment",
    "get_default_data_dir",
    "get_device",
    "seed_everything",
    "setup_logger",
]
