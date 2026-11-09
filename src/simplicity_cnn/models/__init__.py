"""Model architectures: baseline, custom scratch, and transfer learning backbones."""

from simplicity_cnn.models.baseline_cnn import SimplicityBaselineCNN
from simplicity_cnn.models.custom_scratch import SimplicityCustomCNN
from simplicity_cnn.models.factory import AVAILABLE_MODELS, build_model
from simplicity_cnn.models.transfer import build_transfer_model

__all__ = [
    "AVAILABLE_MODELS",
    "SimplicityBaselineCNN",
    "SimplicityCustomCNN",
    "build_model",
    "build_transfer_model",
]
